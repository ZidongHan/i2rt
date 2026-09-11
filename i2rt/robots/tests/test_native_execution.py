"""Guarded native execution over fake I/O; never open a CAN interface."""

import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from i2rt.motor_drivers.utils import MotorInfo
from i2rt.robots.joint_reference import AdmittedReference, JerkPiece, JointReference, _same_state
from i2rt.robots.motor_chain_robot import MotorChainRobot, NativeFeedbackLimits


class FakeChain:
    def __init__(self) -> None:
        self.running = True
        self.commands = []
        self.disabled = False
        self.states = [
            MotorInfo(
                id=i + 1,
                error_code="0x1",
                pos=0 if i < 6 else -1,
                temp_mos=25,
                temp_rotor=25,
                received_monotonic=time.monotonic(),
                receive_sequence=i + 1,
            )
            for i in range(7)
        ]

    def __len__(self) -> int:
        return 7

    def read_states(self) -> list[MotorInfo]:
        return self.states

    def set_commands(self, torques: np.ndarray, **kwargs: Any) -> list[MotorInfo]:
        self.commands.append((torques.copy(), kwargs))
        return self.states

    def disable_motors(self) -> list[dict[str, bool]]:
        self.disabled = True
        self.running = False
        return [{"attempted": True, "confirmed": True} for _ in self.states]

    def close(self) -> None:
        self.running = False


def robot_fixture() -> tuple[MotorChainRobot, FakeChain]:
    chain = FakeChain()
    robot = MotorChainRobot(
        chain,
        use_gravity_comp=False,
        joint_limits=np.array([[-3, 3]] * 6),
        kp=[80, 80, 80, 10, 10, 10, 20],
        kd=[5, 5, 5, 1.5, 1.5, 1.5, 0.5],
        gripper_index=6,
        gripper_limits=np.array([0.0, -2.0]),
        start_server_thread=False,
    )
    # Constructor's compatibility wait is not receipt progress. Refresh explicitly.
    chain.states = [replace(m, received_monotonic=time.monotonic()) for m in chain.states]
    return robot, chain


def stationary_packet(sequence: int = 1, horizon: float = 0.02) -> AdmittedReference:
    reference = JointReference(tuple(() for _ in range(7)), (0, 0, 0, 0, 0, 0, 0.5))
    now = time.monotonic()
    return AdmittedReference(sequence, now, now + horizon, reference, reference, (0.05,) * 7, (0.2,) * 7)


def test_scalar_reference_continuity_keeps_numpy_tolerance() -> None:
    random = np.random.default_rng(42)
    for _ in range(3000):
        second = random.uniform(-4, 4, 3)
        first = second + random.uniform(-2, 2, 3) * (1e-8 + 1e-8 * np.abs(second))
        assert _same_state(tuple(first), tuple(second)) == np.allclose(first, second, atol=1e-8, rtol=1e-8)
    assert not _same_state((float("nan"), 0, 0), (0, 0, 0))
    assert not _same_state((float("inf"), 0, 0), (0, 0, 0))


def test_reference_jerk_reports_actual_piece_and_fallback_clock() -> None:
    reference = JointReference(
        ((JerkPiece(1, 0, 0, 0, 1), JerkPiece(2, 1 / 6, 0.5, 1, -1), JerkPiece(1, 11 / 6, 0.5, -1, 1)),), (2.0,)
    )
    packet = AdmittedReference(1, 10, 11, reference, reference.tail(1), (0.05,), (0.2,))
    for elapsed, expected in ((0, 1), (1, -1), (2, -1), (3, 1), (4, 0), (100, 0)):
        assert reference.jerk_at(elapsed) == (expected,)
        assert packet.jerk_at(10 + elapsed) == (expected,)
    with pytest.raises(ValueError, match="finite"):
        reference.jerk_at(float("nan"))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"pos": float("nan")}, "nonfinite"),
        ({"eff": 21.0}, "raw motor effort"),
        ({"temp_mos": 81.0}, "temperature"),
        ({"temp_mos": -1.0}, "unavailable"),
        ({"temp_rotor": float("nan")}, "nonfinite"),
        ({"vel": 4.0}, "joint velocity"),
        ({"sweep_id": 99}, "incoherent"),
    ],
)
def test_native_installation_health_checks_precede_actuation(change: dict, match: str) -> None:
    robot, chain = robot_fixture()
    robot._feedback_limits = NativeFeedbackLimits(
        ((-3, 3),) * 6 + ((0, 1),), (0.001,) * 7, (3.0,) * 7, (20.0,) * 7, (80.0,) * 7, 0.01
    )
    try:
        chain.states[0] = replace(chain.states[0], **change)
        robot.command_joint_reference(stationary_packet())
        with pytest.raises(RuntimeError, match=match):
            robot.update()
        assert not chain.commands and not chain.running
        assert robot.native_execution_status()["fault"]
    finally:
        robot.close()


def test_native_health_persistence_is_not_extended_by_cache_polling() -> None:
    robot, chain = robot_fixture()
    robot._feedback_limits = NativeFeedbackLimits(
        ((-3, 3),) * 6 + ((0, 1),), (0.001,) * 7, (3.0,) * 7, (20.0,) * 7, (80.0,) * 7, 0.01, 0.01
    )
    try:
        chain.states[0] = replace(chain.states[0], temp_rotor=90)
        robot.command_joint_reference(stationary_packet(horizon=1))
        robot.update()
        time.sleep(0.012)
        with pytest.raises(RuntimeError, match="temperature"):
            robot.update()
        assert len(chain.commands) == 1
    finally:
        robot.close()


def test_staged_native_does_not_publish_and_requires_initial_reference() -> None:
    robot, chain = robot_fixture()
    try:
        assert not chain.commands
        with pytest.raises(RuntimeError, match="startup reference"):
            robot.start_execution()
        with pytest.raises(ValueError, match="PD-only"):
            robot.command_joint_reference(stationary_packet(), gravity_idle=True)
    finally:
        robot.close()
    assert not chain.disabled  # close is not motor disable


def test_guarded_sender_starts_only_after_complete_native_command_is_installed() -> None:
    robot, chain = robot_fixture()
    chain.requires_staged_start = True
    order = []

    def start_sender() -> None:
        assert chain.commands
        ff, command = chain.commands[-1]
        np.testing.assert_array_equal(ff, 0)
        np.testing.assert_allclose(command["pos"], [0, 0, 0, 0, 0, 0, -1])
        np.testing.assert_array_equal(command["kp"], robot._kp)
        assert command["valid_until"] > time.monotonic()
        order.append("sender")

    chain.start_thread = start_sender
    # Exercise start ordering without launching another test background thread.
    robot._server_thread = type("NativeThread", (), {"start": lambda _: order.append("native")})()
    try:
        robot.command_joint_reference(stationary_packet())
        robot.start_execution()
        assert order == ["sender", "native"]
    finally:
        robot._execution_started = False
        robot.close()


def test_limiter_release_occurs_at_native_reference_activation_not_publication() -> None:
    robot, chain = robot_fixture()
    calls = []
    limiter = SimpleNamespace(release_pending=True, defer_release=True, _is_clogged=True)

    def acknowledge() -> None:
        calls.append(time.monotonic())
        limiter.release_pending = False

    limiter.acknowledge_release = acknowledge
    robot._gripper_force_limiter = limiter
    try:
        robot.command_joint_reference(stationary_packet(horizon=1))
        origin = time.monotonic() + 0.015
        proposal = replace(stationary_packet(2), origin=origin, brake_at=origin + 0.02, release_gripper=True)
        robot.command_joint_reference(proposal)
        robot.update()
        assert not calls
        time.sleep(0.02)
        chain.states = [replace(m, received_monotonic=time.monotonic()) for m in chain.states]
        robot.update()
        assert len(calls) == 1 and calls[0] >= origin
        robot.update()
        assert len(calls) == 1
    finally:
        robot.close()


def test_native_reference_maps_raw_jaw_and_sets_sender_expiry() -> None:
    robot, chain = robot_fixture()
    try:
        packet = stationary_packet()
        robot.command_joint_reference(packet)
        robot.update()
        ff, command = chain.commands[-1]
        np.testing.assert_array_equal(ff, 0)
        np.testing.assert_allclose(command["pos"], [0, 0, 0, 0, 0, 0, -1])
        np.testing.assert_array_equal(command["kp"], robot._kp)
        assert command["valid_until"] > time.monotonic()
        with pytest.raises(ValueError, match="sequence"):
            robot.command_joint_reference(packet)
        time.sleep(0.025)
        robot.update()
        assert robot.native_execution_status()["braking_sequence"] == 1
        assert robot.native_execution_status()["update_generation"] == 2
        sample = robot.native_execution_status()["reference_sample"]
        assert sample["sequence"] == packet.sequence
        assert sample["evaluated_monotonic"] >= packet.origin
        assert sample["update_completed_monotonic"] >= sample["evaluated_monotonic"]
        assert sample["position_public"] == packet.nominal.stationary_positions
        assert sample["effective_position_public"] == packet.nominal.stationary_positions
        assert sample["feedforward_raw_motor_nm"] == (0.0,) * 7
    finally:
        robot.close()


@pytest.mark.parametrize("fault", ["stale", "position", "velocity", "status", "gravity"])
def test_native_fault_stops_refresh_and_cannot_reenable(fault: str) -> None:
    robot, chain = robot_fixture()
    try:
        robot.command_joint_reference(stationary_packet(horizon=1))
        if fault == "stale":
            chain.states[0].received_monotonic -= 1
        elif fault == "position":
            chain.states[0].pos = 0.5
        elif fault == "velocity":
            chain.states[0].vel = 1
        elif fault == "status":
            chain.states[0].error_code = "0xb"
        else:
            robot.use_gravity_comp = True

            def fail(_: Any) -> None:
                raise RuntimeError("gravity calculation failed")

            robot._compute_gravity_compensation = fail
        with pytest.raises(RuntimeError):
            robot.update()
        assert not chain.running
        assert not chain.commands
        assert robot.native_execution_status()["fault"]
        with pytest.raises(RuntimeError, match="faulted/closed"):
            robot.command_joint_reference(stationary_packet(2))
    finally:
        robot.close()


def test_gripper_contact_mismatch_is_not_arm_fault_and_disable_is_explicit() -> None:
    robot, chain = robot_fixture()
    try:
        robot.command_joint_reference(stationary_packet())
        chain.states[6].pos = -0.2
        robot.update()
        assert robot.native_execution_status()["fault"] is None
        result = robot.disable_motors()
        assert len(result) == 7 and chain.disabled
        count = len(chain.commands)
        with pytest.raises(RuntimeError):
            robot.command_joint_reference(stationary_packet(2))
        assert len(chain.commands) == count
    finally:
        robot.close()


@pytest.mark.parametrize("cancel", [False, True])
def test_future_reference_waits_for_origin_and_braking_discards_it(monkeypatch: Any, cancel: bool) -> None:
    robot, _chain = robot_fixture()
    try:
        now = time.monotonic()
        clock = [now]
        monkeypatch.setattr("i2rt.robots.motor_chain_robot.time.monotonic", lambda: clock[0])
        first = stationary_packet()
        robot.command_joint_reference(first)
        successor = replace(first, sequence=2, origin=now + 0.02, brake_at=now + 0.04)
        robot.command_joint_reference(successor)
        robot.update()
        assert robot.native_execution_status()["active_reference_sequence"] == 1
        assert robot.native_execution_status()["pending_reference_sequence"] == 2
        if cancel:
            assert robot.request_controlled_braking() == first
        clock[0] += 0.021
        robot.update()
        assert robot.native_execution_status()["active_reference_sequence"] == (1 if cancel else 2)
        assert robot.native_execution_status()["pending_reference_sequence"] is None
        assert robot.native_execution_status()["braking_sequence"] == (1 if cancel else -1)
    finally:
        robot.close()
