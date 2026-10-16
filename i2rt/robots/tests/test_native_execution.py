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
from i2rt.robots.utils import ArmType, GripperEffortPolicy, GripperForceLimiter, GripperType


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


def robot_fixture(*, gripper_effort_policy: GripperEffortPolicy | None = None) -> tuple[MotorChainRobot, FakeChain]:
    chain = FakeChain()
    robot = MotorChainRobot(
        chain,
        use_gravity_comp=False,
        joint_limits=np.array([[-3, 3]] * 6),
        kp=[80, 80, 80, 10, 10, 10, 20],
        kd=[5, 5, 5, 1.5, 1.5, 1.5, 0.5],
        gripper_index=6,
        gripper_limits=np.array([0.0, -2.0]),
        gripper_effort_policy=gripper_effort_policy,
        start_server_thread=False,
    )
    # Constructor's compatibility wait is not receipt progress. Refresh explicitly.
    chain.states = [replace(m, received_monotonic=time.monotonic()) for m in chain.states]
    return robot, chain


def stationary_packet(sequence: int = 1, horizon: float = 0.02) -> AdmittedReference:
    reference = JointReference(tuple(() for _ in range(7)), (0, 0, 0, 0, 0, 0, 0.5))
    now = time.monotonic()
    return AdmittedReference(sequence, now, now + horizon, reference, reference, (0.05,) * 7, (0.2,) * 7)


def native_feedback_limits(**overrides: Any) -> NativeFeedbackLimits:
    values = {
        "position": ((-3, 3),) * 6 + ((0, 1),),
        "position_noise": (0.001,) * 7,
        "velocity": (3.0,) * 7,
        "operating_motor_effort_nm": (10.0,) * 7,
        "hard_motor_effort_nm": (20.0,) * 7,
        "temperature_c": (80.0,) * 7,
        "maximum_sweep_skew_s": 0.01,
    }
    values.update(overrides)
    return NativeFeedbackLimits(**values)


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


def _limiter_state(*, target: float, current: float = 0.5, effort: float = 0.2) -> dict[str, float]:
    # Raw endpoints are deliberately reversed relative to normalized aperture.
    return {
        "target_qpos": -2.0 * target,
        "current_qpos": -2.0 * current,
        "current_qvel": 0.0,
        "current_eff": effort,
        "current_normalized_qpos": current,
        "target_normalized_qpos": target,
        "last_command_qpos": -0.9,
        "last_command_normalized_qpos": 0.45,
    }


def test_bidirectional_effort_latch_uses_public_direction_and_last_effective_command() -> None:
    policy = GripperEffortPolicy(10.0, 10.0, 0.01)
    limiter = GripperForceLimiter(50.0, GripperType.LINEAR_4310, ArmType.YAM, 20.0, policy)
    limiter.defer_release = True

    assert limiter.update(_limiter_state(target=0.2)) == pytest.approx(-0.9)
    status = limiter.status()
    assert status["state"] == "effort_latched_closing"
    assert status["blocked_direction"] == "closing"
    assert status["latched_public_qpos"] == pytest.approx(0.45)

    # Continued closure cannot ratchet the retained command through the load.
    assert limiter.update({**_limiter_state(target=0.1), "last_command_qpos": -1.2}) == pytest.approx(-0.9)
    # Opposite intent requests a bounded release; torque is not consulted.
    assert limiter.update(_limiter_state(target=0.8, effort=2.0)) == pytest.approx(-0.9)
    assert limiter.status()["state"] == "release_pending"
    limiter.acknowledge_release()
    assert limiter.update(_limiter_state(target=0.8, effort=2.0)) == pytest.approx(-1.6)
    assert not limiter.status()["contact_limited"]
    # A later reversal re-arms the limiter and may latch in the original direction.
    assert limiter.update(_limiter_state(target=0.2, effort=2.0)) == pytest.approx(-0.9)
    assert limiter.status()["blocked_direction"] == "closing"

    opening = GripperForceLimiter(50.0, GripperType.LINEAR_4310, ArmType.YAM, 20.0, policy)
    assert opening.update(_limiter_state(target=0.8)) == pytest.approx(-0.9)
    assert opening.status()["state"] == "effort_latched_opening"


def test_below_threshold_gripper_effort_preserves_target() -> None:
    limiter = GripperForceLimiter(
        50.0,
        GripperType.LINEAR_4310,
        ArmType.YAM,
        20.0,
        GripperEffortPolicy(10.0, 10.0, 0.01),
    )
    assert limiter.update(_limiter_state(target=0.2, effort=0.1)) == pytest.approx(-0.4)
    assert limiter.status()["state"] == "normal"


def test_legacy_factory_limiter_retains_speed_gated_force_adjustment() -> None:
    limiter = GripperForceLimiter(50.0, GripperType.LINEAR_4310, ArmType.YAM, 20.0)
    state = _limiter_state(target=0.2, effort=0.6)
    limited = limiter.update(state)
    assert limiter.status()["state"] == "legacy_force_limited"
    assert limited != pytest.approx(state["last_command_qpos"])
    assert limited != pytest.approx(state["target_qpos"])


def test_gripper_contact_is_latched_before_next_send_without_arm_fault() -> None:
    robot, chain = robot_fixture(gripper_effort_policy=GripperEffortPolicy(10.0, 10.0, 0.01))
    robot._feedback_limits = native_feedback_limits(
        operating_motor_effort_nm=(10.0,) * 6 + (4.0,),
        hard_motor_effort_nm=(20.0,) * 6 + (6.0,),
    )
    try:
        packet = stationary_packet(horizon=1)
        packet = replace(
            packet,
            nominal=JointReference(tuple(() for _ in range(7)), (0, 0, 0, 0, 0, 0, 0.2)),
            brake=JointReference(tuple(() for _ in range(7)), (0, 0, 0, 0, 0, 0, 0.2)),
        )
        chain.states[6] = replace(chain.states[6], eff=0.2)
        robot.command_joint_reference(packet)
        robot.update()
        status = robot.native_execution_status()
        assert status["fault"] is None
        assert status["gripper_limited"]
        assert status["gripper_effort_limiter"]["blocked_direction"] == "closing"
        assert chain.commands[-1][1]["pos"][6] == pytest.approx(-1.0)
        assert chain.commands[-1][1]["vel"][6] == 0.0
    finally:
        robot.close()


def test_operating_effort_is_report_only_but_hard_effort_retains_first_fault() -> None:
    robot, chain = robot_fixture()
    robot._feedback_limits = native_feedback_limits(
        operating_motor_effort_nm=(14.0,) * 7,
        hard_motor_effort_nm=(25.0,) * 7,
    )
    try:
        robot.command_joint_reference(stationary_packet(horizon=1))
        chain.states[1] = replace(chain.states[1], eff=-14.174359)
        robot.update()
        status = robot.native_execution_status()
        assert status["fault"] is None
        assert status["effort_assessment"]["operating_exceeded"][0]["motor_id"] == 2

        chain.states[1] = replace(chain.states[1], eff=-25.1, received_monotonic=time.monotonic())
        with pytest.raises(RuntimeError, match="hard raw motor effort"):
            robot.update()
        first = robot.native_execution_status()
        with pytest.raises(RuntimeError, match="faulted/closed"):
            robot.command_joint_reference(stationary_packet(2))
        final = robot.native_execution_status()
        assert final["fault"] == first["fault"]
        assert final["fault_details"]["motor_id"] == 2
        assert final["fault_details"]["observed_value"] == pytest.approx(-25.1)
        assert "faulted/closed native execution" in final["fault_consequences"][0]
    finally:
        robot.close()


def test_aperture_decoder_fringe_uses_slack_and_material_travel_uses_persistence() -> None:
    robot, chain = robot_fixture()
    robot._feedback_limits = native_feedback_limits(
        position_noise=(0.001,) * 7,
        hard_motor_effort_nm=(30.0,) * 7,
        travel_persistence_s=0.01,
    )
    packet = stationary_packet(horizon=1)
    zero_aperture = JointReference(tuple(() for _ in range(7)), (0, 0, 0, 0, 0, 0, 0.0))
    packet = replace(packet, nominal=zero_aperture, brake=zero_aperture)
    try:
        robot.command_joint_reference(packet)
        # Raw +0.001 maps to normalized -0.0005 through [0,-2]: legal decoder fringe.
        chain.states[6] = replace(chain.states[6], pos=0.001)
        robot.update()
        assert robot.native_execution_status()["fault"] is None

        # Raw +0.003 maps to -0.0015: outside slack, but not a one-sweep fault.
        chain.states[6] = replace(chain.states[6], pos=0.003, received_monotonic=time.monotonic())
        robot.update()
        time.sleep(0.012)
        chain.states = [replace(motor, received_monotonic=time.monotonic()) for motor in chain.states]
        with pytest.raises(RuntimeError, match="public joint travel"):
            robot.update()
        details = robot.native_execution_status()["fault_details"]
        assert details["motor_id"] == 7
        assert details["persistence_duration_s"] >= 0.01
    finally:
        robot.close()


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"pos": float("nan")}, "nonfinite"),
        ({"eff": 21.0}, "raw motor effort"),
        ({"temp_mos": 81.0}, "temperature"),
        ({"temp_mos": -1.0}, "unavailable"),
        ({"temp_rotor": float("nan")}, "nonfinite"),
        ({"vel": 4.0}, "joint velocity decoder sanity"),
        ({"sweep_id": 99}, "incoherent"),
    ],
)
def test_native_installation_health_checks_precede_actuation(change: dict, match: str) -> None:
    robot, chain = robot_fixture()
    robot._feedback_limits = native_feedback_limits()
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
    robot._feedback_limits = native_feedback_limits(temperature_persistence_s=0.01)
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
    limiter.status = lambda: {
        "release_pending": limiter.release_pending,
        "contact_limited": limiter._is_clogged,
    }
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


def test_reference_handoff_uses_time_after_feedback_read(monkeypatch: Any) -> None:
    robot, chain = robot_fixture()
    try:
        base = time.monotonic()
        clock = [base]
        monkeypatch.setattr("i2rt.robots.motor_chain_robot.time.monotonic", lambda: clock[0])
        robot.use_gravity_comp = True
        robot._compute_gravity_compensation = lambda _state: np.zeros(7)

        first = stationary_packet(horizon=0.02)
        robot.command_joint_reference(first, gravity_idle=True)
        successor = replace(first, sequence=2, origin=first.brake_at, brake_at=first.brake_at + 0.02)
        robot.command_joint_reference(successor, gravity_idle=True)

        clock[0] = successor.origin - 0.001

        def cross_handoff_during_feedback_read() -> list[MotorInfo]:
            clock[0] = successor.origin + 0.001
            chain.states = [replace(motor, received_monotonic=clock[0]) for motor in chain.states]
            return chain.states

        chain.read_states = cross_handoff_during_feedback_read
        robot.update()

        status = robot.native_execution_status()
        assert status["active_reference_sequence"] == 2
        assert status["pending_reference_sequence"] is None
        assert status["reference_sample"]["sequence"] == 2
        assert status["reference_sample"]["gravity_idle"]
        np.testing.assert_array_equal(status["reference_sample"]["kp_raw_motor"][:6], np.zeros(6))
    finally:
        robot.close()
