"""Whole native factory/update/sender path over fake physical exchanges, not dynamics."""

import time
from dataclasses import replace
from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from i2rt.motor_drivers import dm_driver
from i2rt.motor_drivers.utils import FeedbackFrameInfo
from i2rt.robots.get_robot import create_yam_motor_chain, reconcile_guarded_gripper_limits, resolve_yam_robot
from i2rt.robots.joint_reference import AdmittedReference, JointReference
from i2rt.robots.utils import GripperType


class FakePhysicalExchange:
    """Only replace the individual I/O interface; real chain locks/sender remain."""

    def __init__(self, positions: np.ndarray, fail_enable: int | None = None) -> None:
        self.positions = positions
        self.fail_enable = fail_enable
        self.enabled = []
        self.disabled = []
        self.commands = []
        self.closed = False

    def _drain_bus(self, *, timeout_s: float) -> None:
        assert timeout_s > 0

    def reply(self, motor: int, status: str = "0x1") -> FeedbackFrameInfo:
        return FeedbackFrameInfo(
            motor,
            status,
            "fake physical exchange",
            self.positions[motor - 1],
            0.0,
            0.0,
            25,
            26,
            time.monotonic(),
            time.time(),
        )

    def motor_on(self, motor: int, _kind: str, *, allow_error_recovery: bool) -> FeedbackFrameInfo:
        assert not allow_error_recovery
        self.enabled.append(motor)
        if motor == self.fail_enable:
            raise TimeoutError("injected partial enable failure")
        return self.reply(motor)

    def motor_off(self, motor: int, _kind: str) -> FeedbackFrameInfo:
        self.disabled.append(motor)
        return self.reply(motor, "0x0")

    def set_control(self, **kwargs: Any) -> FeedbackFrameInfo:
        self.commands.append(kwargs)
        return self.reply(kwargs["motor_id"])

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    "gripper",
    (
        GripperType.LINEAR_4310_STOCK,
        GripperType.LINEAR_4310_SOFT,
        GripperType.LINEAR_4310_SOFT_IPHONE_15_PRO,
        GripperType.LINEAR_4310_SOFT_IPHONE_15_PRO_MAX,
    ),
)
def test_actual_factory_native_controller_and_sender_keep_reviewed_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    gripper: GripperType,
) -> None:
    resolved = resolve_yam_robot(gripper_type=gripper, gripper_limits_override=np.array([0, -6.57]))
    # Explicitly reviewed alternative motor direction/zero offsets. Public/model
    # joint-6 sign remains independent of this physical direction mapping.
    resolved = replace(resolved, directions=[1, 1, 1, 1, 1, -1, 1], motor_offsets=[0.1] * 7)
    q = np.array([0, 1.88, 1.26, 0, 0, 0.1, -3.285])
    io = FakePhysicalExchange(q * resolved.directions + resolved.motor_offsets)
    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", lambda **_kwargs: io)
    chain = create_yam_motor_chain(resolved, "fake", guarded_startup=True)
    robot = None
    try:
        assert io.enabled == list(range(1, 8))
        assert not chain.start_thread_flag and not io.commands
        np.testing.assert_array_equal(chain.motor_offset, resolved.motor_offsets)
        np.testing.assert_allclose([m.pos for m in chain.read_states()], q, atol=1e-12)
        acquired = chain.acquire_startup_feedback()
        assert len(io.commands) == 7 and all(c["kp"] == c["kd"] == c["torque"] == 0 for c in io.commands)
        robot = resolved.construct(chain, start_server_thread=False, use_gravity_comp=True)
        # Refresh with explicit bounded probes after disk/model construction.
        fresh = chain.acquire_startup_feedback()
        assert fresh[0].receive_sequence > acquired[0].receive_sequence
        public = q.copy()
        public[6] = 0.5
        reference = JointReference(tuple(() for _ in range(7)), tuple(public))
        now = time.monotonic()
        packet = AdmittedReference(1, now, now + 0.2, reference, reference, (0.1,) * 7, (0.2,) * 7)
        robot.command_joint_reference(packet)
        robot.start_execution()
        deadline = time.monotonic() + 0.1
        while robot.native_execution_status()["update_generation"] < 3 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert robot.native_execution_status()["update_generation"] >= 3
        assert robot.native_execution_status()["fault"] is None
        assert chain.read_states()[0].receive_sequence > fresh[0].receive_sequence
        active = [c for c in io.commands if c["kp"] > 0]
        assert active
        for command in active:
            motor = command["motor_id"] - 1
            assert command["pos"] == pytest.approx(io.positions[motor])
            assert command["kp"] == resolved.robot_kwargs["kp"][motor]
        np.testing.assert_allclose(robot.get_joint_pos(), public, atol=1e-12)
        outcomes = robot.disable_motors()
        assert all(outcome["confirmed"] for outcome in outcomes)
        assert io.disabled == list(range(1, 8))
        assert not chain.running
        with pytest.raises(RuntimeError):
            robot.command_joint_reference(replace(packet, sequence=2))
    finally:
        if robot is not None:
            robot.close()
        else:
            chain.disable_motors()
            chain.close()
    assert io.closed


@pytest.mark.parametrize(
    ("feedback_position", "expected_branch"),
    ((-4.463454642557, 0), (1.819829098955, 1)),
)
def test_guarded_factory_reconciles_saved_gripper_branch_before_mapping(
    monkeypatch: pytest.MonkeyPatch,
    feedback_position: float,
    expected_branch: int,
) -> None:
    calibration = np.array([0.2550164034485398, -4.993705653467613])
    resolved = resolve_yam_robot(
        gripper_type=GripperType.LINEAR_4310_STOCK,
        gripper_limits_override=calibration,
    )
    positions = np.array([0, 1.88, 1.26, 0, 0, 0.1, feedback_position])
    io = FakePhysicalExchange(positions)
    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", lambda **_kwargs: io)
    chain = create_yam_motor_chain(resolved, "fake", guarded_startup=True)
    robot = None
    try:
        reconciled = reconcile_guarded_gripper_limits(
            resolved,
            chain,
            endpoint_tolerance_rad=0.0006,
        )
        record = reconciled.gripper_limit_reconciliation
        assert record is not None and record.branch_index == expected_branch
        np.testing.assert_array_equal(resolved.robot_kwargs["gripper_limits"], calibration)
        np.testing.assert_allclose(
            reconciled.robot_kwargs["gripper_limits"],
            calibration + expected_branch * 2 * np.pi,
        )
        assert len(record.feedback_positions_rad) == 3
        assert all(second > first for first, second in pairwise(record.feedback_receive_sequences))
        assert len(io.commands) == 14
        assert all(command["kp"] == command["kd"] == command["torque"] == 0 for command in io.commands)

        robot = reconciled.construct(chain, start_server_thread=False, use_gravity_comp=True)
        expected_aperture = (feedback_position - record.effective_limits_rad[0]) / (
            record.effective_limits_rad[1] - record.effective_limits_rad[0]
        )
        assert robot.get_joint_pos()[6] == pytest.approx(expected_aperture)
        assert robot.get_robot_info()["gripper_limit_reconciliation"]["branch_index"] == expected_branch
        assert robot.native_execution_status()["gripper_limit_reconciliation"]["branch_index"] == expected_branch
    finally:
        if robot is not None:
            robot.disable_motors()
            robot.close()
        elif chain.running:
            chain.disable_motors()
            chain.close()


def test_guarded_gripper_branch_refusal_disables_and_closes_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = resolve_yam_robot(
        gripper_type=GripperType.LINEAR_4310_STOCK,
        gripper_limits_override=np.array([0.2550164034485398, -4.993705653467613]),
    )
    io = FakePhysicalExchange(np.array([0, 1.88, 1.26, 0, 0, 0.1, 0.75]))
    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", lambda **_kwargs: io)
    chain = create_yam_motor_chain(resolved, "fake", guarded_startup=True)

    with pytest.raises(RuntimeError, match="no unique representable startup branch"):
        reconcile_guarded_gripper_limits(resolved, chain, endpoint_tolerance_rad=0.0006)

    assert io.disabled == list(range(1, 8))
    assert io.closed and not chain.running


@pytest.mark.parametrize("failed_motor", range(1, 8))
def test_partial_enable_failure_attempts_all_disables_and_never_starts_sender(
    monkeypatch: pytest.MonkeyPatch,
    failed_motor: int,
) -> None:
    resolved = resolve_yam_robot(
        gripper_type=GripperType.LINEAR_4310_SOFT, gripper_limits_override=np.array([0, -6.57])
    )
    io = FakePhysicalExchange(np.zeros(7), fail_enable=failed_motor)
    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", lambda **_kwargs: io)
    with pytest.raises(TimeoutError, match="partial enable"):
        create_yam_motor_chain(resolved, "fake", guarded_startup=True)
    assert io.enabled == list(range(1, failed_motor + 1))
    assert io.disabled == list(range(1, 8))
    assert io.closed and not io.commands
