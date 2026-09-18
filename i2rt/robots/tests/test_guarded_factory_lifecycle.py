"""Whole native factory/update/sender path over fake physical exchanges, not dynamics."""

import time
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from i2rt.motor_drivers import dm_driver
from i2rt.motor_drivers.utils import FeedbackFrameInfo
from i2rt.robots.get_robot import create_yam_motor_chain, resolve_yam_robot
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
