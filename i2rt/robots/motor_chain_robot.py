import copy
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from i2rt.motor_drivers.dm_driver import (
    MotorChain,
    MotorInfo,
    PassiveEncoderInfo,
)
from i2rt.robots.joint_reference import AdmittedReference
from i2rt.robots.model_coordinates import ModelCoordinateAdapter
from i2rt.robots.robot import Robot
from i2rt.robots.utils import ArmType, GripperForceLimiter, GripperType, JointMapper, detect_gripper_limits
from i2rt.utils.mujoco_utils import MuJoCoKDL
from i2rt.utils.recording import RobotMcapRecorder


@dataclass
class JointStates:
    names: List[str]
    pos: np.ndarray
    vel: np.ndarray
    eff: np.ndarray
    temp_mos: np.ndarray  # MOS temperature (float): Motor MOS temperature.
    temp_rotor: np.ndarray  # ROTOR temperature (float): Motor ROTOR temperature.
    timestamp: float

    def asdict(self) -> Dict[str, Any]:
        return {
            "names": self.names,
            "pos": self.pos.flatten().tolist(),
            "vel": self.vel.flatten().tolist(),
            "eff": self.eff.flatten().tolist(),
        }


@dataclass
class JointCommands:
    torques: np.ndarray

    pos: np.ndarray
    vel: np.ndarray
    kp: np.ndarray
    kd: np.ndarray

    indices: Optional[List[int]] = None

    @classmethod
    def init_all_zero(cls, n_joints: int) -> "JointCommands":
        return cls(
            torques=np.zeros(n_joints),
            pos=np.zeros(n_joints),
            vel=np.zeros(n_joints),
            kp=np.zeros(n_joints),
            kd=np.zeros(n_joints),
        )


@dataclass(frozen=True)
class NativeFeedbackLimits:
    """Optional installation-qualified public travel and raw-motor health bounds."""

    position: Tuple[Tuple[float, float], ...]
    position_noise: Tuple[float, ...]
    velocity: Tuple[float, ...]
    motor_effort_nm: Tuple[float, ...]
    temperature_c: Tuple[float, ...]
    maximum_sweep_skew_s: float
    persistence_s: float = 0.0

    def validate(self, size: int) -> None:
        position = np.asarray(self.position)
        if (
            position.shape != (size, 2)
            or not np.all(np.isfinite(position))
            or np.any(position[:, 0] >= position[:, 1])
        ):
            raise ValueError("native feedback position bounds must be finite ordered public pairs")
        for name in ("position_noise", "velocity", "motor_effort_nm", "temperature_c"):
            value = np.asarray(getattr(self, name))
            if value.shape != (size,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
                raise ValueError(f"native feedback {name} must contain positive finite per-motor bounds")
        if not np.isfinite(self.maximum_sweep_skew_s) or self.maximum_sweep_skew_s <= 0:
            raise ValueError("native maximum_sweep_skew_s must be positive")
        if not np.isfinite(self.persistence_s) or self.persistence_s < 0:
            raise ValueError("native feedback persistence_s must be finite and nonnegative")


class MotorChainRobot(Robot):
    """A generic Robot protocol."""

    def __init__(
        self,
        motor_chain: MotorChain,
        xml_path: Optional[str] = None,
        use_gravity_comp: bool = True,
        gravity: Optional[np.ndarray] = None,
        gravity_comp_factor: Optional[np.ndarray] = None,
        gripper_index: Optional[int] = None,  # Zero starting index: if you have a 6 dof arm and last one is gripper: 6
        kp: Union[float, List[float]] = 10.0,
        kd: Union[float, List[float]] = 1.0,
        grav_comp_kd: Optional[np.ndarray] = None,  # per-joint MIT-mode kd, active only in grav-comp idle
        coulomb_friction: Optional[
            np.ndarray
        ] = None,  # per-joint Coulomb friction (Nm); applied as coulomb_friction * sign(q_dot)
        use_coulomb_friction: bool = False,  # if True, add the Coulomb friction feedforward in the grav-comp loop
        joint_limits: Optional[np.ndarray] = None,  # if provided, override the mujoco xml joint limits
        model_coordinate_adapter: Optional[ModelCoordinateAdapter] = None,
        gripper_limits: Optional[np.ndarray] = None,  # [closed, open]
        limit_gripper_force: float = -1,  # whether to limit the gripper effort when it is blocked. -1 means no limit.
        clip_motor_torque: float = np.inf,  # clip the offset motor torque, real motor torque can still still be larger than this setting depending on the motor onboard PID loop
        gripper_type: GripperType = GripperType.LINEAR_4310,
        arm_type: ArmType = ArmType.YAM,
        temp_record_flag: bool = False,  # whether record the motor's temperature
        enable_gripper_calibration: bool = False,  # whether to auto-detect gripper limits
        zero_gravity_mode: bool = True,
        # below are calibration parameters
        test_torque: float = 0.5,  # test torque for gripper detection (Nm)
        test_duration: float = 2.0,  # max test duration for each direction (s)
        position_threshold: float = 0.01,  # minimum position change to consider motor still moving (rad)
        check_interval: float = 0.05,  # time interval between checks (s)
        pinned_cpu: int | None = None,
        joint_state_saver_factory: Optional[Callable[[], Any]] = None,
        set_realtime_and_pin_callback: Optional[Callable[[int], None]] = None,
        enable_auto_recovery: Optional[bool] = None,  # None: inherit motor_chain's setting; True/False: override it
        start_server_thread: bool = True,
        feedback_max_age_s: float = 0.05,
        native_command_lease_s: float = 0.03,
        feedback_limits: NativeFeedbackLimits | None = None,
        gripper_limit_reconciliation: dict[str, Any] | None = None,
    ) -> None:
        # Set up CPU pinning and real-time scheduling if requested
        if pinned_cpu is not None and set_realtime_and_pin_callback is not None:
            set_realtime_and_pin_callback(pinned_cpu)

        self._joint_state_saver_factory = joint_state_saver_factory
        self._set_realtime_and_pin_callback = set_realtime_and_pin_callback
        self._arm_type = arm_type
        self._gripper_type = gripper_type
        self._gripper_limit_reconciliation = copy.deepcopy(gripper_limit_reconciliation)
        self._model_coordinate_adapter = model_coordinate_adapter
        if model_coordinate_adapter is not None and len(model_coordinate_adapter.public_names) != len(motor_chain):
            raise ValueError(
                "model coordinate public dimension "
                f"{len(model_coordinate_adapter.public_names)} != motor chain dimension {len(motor_chain)}"
            )
        self.temp_record_flag = temp_record_flag
        if gripper_index is not None:
            assert gripper_index == len(motor_chain) - 1, (
                "Gripper index should be the last one, but got {gripper_index}"
            )

            # Auto-detect gripper limits if enabled and gripper_limits is None
            print(
                f"initializing motorchain robot, gripper_limits: {gripper_limits}, enable_gripper_calibration: {enable_gripper_calibration}"
            )
            if gripper_limits is None and enable_gripper_calibration:
                logger = logging.getLogger(__name__)
                logger.info("Auto-detecting gripper limits...")
                detected_limits = detect_gripper_limits(
                    motor_chain=motor_chain,
                    gripper_index=gripper_index,
                    test_torque=test_torque,
                    max_duration=test_duration,
                    position_threshold=position_threshold,
                    check_interval=check_interval,
                )
                gripper_limits = np.array(detected_limits)
                logger.info(f"Gripper limits auto-detected: {gripper_limits}")
            elif gripper_limits is None:
                raise ValueError(
                    f"{self}: Gripper limits are required if gripper index is provided and auto-calibration is disabled."
                )
            else:
                # Use the provided gripper_limits
                logger = logging.getLogger(__name__)
                logger.info(f"Using provided gripper limits: {gripper_limits}")

        self._last_gripper_command_qpos = 1  # initialize as fully open
        assert clip_motor_torque >= 0.0
        self._clip_motor_torque = clip_motor_torque
        self.motor_chain = motor_chain
        # None means inherit whatever the chain was constructed with; an explicit value overrides it.
        # The chain reads this flag live each control-loop iteration, so a late set is safe.
        if enable_auto_recovery is not None:
            self.motor_chain.enable_auto_recovery = enable_auto_recovery
        self.use_gravity_comp = use_gravity_comp
        self.gravity_comp_factor = (
            gravity_comp_factor if gravity_comp_factor is not None else np.ones(len(motor_chain))
        )

        # variables for gripper effort limiting
        self._gripper_index = gripper_index
        self.remapper = JointMapper({}, len(motor_chain))  # so it works without gripper
        self._gripper_limits = gripper_limits
        self._gripper_force_limiter: Optional[GripperForceLimiter] = None
        self._limit_gripper_force: float = -1.0

        if self._gripper_index is not None:
            self._gripper_force_limiter = GripperForceLimiter(
                max_force=limit_gripper_force, gripper_type=gripper_type, arm_type=arm_type, kp=kp[gripper_index]
            )  # force in newton
            self._limit_gripper_force = limit_gripper_force

            self.remapper = JointMapper(
                index_range_map={gripper_index: gripper_limits},
                total_dofs=len(motor_chain),
            )

        # make sure kp, kd are float number not int
        self._kp = (
            np.array(
                [
                    kp,
                ]
                * len(motor_chain)
            )
            if isinstance(kp, float)
            else np.array(kp)
        )
        self._kd = (
            np.array(
                [
                    kd,
                ]
                * len(motor_chain)
            )
            if isinstance(kd, float)
            else np.array(kd)
        )
        self._grav_comp_kd = (
            np.array(grav_comp_kd, dtype=float) if grav_comp_kd is not None else np.zeros(len(motor_chain))
        )
        assert len(self._grav_comp_kd) == len(motor_chain), (
            f"grav_comp_kd length {len(self._grav_comp_kd)} != motor_chain length {len(motor_chain)}"
        )
        self.use_coulomb_friction = use_coulomb_friction
        self._coulomb_friction = (
            np.array(coulomb_friction, dtype=float) if coulomb_friction is not None else np.zeros(len(motor_chain))
        )
        assert len(self._coulomb_friction) == len(motor_chain), (
            f"coulomb_friction length {len(self._coulomb_friction)} != motor_chain length {len(motor_chain)}"
        )

        self._joint_limits: Optional[np.ndarray] = None
        if xml_path is not None:
            self.xml_path = os.path.expanduser(xml_path)
            self.kdl = MuJoCoKDL(self.xml_path)
            if gravity is not None:
                self.kdl.set_gravity(gravity)
            # Load the joint limits from the xml file
            self._joint_limits = self.kdl.joint_limits
        else:
            assert use_gravity_comp is False, "Gravity compensation requires a valid XML path."

        # override the xml joint limits with the provided joint_limits
        if joint_limits is not None:
            joint_limits = np.array(joint_limits)
            assert np.all(joint_limits[:, 0] < joint_limits[:, 1]), (
                "Lower joint limits must be smaller than upper limits"
            )
            self._joint_limits = joint_limits
        # Initialize joint state saver if factory is provided
        if self._joint_state_saver_factory is not None:
            self._joint_state_saver = self._joint_state_saver_factory()
        else:
            self._joint_state_saver = None

        self._command_lock = threading.Lock()
        self._reference: Optional[AdmittedReference] = None
        self._pending_reference: Optional[Tuple[AdmittedReference, bool]] = None
        self._reference_gravity_idle = False
        self._reference_braking_sequence = -1
        self._gripper_release_sequence = -1
        self._native_fault: Optional[str] = None
        self._native_update_generation = 0
        self._native_update_monotonic = 0.0
        self._native_reference_sample: Optional[Dict[str, Any]] = None
        self._feedback_max_age_s = float(feedback_max_age_s)
        self._native_command_lease_s = float(native_command_lease_s)
        self._feedback_limits = feedback_limits
        self._health_violation_since: dict[str, float] = {}
        if feedback_limits is not None:
            feedback_limits.validate(len(motor_chain))
        if min(feedback_max_age_s, native_command_lease_s) <= 0 or not np.all(
            np.isfinite([feedback_max_age_s, native_command_lease_s])
        ):
            raise ValueError("native feedback and command lease intervals must be positive and finite")
        self._state_lock = threading.Lock()
        self._mcap_lock = threading.Lock()
        self._mcap_recorder: Optional[RobotMcapRecorder] = None
        self._joint_state: Optional[JointStates] = None
        while self._joint_state is None:
            # wait to recive joint data
            if start_server_thread:
                time.sleep(0.05)
            self._joint_state = self._motor_state_to_joint_state(self.motor_chain.read_states())
        if self._gripper_index is not None:
            self._last_gripper_command_qpos = self.remapper.to_robot_joint_pos_space(self._joint_state.pos)[
                self._gripper_index
            ]
        self._commands = JointCommands.init_all_zero(len(motor_chain))
        if zero_gravity_mode:
            self._commands.kd = self._grav_comp_kd.copy()
        # For SWE-454, check if the current qpos is in the joint limits
        self._check_current_qpos_in_joint_limits()

        self._last_motor_torques: Optional[np.ndarray] = None
        self._stop_event = threading.Event()  # Add a stop event
        self._server_thread = threading.Thread(target=self.start_server, name="robot_server")
        self._execution_started = False
        # Staged construction observes only. The session must install its admitted
        # startup brake before explicitly starting native execution.
        if start_server_thread:
            self._server_thread.start()
            self._execution_started = True
        if start_server_thread and not zero_gravity_mode:
            # set current qpos as target pos with the default PD parameters
            self.command_joint_pos(self._joint_state.pos)

    def start_execution(self) -> None:
        """Start a previously staged native update worker; never re-enable faults."""
        if self._stop_event.is_set():
            raise RuntimeError("closed native execution cannot restart")
        if not self._execution_started:
            if self._reference is None:
                raise RuntimeError("staged execution requires an admitted startup reference")
            if getattr(self.motor_chain, "requires_staged_start", False):
                # Install all gains, mapped q/qd and native FF before any CAN
                # repeater starts. Startup observations came from bounded probes.
                self.update()
                self.motor_chain.start_thread()
            self._server_thread.start()
            self._execution_started = True

    def command_joint_reference(self, reference: AdmittedReference, *, gravity_idle: bool = False) -> None:
        """Atomically publish one admitted finite command and independent fallback.

        No solve, Ruckig invocation, silent clipping or automatic fault recovery.
        Call only from the session's commit boundary; legacy command methods are
        not the guarded session interface.
        """
        if len(reference.nominal.pieces) != len(self.motor_chain):
            raise ValueError("native reference dimension differs from motor chain")
        if gravity_idle and not self.use_gravity_comp:
            raise ValueError("gravity-idle is unavailable in PD-only execution")
        if not self.use_gravity_comp and self.use_coulomb_friction:
            raise ValueError("PD-only guarded execution cannot include Coulomb feedforward")
        with self._command_lock:
            if self._native_fault is not None or self._stop_event.is_set():
                raise RuntimeError("faulted/closed native execution cannot accept references")
            latest = self._pending_reference[0] if self._pending_reference else self._reference
            if latest is not None and reference.sequence <= latest.sequence:
                raise ValueError("native reference sequence must advance")
            if time.monotonic() >= reference.brake_at:
                raise ValueError("native reference publication is already expired")
            if self._reference is not None and reference.origin > time.monotonic():
                self._pending_reference = (reference, gravity_idle)
            else:
                self._reference = reference
                self._reference_gravity_idle = gravity_idle
                self._pending_reference = None
            if self._gripper_force_limiter is not None:
                self._gripper_force_limiter.defer_release = True

    def native_execution_status(self) -> Dict[str, Any]:
        return {
            "fault": self._native_fault,
            "update_generation": self._native_update_generation,
            "updated_monotonic": self._native_update_monotonic,
            "reference_sample": self._native_reference_sample,
            "braking_sequence": self._reference_braking_sequence,
            "active_reference_sequence": None if self._reference is None else self._reference.sequence,
            "pending_reference_sequence": None
            if self._pending_reference is None
            else self._pending_reference[0].sequence,
            "gripper_release_pending": bool(
                self._gripper_force_limiter and self._gripper_force_limiter.release_pending
            ),
            "gripper_limited": bool(self._gripper_force_limiter and self._gripper_force_limiter._is_clogged),
            "gripper_limit_reconciliation": copy.deepcopy(self._gripper_limit_reconciliation),
        }

    def request_controlled_braking(self) -> Optional[AdmittedReference]:
        """Cancel the next proposal; consume the active packet's admitted brake.

        The remaining admitted nominal prefix lasts at most its current release
        horizon. Never jump directly to a future brake's initial joint state.
        The session invalidates its planner epoch before calling this method.
        """
        with self._command_lock:
            self._pending_reference = None
            return self._reference

    def disable_motors(self) -> list[dict]:
        """Stop updates, then explicitly request per-motor disable (not close)."""
        self._stop_event.set()
        return self.motor_chain.disable_motors()

    def __repr__(self) -> str:
        return f"MotorChainRobot(arm_type={self._arm_type}, gripper_type={self._gripper_type}, motor_chain={self.motor_chain})"

    def _check_current_qpos_in_joint_limits(self, buffer_rad: float = 0.1) -> None:
        """Check if the self._joint_state is in the joint limits.
        If violated, raise an error.
        """
        if self._joint_state is None or self._joint_limits is None:
            raise RuntimeError(
                f"{self}: Joint limits:{self._joint_limits} or joint state:{self._joint_state} are not set."
            )

        current_pos = self._joint_state.pos

        # Check arm joints (exclude gripper if present)
        if self._gripper_index is not None:
            # Only check arm joints, not the gripper
            arm_pos = current_pos[: self._gripper_index]
            arm_limits = self._joint_limits
        else:
            # Check all joints
            arm_pos = current_pos
            arm_limits = self._joint_limits

        # Check if any joint is outside its limits
        lower_limits = arm_limits[:, 0] - buffer_rad
        upper_limits = arm_limits[:, 1] + buffer_rad

        # Find joints that violate lower limits
        lower_violations = arm_pos < lower_limits
        # Find joints that violate upper limits
        upper_violations = arm_pos > upper_limits

        if np.any(lower_violations) or np.any(upper_violations):
            violation_details = []

            for i, (pos, lower, upper) in enumerate(zip(arm_pos, lower_limits, upper_limits, strict=False)):
                if pos < lower:
                    violation_details.append(f"Joint {i}: {pos:.4f} < {lower:.4f} (lower limit)")
                elif pos > upper:
                    violation_details.append(f"Joint {i}: {pos:.4f} > {upper:.4f} (upper limit)")

            violation_msg = "; ".join(violation_details)
            # turn off the main motor control thread as well.
            self.motor_chain.running = False
            raise RuntimeError(
                f"{self}: Joint limit violation detected: {violation_msg}, the root reason should be zero position offset. possible solution: 1. move the arm to zero position and power cycle the robot. 2. Recalibrate the motor zero position."
            )

    def get_robot_info(self) -> Dict[str, Any]:
        """Get the robot information, such as kp, kd, joint limits, gripper limits, etc."""
        info: Dict[str, Any] = {
            "arm_type": self._arm_type,
            "gripper_type": self._gripper_type,
            "kp": self._kp,
            "kd": self._kd,
            "grav_comp_kd": self._grav_comp_kd,
            "coulomb_friction": self._coulomb_friction,
            "use_coulomb_friction": self.use_coulomb_friction,
            "joint_limits": self._joint_limits,
            "gripper_limits": self._gripper_limits,
            "gripper_limit_reconciliation": copy.deepcopy(self._gripper_limit_reconciliation),
            "gravity_comp_factor": self.gravity_comp_factor,
            "gripper_index": self._gripper_index,
            "enable_auto_recovery": getattr(self.motor_chain, "enable_auto_recovery", False),
            "model_coordinate_schema": (
                self._model_coordinate_adapter.SCHEMA_VERSION if self._model_coordinate_adapter is not None else None
            ),
        }
        if self._gripper_index is not None:
            info["limit_gripper_effort"] = self._limit_gripper_force
        return info

    def start_server(self) -> None:
        """Start the server."""
        last_time = time.time()
        iteration_count = 0
        logging.info("initializing, ....")

        while not self._stop_event.is_set():  # Check the stop event
            current_time = time.time()
            elapsed_time = current_time - last_time

            try:
                self.update()
                if self._stop_event.is_set():
                    return
                if not self.motor_chain.running:
                    raise RuntimeError(f"{self}: motor chain is not running, exiting the robot server")
            except Exception as error:
                if self._reference is None:
                    raise  # Preserve the legacy thread behavior for legacy callers.
                self._native_fault = str(error)
                self._stop_event.set()
                self.motor_chain.running = False
                logging.exception("guarded native execution ended; fault is available through native_execution_status")
                return
            time.sleep(0.001)

            iteration_count += 1
            if elapsed_time >= 10.0:
                control_frequency = iteration_count / elapsed_time
                # Overwrite the current line with the new frequency information
                logging.info(f"{self}: Grav Comp Control Frequency: {control_frequency:.2f} Hz")
                if control_frequency < 100:
                    logging.warning(
                        f"{self}: Gravity compensation control loop is slow, current frequency: {control_frequency:.2f} Hz"
                    )
                # Reset the counter and timer
                last_time = current_time
                iteration_count = 0

    def update(self) -> None:
        """Update native commands, retaining a guarded execution fault outcome."""
        try:
            self._update_once()
        except Exception as error:
            if self._reference is not None:
                self._native_fault = str(error)
                self._stop_event.set()
                self.motor_chain.running = False
            raise

    def _check_native_feedback_limits(self, motors: list[MotorInfo], observed: JointStates, now: float) -> None:
        limits = self._feedback_limits
        if not np.all(np.isfinite((observed.pos, observed.vel))):
            raise RuntimeError("native feedback position/velocity is nonfinite")
        if limits is None:
            return  # Explicitly retained legacy/simulation fixture contract.
        stamps = [motor.received_monotonic for motor in motors]
        if np.ptp(stamps) > limits.maximum_sweep_skew_s or len({m.sweep_id for m in motors}) != 1:
            raise RuntimeError("native feedback sweep is incoherent")
        effort = np.asarray([motor.eff for motor in motors])
        temperatures = np.asarray([[motor.temp_mos, motor.temp_rotor] for motor in motors])
        if not np.all(np.isfinite(effort)) or not np.all(np.isfinite(temperatures)) or np.any(temperatures < 0):
            # DAMIAO encodes temperatures as unsigned bytes. The API's -1
            # sentinel means no temperature measurement, not a cold motor.
            raise RuntimeError("native motor effort/temperature feedback is unavailable or nonfinite")
        positions = np.asarray(limits.position)
        violations = {
            "public joint travel": np.any(observed.pos < positions[:, 0] - limits.position_noise)
            or np.any(observed.pos > positions[:, 1] + limits.position_noise),
            "public joint velocity": np.any(np.abs(observed.vel) > limits.velocity),
            "raw motor effort": np.any(np.abs(effort) > limits.motor_effort_nm),
            "motor temperature": np.any(temperatures > np.asarray(limits.temperature_c)[:, None]),
        }
        for name, exceeded in violations.items():
            if exceeded:
                since = self._health_violation_since.setdefault(name, now)
                if now - since >= limits.persistence_s:
                    raise RuntimeError(f"native installation envelope exceeded: {name}")
            else:
                self._health_violation_since.pop(name, None)

    def _update_once(self) -> None:
        """Update the robot.

        Send Torques and update the joint state.
        """
        with self._command_lock:
            if self._reference is not None and self._stop_event.is_set():
                return
            if self._pending_reference is not None and time.monotonic() >= self._pending_reference[0].origin:
                self._reference, self._reference_gravity_idle = self._pending_reference
                self._pending_reference = None
            joint_commands = copy.deepcopy(self._commands)
            reference = self._reference
            gravity_idle = self._reference_gravity_idle
        if reference is not None:
            try:
                motors = self.motor_chain.read_states()
                now = time.monotonic()
                # A native update may start just before a scheduled handoff and
                # finish its feedback read just after it. Select the reference
                # against the actual evaluation time so an expired predecessor
                # cannot apply its braking/full-stiffness branch for one update.
                with self._command_lock:
                    if self._pending_reference is not None and now >= self._pending_reference[0].origin:
                        self._reference, self._reference_gravity_idle = self._pending_reference
                        self._pending_reference = None
                    reference = self._reference
                    gravity_idle = self._reference_gravity_idle
                assert reference is not None
                if any(
                    m.error_code not in ("0x1", 1)
                    or m.receive_sequence <= 0
                    or not 0 <= now - m.received_monotonic <= self._feedback_max_age_s
                    for m in motors
                ):
                    raise RuntimeError("native feedback unhealthy or stale")
                observed = self._motor_state_to_joint_state(motors)
                self._check_native_feedback_limits(motors, observed, now)
                q, qd, qdd = reference.at(now)
                check_from = (
                    (self._gripper_index if self._gripper_index is not None else len(motors))
                    if (gravity_idle and now < reference.brake_at)
                    else 0
                )
                # Object-limited aperture mismatch is not an arm execution fault.
                # Native jaw fault/status/range checks remain separate.
                check_to = self._gripper_index if self._gripper_index is not None else len(motors)
                if np.any(
                    np.abs(observed.pos[check_from:check_to] - q[check_from:check_to])
                    > reference.following_error[check_from:check_to]
                ):
                    raise RuntimeError(
                        "native reference following corridor exceeded: "
                        f"observed={observed.pos[check_from:check_to].tolist()}, "
                        f"reference={q[check_from:check_to].tolist()}, command={reference.sequence}"
                    )
                velocity_error = np.abs(observed.vel[check_from:check_to] - qd[check_from:check_to])
                if np.any(velocity_error > reference.following_velocity_error[check_from:check_to]):
                    raise RuntimeError(
                        "native reference velocity corridor exceeded: "
                        f"observed={observed.vel[check_from:check_to].tolist()}, "
                        f"reference={qd[check_from:check_to].tolist()}, command={reference.sequence}"
                    )
                joint_commands.pos = self.remapper.to_robot_joint_pos_space(q)
                joint_commands.vel = self.remapper.to_robot_joint_vel_space(qd)
                joint_commands.torques[:] = 0
                joint_commands.kp = self._kp.copy()
                joint_commands.kd = self._kd.copy()
                if gravity_idle and now < reference.brake_at:
                    arm_end = self._gripper_index if self._gripper_index is not None else len(motors)
                    joint_commands.kp[:arm_end] = 0
                    joint_commands.kd[:arm_end] = self._grav_comp_kd[:arm_end]
                    # Native gravity-idle damps measured motion toward zero;
                    # a measured-state fallback must not become a velocity target.
                    joint_commands.vel[:arm_end] = 0
                if now >= reference.brake_at:
                    self._reference_braking_sequence = reference.sequence
                with self._state_lock:
                    self._joint_state = observed
                    if (
                        reference.release_gripper
                        and now < reference.brake_at
                        and self._gripper_release_sequence != reference.sequence
                    ):
                        if self._gripper_force_limiter is not None and self._gripper_force_limiter.release_pending:
                            index = self._gripper_index
                            if (
                                abs(observed.pos[index] - q[index]) > reference.following_error[index]
                                or abs(observed.vel[index] - qd[index]) > reference.following_velocity_error[index]
                            ):
                                raise RuntimeError("native jaw release continuation is outside measured corridor")
                            self._gripper_force_limiter.acknowledge_release()
                        self._gripper_release_sequence = reference.sequence
            except Exception as error:
                self._native_fault = str(error)
                self._stop_event.set()
                self.motor_chain.running = False
                raise
        with self._state_lock:
            g = (
                self._compute_gravity_compensation(self._joint_state)
                if self.use_gravity_comp
                else np.zeros(len(self.motor_chain))
            )
            friction_comp = (
                self._coulomb_friction * np.sign(self._joint_state.vel) if self.use_coulomb_friction else 0.0
            )
            motor_torques = joint_commands.torques + g * self.gravity_comp_factor + friction_comp
            motor_torques = np.clip(motor_torques, -self._clip_motor_torque, self._clip_motor_torque)
            self._last_motor_torques = motor_torques.copy()

            if self._gripper_index is not None:
                if self._limit_gripper_force > 0 and self._joint_state is not None:
                    # Get current gripper state in raw robot joint pos space
                    gripper_state = {
                        "target_qpos": joint_commands.pos[self._gripper_index],
                        "current_qpos": self.remapper.to_robot_joint_pos_space(self._joint_state.pos)[
                            self._gripper_index
                        ],
                        "current_qvel": self.remapper.to_robot_joint_vel_space(self._joint_state.vel)[
                            self._gripper_index
                        ],
                        "current_eff": self._joint_state.eff[self._gripper_index],
                        "current_normalized_qpos": self._joint_state.pos[self._gripper_index],
                        "target_normalized_qpos": self.remapper.to_command_joint_pos_space(joint_commands.pos)[
                            self._gripper_index
                        ],
                        "last_command_qpos": self._last_gripper_command_qpos,
                    }

                    self._gripper_force_limiter._kp = float(joint_commands.kp[self._gripper_index])
                    joint_commands.pos[self._gripper_index] = self._gripper_force_limiter.update(gripper_state)

                # add final clip so the gripper won't be over-adjusted
                joint_commands.pos[self._gripper_index] = np.clip(
                    joint_commands.pos[self._gripper_index],
                    min(self._gripper_limits),
                    max(self._gripper_limits),
                )
                self._last_gripper_command_qpos = joint_commands.pos[self._gripper_index]
            if not self._update_joint_state(motor_torques, joint_commands):
                return
            self._native_update_generation += 1
            self._native_update_monotonic = time.monotonic()
            if reference is not None:
                # Actual native evaluation, not a future published packet sampled
                # early by the supervisor. This is command evidence, NOT a claim
                # of simultaneous motor replies or observed position arrival.
                self._native_reference_sample = {
                    "sequence": reference.sequence,
                    "evaluated_monotonic": now,
                    "update_completed_monotonic": self._native_update_monotonic,
                    "position_public": tuple(q),
                    "velocity_public": tuple(qd),
                    "acceleration_public": tuple(qdd),
                    "jerk_public": reference.jerk_at(now),
                    "effective_position_public": tuple(self.remapper.to_command_joint_pos_space(joint_commands.pos)),
                    "effective_velocity_public": tuple(self.remapper.to_command_joint_vel_space(joint_commands.vel)),
                    "kp_raw_motor": tuple(joint_commands.kp),
                    "kd_raw_motor": tuple(joint_commands.kd),
                    "feedforward_raw_motor_nm": tuple(motor_torques),
                    "gravity_idle": bool(gravity_idle and now < reference.brake_at),
                }

    def _update_joint_state(
        self,
        motor_torques: np.ndarray,
        joint_commands: "JointCommands",
        encoder_infos: Optional[List[PassiveEncoderInfo]] = None,
    ) -> bool:
        """Send commands to motor chain, update joint state, and optionally save to disk."""
        if getattr(self, "_reference", None) is not None and self._stop_event.is_set():
            return False
        if (
            hasattr(self.motor_chain, "get_same_bus_device_states")
            and callable(self.motor_chain.get_same_bus_device_states)
            and self.motor_chain.same_bus_device_driver is not None
        ):
            has_gripper_encoder = True
            encoder_infos = self.motor_chain.get_same_bus_device_states()
            assert len(encoder_infos) == 1, "Only one encoder is supported"
            assert isinstance(encoder_infos[0], PassiveEncoderInfo), "Encoder info must be a PassiveEncoderInfo"
        else:
            has_gripper_encoder = False

        lease = (
            {"valid_until": time.monotonic() + self._native_command_lease_s}
            if getattr(self, "_reference", None) is not None
            else {}
        )
        motor_state = self.motor_chain.set_commands(
            motor_torques,
            pos=joint_commands.pos,
            vel=joint_commands.vel,
            kp=joint_commands.kp,
            kd=joint_commands.kd,
            **lease,
        )
        self._joint_state = self._motor_state_to_joint_state(motor_state)

        with self._mcap_lock:
            if self._mcap_recorder is not None:
                try:
                    self._mcap_recorder.add(
                        timestamp=self._joint_state.timestamp,
                        position=self._joint_state.pos,
                        velocity=self._joint_state.vel,
                        effort=self._joint_state.eff,
                        required_torque=motor_torques,
                        temp_mos=self._joint_state.temp_mos,
                        temp_rotor=self._joint_state.temp_rotor,
                    )
                except RuntimeError:
                    logging.exception("MCAP recording stopped after its writer failed")
                    self._mcap_recorder = None

        # For SWE-454: keep monitoring qpos during runtime
        self._check_current_qpos_in_joint_limits()

        if self._joint_state_saver is not None:
            assert not (has_gripper_encoder and self._gripper_index is not None), (
                "Either has_gripper_encoder=True or self._gripper_index is not None"
            )
            ee_pos = ee_vel = ee_eff = None
            if has_gripper_encoder:
                ee_pos = np.array([info.position for info in encoder_infos])
                ee_vel = np.array([info.velocity for info in encoder_infos])
            elif self._gripper_index is not None:
                ee_pos = self._joint_state.pos[self._gripper_index]
                ee_vel = self._joint_state.vel[self._gripper_index]
                ee_eff = self._joint_state.eff[self._gripper_index]

            if self._gripper_index is None:
                pos = self._joint_state.pos
                vel = self._joint_state.vel
                eff = self._joint_state.eff
            else:
                pos = self._joint_state.pos[: self._gripper_index]
                vel = self._joint_state.vel[: self._gripper_index]
                eff = self._joint_state.eff[: self._gripper_index]

            self._joint_state_saver.add(
                timestamp=self._joint_state.timestamp,
                pos=pos,
                vel=vel,
                eff=eff,
                ee_pos=ee_pos,
                ee_vel=ee_vel,
                ee_eff=ee_eff,
            )
        return True

    def _motor_state_to_joint_state(self, motor_state: List[MotorInfo]) -> JointStates:
        """Convert motor state to joint state.

        Args:
            motor_state (List[Any]): The motor state.

        Returns:
            Dict[str, np.ndarray]: The joint state.
        """
        names = [f"joint{i + 1}" for i in range(len(motor_state))]
        if self._gripper_index is not None:
            names[self._gripper_index] = "gripper"
        pos = np.array([motor.pos for motor in motor_state], dtype=float)
        pos = self.remapper.to_command_joint_pos_space(pos)
        vel = np.array([motor.vel for motor in motor_state], dtype=float)
        vel = self.remapper.to_command_joint_vel_space(vel)
        eff = np.array([motor.eff for motor in motor_state])
        temp_mos = np.array([motor.temp_mos for motor in motor_state])
        temp_rotor = np.array([motor.temp_rotor for motor in motor_state])
        timestamp = motor_state[0].timestamp
        return JointStates(
            names=names,
            pos=pos,
            vel=vel,
            eff=eff,
            temp_mos=temp_mos,
            temp_rotor=temp_rotor,
            timestamp=timestamp,
        )

    def _compute_gravity_compensation(self, joint_state: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
        if joint_state is None or not self.use_gravity_comp:
            return np.zeros(len(self.motor_chain))
        elif self.use_gravity_comp:
            if self._model_coordinate_adapter is not None:
                public_q = np.asarray(joint_state.pos, dtype=float)
                model_q = self._model_coordinate_adapter.public_position_to_model(public_q)
                model_torque = self.kdl.compute_inverse_dynamics(
                    model_q,
                    np.zeros(self._model_coordinate_adapter.model_nv),
                    np.zeros(self._model_coordinate_adapter.model_nv),
                )
                public_torque = self._model_coordinate_adapter.model_effort_to_public(model_torque)
                if np.max(np.abs(public_torque)) > 25.0:
                    raise RuntimeError(f"{self}: too large torques {public_torque}")
                return public_torque
            q = joint_state.pos[: self._gripper_index] if self._gripper_index is not None else joint_state.pos
            t = self.kdl.compute_inverse_dynamics(q, np.zeros(q.shape), np.zeros(q.shape))
            # print gravity torque to 2f
            if np.max(np.abs(t)) > 25.0:
                print([f"{s:.2f}" for s in t])
                raise RuntimeError(f"{self}: too large torques")
            if self._gripper_index is None:
                return self.kdl.compute_inverse_dynamics(q, np.zeros(q.shape), np.zeros(q.shape))
            else:
                t = self.kdl.compute_inverse_dynamics(q, np.zeros(q.shape), np.zeros(q.shape))
                return np.append(t, 0.0)

    # ----------------- Server Functions ----------------- #

    def num_dofs(self) -> int:
        """Get the number of joints of the robot, including the gripper.

        Returns:
            int: The number of joints of the robot.
        """
        return len(self.motor_chain)

    def get_motor_torques(self) -> Optional[np.ndarray]:
        """Return the last computed motor torques (gravity comp + any command torques)."""
        return self._last_motor_torques

    def get_joint_pos(self) -> np.ndarray:
        """Get the current state of the leader robot, including the gripper in radian.

        Returns:
            T: The current state of the leader robot.
        """
        with self._state_lock:
            return self._joint_state.pos

    def _clip_robot_joint_pos_command(self, pos: np.ndarray) -> np.ndarray:
        """Clip the robot joint pos command to the joint limits. Do not clip the gripper pos.
        Args:
            pos (np.ndarray): The joint pos command to clip.
        Returns:
            np.ndarray: The clipped joint pos command.
        """

        if self._joint_limits is not None:
            if self._gripper_index is not None:
                pos[: self._gripper_index] = np.clip(
                    pos[: self._gripper_index],
                    self._joint_limits[:, 0],
                    self._joint_limits[:, 1],
                )
            else:
                pos = np.clip(pos, self._joint_limits[:, 0], self._joint_limits[:, 1])
        return pos

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_pos (np.ndarray): The state to command the leader robot to.
        """
        self._require_legacy_command_path()
        pos = self._clip_robot_joint_pos_command(joint_pos)
        with self._command_lock:
            self._commands = JointCommands.init_all_zero(len(self.motor_chain))
            self._commands.pos = self.remapper.to_robot_joint_pos_space(pos)
            self._commands.kp = self._kp
            self._commands.kd = self._kd

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (Dict[str, np.ndarray]): The state to command the leader robot to.
        """
        self._require_legacy_command_path()
        pos = self._clip_robot_joint_pos_command(joint_state["pos"])
        vel = joint_state["vel"]
        kp = joint_state.get("kp", self._kp)
        kd = joint_state.get("kd", self._kd)
        with self._command_lock:
            self._commands = JointCommands.init_all_zero(len(self.motor_chain))
            self._commands.pos = self.remapper.to_robot_joint_pos_space(pos)
            self._commands.vel = self.remapper.to_robot_joint_vel_space(vel)
            self._commands.kp = kp
            self._commands.kd = kd

    def zero_torque_mode(self) -> None:
        self._require_legacy_command_path()
        logging.info(f"Entering zero_torque_mode for {self}")
        with self._command_lock:
            self._commands = JointCommands.init_all_zero(len(self.motor_chain))
            self._kp = np.zeros(len(self.motor_chain))
            self._kd = np.zeros(len(self.motor_chain))

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the robot.

        This is to extract all the information that is available from the robot,
        such as joint positions, joint velocities, etc. This may also include
        information from additional sensors, such as cameras, force sensors, etc.

        Returns:
            Dict[str, np.ndarray]: A dictionary of observations.
        """
        with self._state_lock:
            if self._gripper_index is None:
                result = {
                    "joint_pos": self._joint_state.pos,
                    "joint_vel": self._joint_state.vel,
                    "joint_eff": self._joint_state.eff,
                }
            else:
                result = {
                    "joint_pos": self._joint_state.pos[: self._gripper_index],
                    "joint_vel": self._joint_state.vel[: self._gripper_index],
                    "joint_eff": self._joint_state.eff[: self._gripper_index],
                    "gripper_pos": np.array([self._joint_state.pos[self._gripper_index]]),
                    "gripper_vel": np.array([self._joint_state.vel[self._gripper_index]]),
                    "gripper_eff": np.array([self._joint_state.eff[self._gripper_index]]),
                }
            if self.temp_record_flag:
                result["temp_mos"] = self._joint_state.temp_mos
                result["temp_rotor"] = self._joint_state.temp_rotor
            return result

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit the runtime context related to this object."""
        self.close()

    def move_joints(self, target_joint_positions: np.ndarray, time_interval_s: float = 2.0) -> None:
        """Move the robot to a given joint positions."""
        with self._state_lock:
            current_pos = self._joint_state.pos
        assert len(current_pos) == len(target_joint_positions)
        steps = 50  # 50 steps over time_interval_s
        for i in range(steps + 1):
            alpha = i / steps  # Interpolation factor
            target_pos = (1 - alpha) * current_pos + alpha * target_joint_positions  # Linear interpolation
            self.command_joint_pos(target_pos)
            time.sleep(time_interval_s / steps)

    def close(self) -> None:
        """Close workers and transport. This is NOT a motor-disable operation."""
        # self.move_to_zero()
        self._stop_event.set()  # Signal the thread to stop
        if self._execution_started:
            self._server_thread.join(timeout=1.0)
        try:
            self.motor_chain.close()
        finally:
            self.stop_mcap_recording()
        logging.info("Robot workers/transport closed; close does not confirm motor disable")

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._require_legacy_command_path()
        assert kp.shape == self._kp.shape == kd.shape
        self._kp = kp
        self._kd = kd

    def enter_gravity_comp_idle(self) -> None:
        """Reset active commands to gravity-comp idle.

        Sets ``self._commands`` to zeros with ``kd = self._grav_comp_kd``,
        mirroring the ``zero_gravity_mode=True`` branch in ``__init__``. Use
        this after running PD control (e.g. ``command_joint_pos``) to re-enter
        grav-comp idle without leaving the previous target pose/gains active.
        Leaves ``self._kp`` / ``self._kd`` unchanged so subsequent control
        commands still use the configured control gains.
        """
        self._require_legacy_command_path()
        with self._command_lock:
            self._commands = JointCommands.init_all_zero(len(self.motor_chain))
            self._commands.kd = self._grav_comp_kd.copy()

    def _require_legacy_command_path(self) -> None:
        if self._reference is not None:
            raise RuntimeError(
                "guarded session accepts only admitted references; direct command/mode/gain mutation refused"
            )

    def start_recording(self, save_dir: str) -> bool:
        """Start recording joint state data asynchronously."""
        if self._joint_state_saver is None:
            raise RuntimeError("Joint state saver factory not provided, recording not available")
        self._joint_state_saver.start_recording(save_dir)
        return True

    def stop_recording(self, prefix: str = "") -> Tuple[bool, str]:
        """Stop recording joint state data asynchronously."""
        if self._joint_state_saver is None:
            raise RuntimeError("Joint state saver not available")
        succ = self._joint_state_saver.stop_recording(prefix)
        if succ:
            return succ, "Recording stopped successfully"
        return succ, "Recording failed to stop"

    def start_mcap_recording(self) -> Path:
        """Start recording every motor feedback frame to a timestamped ROS 2 MCAP file."""
        with self._mcap_lock:
            if self._mcap_recorder is not None:
                raise RuntimeError(f"MCAP recording is already active: {self._mcap_recorder.path}")
            assert self._joint_state is not None
            self._mcap_recorder = RobotMcapRecorder.create(self._joint_state.names)
            return self._mcap_recorder.path

    def stop_mcap_recording(self) -> None:
        """Finish the active MCAP recording, if any."""
        with self._mcap_lock:
            recorder = self._mcap_recorder
            self._mcap_recorder = None
        if recorder is not None:
            recorder.close()


@dataclass
class _CliArgs:
    """Drive a YAM-family arm over CAN."""

    arm: str = "yam"
    """Arm variant (yam, yam_pro, yam_ultra, yam_ultra_2, big_yam)."""
    gripper: str = "linear_4310"
    """Gripper variant."""
    channel: str = "can0"
    """CAN channel."""
    operation_mode: Literal["gravity_comp", "test_gripper", "stay_current_qpos"] = "gravity_comp"
    """Operation mode: gravity compensation, gripper cycling, or holding the startup joint positions."""
    record: bool = False
    """Record motor feedback and computed required torques to a ROS 2 CDR MCAP file."""


if __name__ == "__main__":
    import tyro

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.utils.utils import override_log_level

    override_log_level(level=logging.INFO)

    args = tyro.cli(_CliArgs)

    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)

    print(f"Initializing robot with arm_type: {arm_type}, gripper_type: {gripper_type}")
    robot = get_yam_robot(args.channel, arm_type=arm_type, gripper_type=gripper_type)

    try:
        if args.record:
            print(f"Recording motor feedback to {robot.start_mcap_recording()}")
        if args.operation_mode == "gravity_comp":
            while True:
                time.sleep(1)
        elif args.operation_mode == "test_gripper":
            assert gripper_type != GripperType.YAM_TEACHING_HANDLE, (
                "test_gripper is not supported for YAM_TEACHING_HANDLE, teaching handle is a passive device"
            )
            for _ in range(30):
                for gripper_pos in [0.8, 0.0]:
                    print(f"gripper_pos: {gripper_pos}")
                    robot.command_joint_pos(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_pos]))
                    time.sleep(4)
                    print(robot.get_observations())
        elif args.operation_mode == "stay_current_qpos":
            current_qpos = robot.get_joint_pos()
            robot.command_joint_pos(current_qpos)
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()
