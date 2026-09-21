import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any, Callable, Optional

import mujoco
import numpy as np

from i2rt.motor_drivers.dm_driver import (
    CanInterface,
    DMChainCanInterface,
    EncoderChain,
    MotorChain,
    PassiveEncoderReader,
    ReceiveMode,
)
from i2rt.motor_drivers.utils import MotorType
from i2rt.robots.model_coordinates import ModelCoordinateAdapter
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import (
    ArmType,
    GripperType,
    _load_arm_config,
    combine_arm_and_gripper_xml,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GripperLimitReconciliation:
    """One motionless choice of a saved gripper calibration's periodic branch."""

    gripper_index: int
    motor_id: int
    motor_type: str
    branch_index: int
    period_rad: float
    calibration_limits_rad: tuple[float, float]
    effective_limits_rad: tuple[float, float]
    feedback_positions_rad: tuple[float, ...]
    feedback_receive_sequences: tuple[int, ...]
    endpoint_tolerance_rad: float
    calibrated_coordinate_limits_rad: tuple[float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "i2rt-gripper-limit-reconciliation/v1",
            "gripper_index": self.gripper_index,
            "motor_id": self.motor_id,
            "motor_type": self.motor_type,
            "branch_index": self.branch_index,
            "period_rad": self.period_rad,
            "calibration_limits_rad": list(self.calibration_limits_rad),
            "effective_limits_rad": list(self.effective_limits_rad),
            "feedback_positions_rad": list(self.feedback_positions_rad),
            "feedback_receive_sequences": list(self.feedback_receive_sequences),
            "endpoint_tolerance_rad": self.endpoint_tolerance_rad,
            "calibrated_coordinate_limits_rad": list(self.calibrated_coordinate_limits_rad),
        }


def _load_joint_limits_from_xml(*xml_paths: str) -> np.ndarray:
    """Parse joint limits (range attributes) from one or more XML files.

    Collects all ``<joint name="jointN" range="lo hi">`` elements across the
    given XML files.  Returns an (N, 2) array of [lower, upper] limits,
    ordered by joint name (joint1, joint2, ...).  Duplicate joint names are
    ignored (first occurrence wins).
    """
    seen: set[str] = set()
    joints: list[tuple[str, float, float]] = []
    for xml_path in xml_paths:
        logger.info(f"Loading joint limits from XML: {xml_path}")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for joint_elem in root.iter("joint"):
            name = joint_elem.get("name", "")
            range_str = joint_elem.get("range")
            if range_str and name.startswith("joint") and name not in seen:
                lo, hi = (float(x) for x in range_str.split())
                joints.append((name, lo, hi))
                seen.add(name)

    limits = np.array([[lo, hi] for _, lo, hi in joints])
    logger.info(f"  joint limits ({len(joints)} joints):")
    for name, lo, hi in joints:
        logger.info(f"    {name}: [{lo:.5f}, {hi:.5f}]")
    return limits


def get_encoder_chain(can_interface: CanInterface) -> EncoderChain:
    passive_encoder_reader = PassiveEncoderReader(can_interface)
    return EncoderChain([0x50E], passive_encoder_reader)


def _get_gripper_only_robot(
    channel: str = "can0",
    gripper_type: GripperType = GripperType.LINEAR_4310,
    sim: bool = False,
    enable_auto_recovery: bool = False,
) -> "Robot":
    """Create a gripper-only robot (no arm).

    Args:
        channel: CAN interface name (e.g. "can0"). Ignored in sim mode.
        gripper_type: Which gripper to load. Must not be NO_GRIPPER.
        sim: If True, return a SimRobot instead of connecting to real hardware.
        enable_auto_recovery: If True, the motor chain tries to clean+re-enable errored motors in its
            control loop instead of failing fast. Defaults to False (fail-fast).
    """
    if gripper_type == GripperType.NO_GRIPPER:
        raise ValueError("gripper_type cannot be NO_GRIPPER when arm_type is NO_ARM")

    xml_path = gripper_type.get_xml_path()
    # One motor drives the gripper; extra XML joints are coupled via equality constraints.
    n_dofs = 1

    nominal_arm = ArmType.YAM
    gripper_limits = gripper_type.get_gripper_limits(nominal_arm)
    gripper_needs_cal = gripper_type.get_gripper_needs_calibration(nominal_arm)

    if sim:
        from i2rt.robots.sim_robot import SimRobot

        sim_gripper_limits = gripper_limits
        if sim_gripper_limits is None:
            sim_gripper_limits = np.array([0.0, 1.0])

        return SimRobot(
            xml_path=xml_path,
            n_dofs=n_dofs,
            gripper_index=0,
            gripper_limits=sim_gripper_limits,
        )

    # --- Real hardware path ---------------------------------------------------
    motor_type = gripper_type.get_motor_type(nominal_arm)
    gripper_kp, gripper_kd = gripper_type.get_motor_kp_kd(nominal_arm)
    direction = gripper_type.get_motor_direction(nominal_arm)

    motor_chain = DMChainCanInterface(
        [[0x07, motor_type]],
        [0.0],
        [direction],
        channel,
        motor_chain_name="gripper_only",
        receive_mode=ReceiveMode.p16,
        start_thread=True,
        enable_auto_recovery=enable_auto_recovery,
    )

    return MotorChainRobot(
        motor_chain=motor_chain,
        xml_path=xml_path,
        use_gravity_comp=False,
        joint_limits=None,
        kp=np.array([gripper_kp]),
        kd=np.array([gripper_kd]),
        gripper_index=0,
        gripper_limits=gripper_limits,
        enable_gripper_calibration=gripper_needs_cal,
        gripper_type=gripper_type,
        arm_type=nominal_arm,
        zero_gravity_mode=False,
    )


@dataclass(frozen=True)
class ResolvedYamRobot:
    """Disk-only native configuration; resolving this never constructs CAN."""

    motor_list: list
    motor_offsets: list
    directions: list
    robot_kwargs: dict
    sim_joint_limits: np.ndarray
    with_teaching_handle: bool
    gripper_limit_reconciliation: GripperLimitReconciliation | None = None

    def construct(self, motor_chain: MotorChain, **execution_options: Any) -> MotorChainRobot:
        """Construct the same native controller over an explicitly supplied chain."""
        try:
            options = self.robot_kwargs | execution_options
            if self.gripper_limit_reconciliation is not None:
                options["gripper_limit_reconciliation"] = self.gripper_limit_reconciliation.as_dict()
            return MotorChainRobot(motor_chain=motor_chain, **options)
        except Exception:
            # Construction may have enabled motors already. Cleanup must not
            # hide the original failure or claim that closing disabled them.
            disable = getattr(motor_chain, "disable_motors", None)
            if callable(disable):
                try:
                    logger.error("native construction failed; disable outcomes: %s", disable())
                except Exception:
                    logger.exception("native construction cleanup could not confirm motor disable")
            try:
                motor_chain.close()
            except Exception:
                logger.exception("native construction cleanup could not close transport")
            raise


def resolve_yam_robot(
    arm_type: ArmType = ArmType.YAM,
    gripper_type: GripperType = GripperType.LINEAR_4310,
    *,
    ee_mass: Optional[float] = None,
    ee_inertia: Optional[np.ndarray] = None,
    gravity_comp_factor: Optional[np.ndarray] = None,
    gripper_limits_override: Optional[np.ndarray] = None,
    gripper_kp: Optional[float] = None,
    gripper_kd: Optional[float] = None,
) -> ResolvedYamRobot:
    """Resolve model, mappings and native defaults without motor discovery or enable."""
    if arm_type == ArmType.NO_ARM:
        raise ValueError("gripper-only routing remains in get_yam_robot")
    with_gripper = gripper_type not in (GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER)
    with_teaching_handle = gripper_type == GripperType.YAM_TEACHING_HANDLE

    hw = _load_arm_config(arm_type)
    effective_gravity_comp = hw.gravity_comp_factor if gravity_comp_factor is None else gravity_comp_factor
    if with_gripper:
        effective_gravity_comp = np.append(effective_gravity_comp, 1.0)

    model_coordinate_adapter: Optional[ModelCoordinateAdapter] = None
    if gripper_type.is_custom_complete_model:
        if ee_mass is not None or ee_inertia is not None:
            raise ValueError("complete custom assemblies do not support runtime end-effector inertial overrides")
        model_path, interface_path = gripper_type.get_complete_model_paths(arm_type)
        model_coordinate_adapter = ModelCoordinateAdapter.from_path(interface_path, model_path)
    else:
        model_path = combine_arm_and_gripper_xml(
            arm_type,
            gripper_type,
            ee_mass=ee_mass,
            ee_inertia=ee_inertia,
        )

    # Load limits for motor-driven joints only (arm joints + last wrist joint from gripper XML).
    # Real commands retain the official arm model's physical envelope.  A complete
    # custom model supplies a separate exact model envelope for simulation/IK.
    limit_paths = (arm_type.get_xml_path(),)
    if not gripper_type.is_custom_complete_model:
        limit_paths += (gripper_type.get_xml_path(),)
    all_joint_limits = _load_joint_limits_from_xml(*limit_paths)
    n_arm_joints = len(hw.motor_list)
    joint_limits = all_joint_limits[:n_arm_joints]
    if not gripper_type.is_custom_complete_model:
        # Preserve the existing stock-route command envelope byte-for-byte.
        joint_limits[:, 0] -= 0.15
        joint_limits[:, 1] += 0.15
    sim_joint_limits = joint_limits
    if model_coordinate_adapter is not None:
        complete_model = mujoco.MjModel.from_xml_path(model_path)
        sim_joint_limits = model_coordinate_adapter.public_position_limits(complete_model)[:n_arm_joints]

    # Build mutable lists from the frozen arm config, then extend for gripper.
    motor_list = [[can_id, mtype] for can_id, mtype in hw.motor_list]
    directions = list(hw.directions)
    kp = hw.kp.copy()
    kd = hw.kd.copy()
    grav_comp_kd = hw.grav_comp_kd.copy()
    coulomb_friction = hw.coulomb_friction.copy()
    motor_offsets = [0.0] * len(motor_list)

    if with_gripper:
        motor_type = gripper_type.get_motor_type(arm_type)
        default_kp, default_kd = gripper_type.get_motor_kp_kd(arm_type)
        _gripper_kp = gripper_kp if gripper_kp is not None else default_kp
        _gripper_kd = gripper_kd if gripper_kd is not None else default_kd
        logging.info(f"adding gripper motor type={motor_type}, kp={_gripper_kp}, kd={_gripper_kd}")
        motor_list.append([0x07, motor_type])
        motor_offsets.append(0.0)
        directions.append(gripper_type.get_motor_direction(arm_type))
        kp = np.append(kp, _gripper_kp)
        kd = np.append(kd, _gripper_kd)
        grav_comp_kd = np.append(grav_comp_kd, 0.0)
        coulomb_friction = np.append(coulomb_friction, 0.0)

    if gripper_limits_override is not None and with_gripper:
        gripper_limits = np.asarray(gripper_limits_override)
        gripper_needs_cal = False
    else:
        gripper_limits = gripper_type.get_gripper_limits(arm_type) if with_gripper else None
        gripper_needs_cal = gripper_type.get_gripper_needs_calibration(arm_type) if with_gripper else False

    kwargs = dict(
        xml_path=model_path,
        use_gravity_comp=True,
        gravity_comp_factor=effective_gravity_comp,
        joint_limits=joint_limits,
        kp=kp,
        kd=kd,
        grav_comp_kd=grav_comp_kd,
        coulomb_friction=coulomb_friction,
        model_coordinate_adapter=model_coordinate_adapter,
    )
    if with_gripper:
        kwargs.update(
            gripper_index=n_arm_joints,
            gripper_limits=gripper_limits,
            enable_gripper_calibration=gripper_needs_cal,
            gripper_type=gripper_type,
            arm_type=arm_type,
            limit_gripper_force=50.0,
        )
    return ResolvedYamRobot(motor_list, motor_offsets, directions, kwargs, sim_joint_limits, with_teaching_handle)


def create_yam_motor_chain(
    resolved: ResolvedYamRobot,
    channel: str,
    *,
    enable_auto_recovery: bool = False,
    guarded_startup: bool = True,
    transaction_timeout_s: float = 0.02,
) -> DMChainCanInterface:
    """ACTIVE hardware operation: enable/discover the configured motors.

    The caller must authorize/support the arm before this function. Guarded
    Startup never clears faults. Guarded mode leaves the sender stopped until
    the native controller has installed an admitted complete initial reference.
    """
    # Single pass: create chain, read positions, fix wrap-around offsets in-place, then start thread.
    motor_chain = DMChainCanInterface(
        resolved.motor_list,
        resolved.motor_offsets,
        resolved.directions,
        channel,
        motor_chain_name="yam_real",
        receive_mode=ReceiveMode.p16,
        start_thread=False,
        get_same_bus_device_driver=get_encoder_chain if resolved.with_teaching_handle else None,
        use_buffered_reader=False,
        enable_auto_recovery=enable_auto_recovery,
        guarded_startup=guarded_startup,
        transaction_timeout_s=transaction_timeout_s,
    )
    motor_states = motor_chain.read_states()
    logging.debug(f"motor_states: {motor_states}")

    logging.info(f"current_pos: {[m.pos for m in motor_states]}")
    if not guarded_startup:
        # Legacy discovery heuristic. Guarded installations already declare
        # reviewed encoder offsets and saved jaw endpoints; rewriting an offset
        # from the current pose would invalidate that calibration (jaw stroke can
        # exceed 2π) and reinterpret legitimate nonzero startup configurations.
        for idx, state in enumerate(motor_states):
            if state.pos < -np.pi:
                logging.info(f"motor {idx} pos={state.pos:.3f}, offset -2π")
                motor_chain.motor_offset[idx] -= 2 * np.pi
            elif state.pos > np.pi:
                logging.info(f"motor {idx} pos={state.pos:.3f}, offset +2π")
                motor_chain.motor_offset[idx] += 2 * np.pi

    logging.info(f"adjusted motor_offsets: {motor_chain.motor_offset.tolist()}")

    # Start the control thread with corrected offsets.
    if not guarded_startup:
        motor_chain.start_thread()
    logging.info(f"YAM initial motor_states: {motor_chain.read_states()}")

    return motor_chain


def reconcile_guarded_gripper_limits(
    resolved: ResolvedYamRobot,
    motor_chain: DMChainCanInterface,
    *,
    period_rad: float = 2 * np.pi,
    endpoint_tolerance_rad: float = 0.0,
    feedback_sample_count: int = 3,
    feedback_sweep_timeout_s: float = 0.05,
) -> ResolvedYamRobot:
    """Select one saved gripper-calibration branch from guarded startup feedback.

    This is an active guarded-startup operation: after the enable response it
    obtains additional zero-gain, zero-effort feedback sweeps while the normal
    sender remains stopped. It never changes firmware zero, software offsets or
    the supplied calibration. On any refusal it disables and closes the chain so
    no caller can accidentally continue with unreconciled endpoints.
    """

    def fail_closed() -> None:
        try:
            motor_chain.disable_motors()
        except Exception:
            logger.exception("gripper branch reconciliation could not confirm motor disable")
        try:
            motor_chain.close()
        except Exception:
            logger.exception("gripper branch reconciliation could not close transport")

    try:
        if resolved.gripper_limit_reconciliation is not None:
            raise ValueError("gripper limits have already been reconciled for this startup")
        if not motor_chain.requires_staged_start:
            raise ValueError("gripper branch reconciliation requires guarded startup before sender activation")
        if type(feedback_sample_count) is not int or feedback_sample_count < 3:
            raise ValueError("gripper branch reconciliation requires at least three feedback samples")
        if (
            not np.isfinite(period_rad)
            or period_rad <= 0
            or not np.isfinite(endpoint_tolerance_rad)
            or endpoint_tolerance_rad < 0
            or not np.isfinite(feedback_sweep_timeout_s)
            or feedback_sweep_timeout_s <= 0
        ):
            raise ValueError("gripper branch period/tolerance/feedback timeout must be finite and valid")

        gripper_index = resolved.robot_kwargs.get("gripper_index")
        limits = resolved.robot_kwargs.get("gripper_limits")
        if gripper_index is None or limits is None:
            raise ValueError("saved gripper limits are required for startup branch reconciliation")
        if gripper_index != len(resolved.motor_list) - 1 or len(motor_chain) != len(resolved.motor_list):
            raise ValueError("gripper branch reconciliation requires the resolved complete motor chain")
        calibration = np.asarray(limits, dtype=float)
        if calibration.shape != (2,) or not np.all(np.isfinite(calibration)) or calibration[0] == calibration[1]:
            raise ValueError("saved gripper limits must contain two distinct finite endpoints")
        lower, upper = sorted(float(value) for value in calibration)
        if upper - lower + 2 * endpoint_tolerance_rad >= period_rad:
            raise ValueError("saved gripper span plus endpoint tolerance must be smaller than one period")

        motor_id, motor_type = resolved.motor_list[gripper_index]
        constants = MotorType.get_motor_constants(motor_type)
        direction = resolved.directions[gripper_index]
        offset = resolved.motor_offsets[gripper_index]
        calibrated_coordinate_limits = tuple(
            sorted(
                (
                    (constants.POSITION_MIN - offset) * direction,
                    (constants.POSITION_MAX - offset) * direction,
                )
            )
        )

        positions: list[float] = []
        sequences: list[int] = []
        for sample_index in range(feedback_sample_count):
            states = (
                motor_chain.read_states()
                if sample_index == 0
                else motor_chain.acquire_startup_feedback(maximum_duration_s=feedback_sweep_timeout_s)
            )
            if len(states) != len(resolved.motor_list):
                raise RuntimeError("gripper branch feedback does not contain the resolved motor chain")
            state = states[gripper_index]
            if (
                state.id != motor_id
                or state.error_code not in ("0x1", 1)
                or state.receive_sequence <= 0
                or not np.all(np.isfinite([state.pos, state.vel, state.eff, state.received_monotonic]))
            ):
                raise RuntimeError("gripper branch feedback is unhealthy, incomplete or mismatched")
            positions.append(float(state.pos))
            sequences.append(int(state.receive_sequence))
        if any(second <= first for first, second in pairwise(sequences)):
            raise RuntimeError("gripper branch feedback did not progress across startup sweeps")

        protocol_lower, protocol_upper = calibrated_coordinate_limits
        first_k = math.floor((protocol_lower - lower) / period_rad) - 1
        last_k = math.ceil((protocol_upper - upper) / period_rad) + 1
        representable = {
            k
            for k in range(first_k, last_k + 1)
            if lower + k * period_rad >= protocol_lower and upper + k * period_rad <= protocol_upper
        }
        matching = representable.copy()
        for position in positions:
            matching &= {
                k
                for k in representable
                if lower + k * period_rad - endpoint_tolerance_rad
                <= position
                <= upper + k * period_rad + endpoint_tolerance_rad
            }
        if len(matching) != 1:
            raise RuntimeError(
                "saved gripper calibration has no unique representable startup branch: "
                f"feedback={positions}, calibration={calibration.tolist()}, "
                f"period={period_rad}, candidates={sorted(matching)}"
            )

        branch_index = matching.pop()
        effective = calibration + branch_index * period_rad
        reconciliation = GripperLimitReconciliation(
            gripper_index=gripper_index,
            motor_id=motor_id,
            motor_type=motor_type,
            branch_index=branch_index,
            period_rad=float(period_rad),
            calibration_limits_rad=tuple(float(value) for value in calibration),
            effective_limits_rad=tuple(float(value) for value in effective),
            feedback_positions_rad=tuple(positions),
            feedback_receive_sequences=tuple(sequences),
            endpoint_tolerance_rad=float(endpoint_tolerance_rad),
            calibrated_coordinate_limits_rad=calibrated_coordinate_limits,
        )
        kwargs = resolved.robot_kwargs.copy()
        kwargs["gripper_limits"] = effective
        logger.info("Selected guarded gripper calibration branch: %s", reconciliation.as_dict())
        return replace(
            resolved,
            robot_kwargs=kwargs,
            gripper_limit_reconciliation=reconciliation,
        )
    except BaseException:
        fail_closed()
        raise


def get_yam_robot(
    channel: str = "can0",
    arm_type: ArmType = ArmType.YAM,
    gripper_type: GripperType = GripperType.LINEAR_4310,
    zero_gravity_mode: bool = True,
    ee_mass: Optional[float] = None,
    ee_inertia: Optional[np.ndarray] = None,
    gravity_comp_factor: Optional[np.ndarray] = None,
    gripper_limits_override: Optional[np.ndarray] = None,
    gripper_kp: Optional[float] = None,
    gripper_kd: Optional[float] = None,
    sim: bool = False,
    joint_state_saver_factory: Optional[Callable[[], Any]] = None,
    set_realtime_and_pin_callback: Optional[Callable[[int], None]] = None,
    enable_auto_recovery: bool = False,
    use_coulomb_friction: bool = False,
) -> "Robot":
    """Create a YAM-family robot (real or sim).

    Args:
        channel: CAN interface name (e.g. "can0"). Ignored in sim mode.
        arm_type: Which arm variant to use. A hardware revision is its own variant
            (e.g. ``ArmType.YAM_ULTRA_2``). Use ``ArmType.NO_ARM`` for gripper-only.
        gripper_type: Which gripper (or NO_GRIPPER / YAM_TEACHING_HANDLE).
        zero_gravity_mode: Start in gravity-compensation mode.
        ee_mass: Optional end-effector mass override (kg) for MuJoCo inertial.
        ee_inertia: Optional 10-element inertia override [ipos(3), quat(4), diaginertia(3)].
        gravity_comp_factor: Per-joint array (6 elements, arm joints only) multiplied against gravity torques.
            Overrides the arm-type default when provided.
        gripper_limits_override: Optional [closed, open] limits. If provided, skips calibration.
        gripper_kp: Optional gripper kp override. Defaults to gripper_type's default.
        gripper_kd: Optional gripper kd override. Defaults to gripper_type's default.
        sim: If True, return a SimRobot instead of connecting to real hardware.
        enable_auto_recovery: If True, the motor chain tries to clean+re-enable errored motors in its
            control loop instead of failing fast. Defaults to False (fail-fast).
        use_coulomb_friction: If True, add the per-joint Coulomb friction feedforward (from the arm
            config) during gravity compensation. Defaults to False. Only affects real hardware; ignored
            in sim mode (SimRobot has no friction feedforward).
    """
    # --- Gripper-only path (no arm) -------------------------------------------
    if arm_type == ArmType.NO_ARM:
        return _get_gripper_only_robot(
            channel=channel, gripper_type=gripper_type, sim=sim, enable_auto_recovery=enable_auto_recovery
        )

    resolved = resolve_yam_robot(
        arm_type,
        gripper_type,
        ee_mass=ee_mass,
        ee_inertia=ee_inertia,
        gravity_comp_factor=gravity_comp_factor,
        gripper_limits_override=gripper_limits_override,
        gripper_kp=gripper_kp,
        gripper_kd=gripper_kd,
    )
    config = resolved.robot_kwargs
    with_gripper = config.get("gripper_index") is not None

    if sim:
        from i2rt.robots.sim_robot import SimRobot

        # In sim mode, grippers that need calibration have no limits yet — use [0, 1] default.
        sim_gripper_limits = config.get("gripper_limits")
        if with_gripper and sim_gripper_limits is None:
            sim_gripper_limits = np.array([0.0, 1.0])

        sim_grav_comp = np.ones(len(resolved.motor_list))

        return SimRobot(
            xml_path=config["xml_path"],
            n_dofs=len(resolved.motor_list),
            joint_limits=resolved.sim_joint_limits,
            gripper_index=config.get("gripper_index"),
            gripper_limits=sim_gripper_limits,
            gravity_comp_factor=sim_grav_comp,
            model_coordinate_adapter=config.get("model_coordinate_adapter"),
        )

    # --- Real hardware path ---------------------------------------------------

    motor_chain = create_yam_motor_chain(
        resolved, channel, enable_auto_recovery=enable_auto_recovery, guarded_startup=False
    )

    return resolved.construct(
        motor_chain,
        use_coulomb_friction=use_coulomb_friction,
        zero_gravity_mode=zero_gravity_mode,
        joint_state_saver_factory=joint_state_saver_factory,
        set_realtime_and_pin_callback=set_realtime_and_pin_callback,
    )
