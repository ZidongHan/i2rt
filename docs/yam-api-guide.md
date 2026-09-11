# Standard YAM API guide

This guide explains the public Python surface for the **standard i2RT YAM** arm. Its stock behavior was originally
reviewed at `v1.2.4` (`5d47b35`); the lab-fork complete-model extensions below were rechecked on 2026-09-08.
Other arm products and Flow Base are excluded. Hardware examples describe API effects, not a qualified lab SOP.
The intended reader understands joint coordinates but may be new to Python.

For the implementation-level path from each call to a DAMIAO CAN frame, see
[YAM API internals: user call to DAMIAO motor](yam-api-internals.md).

## Lab-fork complete custom assemblies

The implemented `LINEAR_4310_SOFT`, `LINEAR_4310_SOFT_IPHONE_15_PRO` and
`LINEAR_4310_SOFT_IPHONE_15_PRO_MAX` gripper types select complete generated models under
`i2rt/robot_models/assembled/yam/`, with named coordinate metadata. They require `ArmType.YAM`, bypass
`combine_arm_and_gripper_xml`, and reject `ee_mass`/`ee_inertia` overrides. Stock `LINEAR_4310` remains composed.

Custom public commands remain seven coordinates: six arm radians and normalized aperture. Their complete models
have eight coordinates, including reversed model joint 6 and two mapped jaws. `ModelCoordinateAdapter` maps
position, velocity and effort explicitly; raw motor endpoint calibration is a different mapping. Custom hardware
commands retain the official arm's unexpanded physical limits, while simulation/IK uses mapped complete-model
limits. Do not import the stock ±0.15-rad factory expansion into this custom physical route.

In a paired YAM Deployment workspace, its [model cookbook](../../yam-policy-deployment/docs/model-generation-and-deployment-runbook.md)
owns canonical URDF/YAML generation, and its [hardware reference](../../yam-policy-deployment/docs/hardware-reference.md)
owns the unqualified commissioning boundary. The underlying vendor API being implemented does not mean the
outer PocketNav or general policy hardware route is enabled. `SimRobot` is state/API simulation, not the outer
stepped synthetic plant and not a motor model.

### Guarded native sessions (lab fork, September 11)

The paired deployment project's [native session](../../yam-policy-deployment/docs/run-native-point-tracking.md)
uses a staged path alongside the legacy `get_yam_robot` examples below:

- `resolve_yam_robot` resolves defaults, complete models and mappings without motor I/O.
- `create_yam_motor_chain(..., guarded_startup=True)` is **active physical discovery/enable**, not preflight.
  It preserves reviewed encoder offsets, does not clear faults and leaves the repeated sender stopped.
- `ResolvedYamRobot.construct` builds the existing `MotorChainRobot` on the selected chain. Guarded execution
  requires fresh measured-state initialization and an admitted finite reference before `start_execution()`.
- `command_joint_reference` publishes an immutable timed joint reference with its precomputed braking
  continuation. The native updater evaluates it and retains the same MIT PD, gravity and jaw limiter owners.
  Legacy direct setpoint/idle methods cannot bypass an enabled guarded reference interface.
- `request_controlled_braking` selects the admitted continuation. Observed settling and operator state belong
  to the outer session, not this call. `native_execution_status` exposes publication, feedback, limiter and fault
  evidence. `disable_motors` attempts explicit individual disables and returns their acknowledgement outcomes.
  `close()` still does not substitute for supported disable.

The outer force-driven `mujoco-actuation` chain consumes these same native commands; it is not `SimRobot`.
PD-only supplies zero additive torque, without disabling physical/world gravity. Native feedback guards,
command/producer lifetimes and guarded calibration have fake-I/O tests; no physical qualification is implied.
Installation authority, source handling and the pause/recovery state machine remain in the outer project.
The following historical sections describe the legacy public factory unless explicitly stated otherwise.

## 1. The control model in one page

The standard YAM has six revolute arm joints. A motorized gripper adds one controllable coordinate:

| Assembly | Public `num_dofs()` | Public command vector |
| --- | ---: | --- |
| YAM with `no_gripper` or teaching handle | 6 | `[joint1, ..., joint6]` |
| YAM with a motorized gripper | 7 | `[joint1, ..., joint6, gripper]` |

The two gripper jaws do **not** create two independent user actions. The MuJoCo model may contain two coupled
finger joints (`joint7` and `joint8`), but one motor and one normalized public coordinate drive them together.
The repository tests this 6-versus-7 contract in
[`test_robot_variants.py`](../i2rt/robots/tests/test_robot_variants.py).

Use these units and meanings:

| Quantity | Arm joints 1–6 | Motorized gripper |
| --- | --- | --- |
| Position command/feedback | radians | normalized: `0 = closed`, `1 = open` |
| Velocity command/feedback | radians/second | normalized stroke/second |
| Effort feedback | decoded motor torque, nominally N·m | decoded gripper-motor torque, nominally N·m |

The last coordinate returned by `get_joint_pos()` is therefore **not radians** when a gripper is present. The
implementation maps it between `[0, 1]` and the calibrated raw motor-angle endpoints; see
[`JointMapper`](../i2rt/robots/utils.py).

At the motor, the DAMIAO MIT-mode command has the familiar form

```text
motor torque = torque_feedforward
             + kp * (position_target - position)
             + kd * (velocity_target - velocity)
```

The i2RT layer normally sets `torque_feedforward` to its MuJoCo gravity-compensation result, optionally plus
Coulomb-friction compensation. `kp` is position stiffness; `kd` is velocity damping.

## 2. Before allowing motion

Constructing a real robot is an active hardware operation. It opens CAN, sends the DAMIAO motor-enable command,
reads every motor, starts communication threads, may move a calibrating gripper to both hard stops, and begins
gravity compensation. Do not treat `get_yam_robot(...)` as a read-only discovery call.

Before connecting:

1. Secure the base and fixtures, clear the workspace, provide a reachable hardware emergency stop, and support
   an unexpectedly unbalanced arm.
2. Confirm the physical robot is the standard six-joint YAM and that CAN IDs `1` through `6` correspond to
   joints `1` through `6`. A motorized gripper uses ID `7`.
3. Confirm the robot model includes every persistent fixture and the gripper's **combined** mass, center of mass,
   and inertia. A payload that is picked up later is not detected or modeled automatically.
4. Bring up SocketCAN at 1 Mbit/s as described in the root [README](../README.md#can-bus-setup).
5. Start with conservative gains, bounded position/velocity increments, a command watchdog, and logging of joint
   position, velocity, effort, temperature, error, and communication rate.

The DAMIAO firmware timeout is an important final layer of protection. The repository README says the factory
default enters damping after 400 ms without commands. Do not disable it for policy deployment unless a separate,
tested safety architecture replaces it.

## 3. Constructing a YAM

### `get_yam_robot(...)`

This is the normal entry point. It returns a real `MotorChainRobot` or a `SimRobot` with a similar high-level
surface. Always select the standard arm explicitly in long-lived deployment code.

```python
import numpy as np

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType

robot = get_yam_robot(
    channel="can0",
    arm_type=ArmType.YAM,
    gripper_type=GripperType.LINEAR_4310,
    zero_gravity_mode=False,
)

try:
    expected = 7
    if robot.num_dofs() != expected:
        raise RuntimeError(f"Expected {expected} DOF, got {robot.num_dofs()}")

    q_hold = robot.get_joint_pos().copy()
    robot.command_joint_pos(q_hold)
finally:
    robot.close()
```

`zero_gravity_mode=False` installs a measured-position target and configured PD gains after the background
server has started; this is not an externally verified startup hold barrier. `True` starts with zero position stiffness plus gravity feedforward and small per-joint damping. The name
does not mean that gravity is disabled.

Important parameters from [`get_yam_robot`](../i2rt/robots/get_robot.py):

| Parameter | Meaning for the standard YAM |
| --- | --- |
| `channel` | SocketCAN name such as `can0`; ignored in simulation. |
| `arm_type` | Use `ArmType.YAM`. |
| `gripper_type` | Selects the physical/model gripper and whether a seventh motor is added. |
| `zero_gravity_mode` | Choose gravity-compensation idle (`True`) or current-pose PD hold (`False`) at startup. |
| `sim` | Returns `SimRobot`; no CAN connection. |
| `gravity_comp_factor` | Six multipliers applied to the calculated arm gravity torques. |
| `use_coulomb_friction` | Adds configured signed friction feedforward on real hardware only. |
| `gripper_limits_override` | Raw closed/open motor-angle endpoints. Skips automatic gripper calibration. |
| `gripper_kp`, `gripper_kd` | Override gripper motor stiffness/damping. |
| `enable_auto_recovery` | Clear and re-enable a faulted motor automatically. Default `False` fails fast. |
| `ee_mass` | Replaces the gripper body's modeled mass; it is not an added payload mass. |
| `ee_inertia` | Intended to replace COM/orientation/principal inertia, but is broken in v1.2.4; see below. |

#### End-effector mass and inertia

`ee_mass` changes the mass used by the runtime MuJoCo inverse-dynamics model. Supply the mass of everything
represented by the composed `gripper` body, not just the newly attached fixture. Replacing a 0.553 kg modeled
gripper with `ee_mass=0.200` makes gravity compensation believe the whole body weighs 0.200 kg.

As of v1.2.4, do **not** pass `ee_inertia`: the composition code writes `ipos`, which MuJoCo 3.8 rejects as an
unknown `<inertial>` attribute. The current XML-only unit test does not compile this override. Correct the code to
write and validate `pos`, validate the ten-element input, and add a MuJoCo compilation test before using it.
Relevant implementation: [`combine_arm_and_gripper_xml`](../i2rt/robots/utils.py) and
[`test_assembly.py`](../i2rt/robots/tests/test_assembly.py).

#### Gripper construction and calibration

The standard linear and crank configurations have no fixed raw motor endpoints and request automatic
calibration. Construction applies a test torque in both directions and detects when motion stops. Clear the
gripper workspace before calling the factory. Use `gripper_limits_override=[closed_raw, open_raw]` only with
measured, configuration-controlled endpoints for that exact mechanism.

### `ArmType` and `GripperType`

For this guide, use `ArmType.YAM`. Supported motorized grippers include `LINEAR_4310`, `LINEAR_3507`,
`CRANK_4310`, and `FLEXIBLE_4310`. `NO_GRIPPER` and `YAM_TEACHING_HANDLE` keep the arm at six controlled DOF.
Enum conversion helpers are useful for configuration files and command-line tools:

```python
from i2rt.robots.utils import ArmType, GripperType

arm = ArmType.from_string_name("yam")
gripper = GripperType.from_string_name("linear_4310")
```

## 4. Reading state and metadata

### `num_dofs()`

Returns the number of commanded motors: six for the arm alone, seven with a motorized gripper. Check this before
accepting a policy action. Do not infer it from MuJoCo `model.nq`, because two coupled finger joints can make the
model's coordinate count larger than the hardware action count.

### `get_joint_pos()`

Returns one vector in action order. For a seven-DOF assembly:

```text
[q1, q2, q3, q4, q5, q6, gripper_open_fraction]
```

On real hardware this is the latest cached motor feedback; the call itself performs no CAN transaction. Call
`.copy()` before editing it. The real implementation returns its internal NumPy array rather than a defensive
copy, so modifying the returned array can corrupt the cached state.

```python
q = robot.get_joint_pos().copy()
q[0] += np.deg2rad(1.0)
robot.command_joint_pos(q)
```

### `get_observations()`

This is the preferred policy observation surface. A motorized-gripper robot returns arm and gripper separately:

| Key | Shape | Meaning |
| --- | ---: | --- |
| `joint_pos` | `(6,)` | Arm position in rad. |
| `joint_vel` | `(6,)` | Arm velocity in rad/s. |
| `joint_eff` | `(6,)` | Direction-corrected decoded motor torque. |
| `gripper_pos` | `(1,)` | Normalized opening, nominally `[0, 1]`. |
| `gripper_vel` | `(1,)` | Normalized opening speed. |
| `gripper_eff` | `(1,)` | Decoded gripper-motor torque. |

If `temp_record_flag` was enabled on a directly constructed `MotorChainRobot`, two additional full-chain arrays
appear: `temp_mos` and `temp_rotor`. The standard factory does not expose that flag.

Build a policy vector deliberately rather than relying on dictionary order:

```python
obs = robot.get_observations()
q = np.concatenate([obs["joint_pos"], obs.get("gripper_pos", np.empty(0))])
qd = np.concatenate([obs["joint_vel"], obs.get("gripper_vel", np.empty(0))])
```

### `get_joint_state()`

Do not use this as a portable real/sim API in v1.2.4. `SimRobot` returns `{"pos": ..., "vel": ...}`, but
`MotorChainRobot` inherits the protocol's empty stub and returns `None`. Use `get_observations()` on both paths.

### `get_robot_info()`

On real hardware this reports arm/gripper type, `kp`, `kd`, gravity-idle damping, Coulomb-friction values, joint
limits, gripper raw limits, gravity multipliers, gripper index, force limit, and auto-recovery setting. In
simulation it reports only joint/gripper limits, gripper index, simulation flag, and gravity multiplier.

Treat this dictionary as implementation metadata, not a stable serialization schema. Copy arrays before changing
them; use `update_kp_kd(...)` to update real gains.

### `joint_pos_spec()` and `joint_state_spec()`

These inherited helpers create `dm_env` array specifications. They describe shape and `float32` dtype only. They
do not encode YAM joint limits, gripper normalization bounds, a control period, or a safety envelope.

### `get_motor_torques()`

Returns the most recently calculated outbound torque-feedforward vector: gravity compensation, plus any command
torque and optional friction, after software clipping. It is **not** the same as measured `joint_eff`. In the
factory's public API there is no method for setting command torque directly.

## 5. Commanding motion

### `command_joint_pos(joint_pos)`

This is the primary position-policy API. Supply the complete six- or seven-element target vector every time.

```python
control_hz = 50.0
dt = 1.0 / control_hz

target = robot.get_joint_pos().copy()
target[:6] += np.deg2rad([0.5, 0, 0, 0, 0, 0])
target[-1] = 0.5  # only when a motorized gripper is present
robot.command_joint_pos(target)
```

The call updates a shared setpoint and returns; a background loop repeatedly sends the latest setpoint to all
DAMIAO motors. It does not wait for the target to be reached. Arm positions are clipped to model limits expanded
by 0.15 rad; the gripper target is mapped from normalized space to raw calibrated motor angle and later clipped
to those raw endpoints.

The library supplies no trajectory timing, velocity/acceleration limiter, finite-value check, action-delta limit,
or policy watchdog here. A deployment wrapper must supply them. Also note that arm clipping can modify the caller's
NumPy array in place.

### `command_joint_state(joint_state)`

This is the real-hardware surface for combined position and velocity targets in DAMIAO MIT mode:

```python
q_target = robot.get_joint_pos().copy()
qd_target = np.zeros(robot.num_dofs())

robot.command_joint_state(
    {
        "pos": q_target,
        "vel": qd_target,
        # Optional full-chain gain arrays:
        # "kp": kp,
        # "kd": kd,
    }
)
```

Both `pos` and `vel` are required by the real implementation. Optional `kp` and `kd` replace the factory gains
for that command. The velocity is a **target inside a position/velocity PD command**, not a standalone integrated
velocity controller. Set `kp=0` if the intended law should have no position-restoring term, but do so only inside
a separately bounded and tested controller.

For a gripper, position and velocity are normalized before they are converted to raw motor-angle units.

### `command_target_vel(joint_vel)`

Do not use this for the real YAM in v1.2.4. `SimRobot` stores the supplied velocity, but `MotorChainRobot` inherits
an empty protocol stub, so the call returns without commanding a DAMIAO motor. Use `command_joint_state(...)` for
a velocity-bearing MIT command, or add and validate a real implementation before exposing a velocity-only policy.

### `move_joints(target_joint_positions, time_interval_s=2.0)`

This hardware-only convenience method linearly interpolates 51 position setpoints from the measured pose to the
target and sleeps between them. It blocks the caller and is useful for slow transitions into a policy's initial
pose. It does not use Ruckig, enforce a Cartesian path, or explicitly constrain velocity/acceleration beyond the
chosen interval.

## 6. Gravity compensation, gains, and idle modes

### `enter_gravity_comp_idle()`

Clears the active position target and stiffness, keeps the configured small gravity-idle `kd`, and leaves the
MuJoCo gravity feedforward active. Use it after position control when an operator should be able to guide the arm
by hand.

### `update_kp_kd(kp, kd)`

Updates the default full-chain gains used by later `command_joint_pos(...)` calls. Shapes must exactly match the
current gain arrays. It does not retroactively rewrite an already-copied active command until a new position
command is issued.

### `zero_torque_mode()`

The name is misleading. It clears the explicit command and sets the stored default `kp` and `kd` arrays to zero,
but the regular update loop still adds gravity compensation (and optional friction). It also permanently changes
the default gains for subsequent position commands. It is not a motor-disable or emergency-stop API.

### Gravity and external loads

Every update evaluates MuJoCo inverse dynamics with measured `q`, zero `qdot`, and zero `qddot`. Thus only the
modeled gravity term is requested; inertial, Coriolis, contact, and observed external-force compensation are not
computed. An object picked up after startup changes the real gravity load but not the model. External perturbations
appear only in motor feedback and the onboard PD response; this layer has no force/torque sensor observer or
disturbance estimator.

## 7. Gripper behavior

### Normalized command and feedback

Use `0` for closed and `1` for open. The calibrated raw endpoints may be in either numerical order, so never
replace the mapping with `raw = fraction * max_angle`.

### Automatic endpoint calibration

`detect_gripper_limits(...)` applies test torque in both directions, samples position, and accepts an endpoint
after three low-motion observations. It then orders the two raw endpoints according to the configured motor
direction. Calibration is open-loop torque motion against mechanical stops; keep objects and fingers clear.

### Force limiter

For a factory-created motorized gripper, the software force limit is 50 N. The limiter detects high average motor
effort combined with low speed, converts the force request to a target motor torque using either a linear or crank
mechanism model, and relaxes the position setpoint to approximate that torque through `kp`. This is a model-based
motor limit, not closed-loop fingertip-force sensing. Contact geometry, friction, linkage error, and object
compliance affect the actual grasp force.

## 8. Kinematics and model composition

### `combine_arm_and_gripper_xml(...)`

This creates a temporary combined MJCF file. It attaches the selected gripper model to the YAM terminal mount,
merges mesh assets and equality/contact sections, and optionally changes the gripper-body inertial. The path is
stored as `robot.xml_path`.

The arm model has six hinge joints. Linear/crank jaw models may contain two MuJoCo slide joints coupled by an
equality constraint, but these are visualization/dynamics coordinates rather than extra motor actions.

### `Kinematics.fk(q, site_name=None)`

Returns the selected site's 4-by-4 pose in the world/base frame. `q` must match the composed MuJoCo model's
coordinates, which can differ from the public hardware action count when coupled fingers exist. For arm-only
kinematics, compose with `NO_GRIPPER` and use a six-element `q`.

### `Kinematics.ik(...)`

Solves differential inverse kinematics with Mink. It returns `(success, q_solution)` and does not command the
robot. Pass a nearby measured `init_q`, enforce the model's joint limits, validate convergence, collision, and
action deltas, then send a separately rate-limited arm command while preserving the gripper coordinate.

`limits=None` retains Mink's default `ConfigurationLimit`; pass an empty list only for an explicitly labelled
diagnostic no-limit ablation. The lab fork also provides `Kinematics.ik_with_diagnostics(...)` with immutable
`IKDiagnosticOptions` and `IKDiagnosticResult`. Its defaults are numerically aligned with `ik(...)`, while its
result records convergence reason, iterations, residuals, seed/solution delta, Jacobian conditioning, joint-limit
margin, solver damping, frame-task damping, costs, and limit mode. This method remains kinematics-only and does
not relax command or hardware safety limits.

## 9. Simulation

```python
robot = get_yam_robot(
    arm_type=ArmType.YAM,
    gripper_type=GripperType.LINEAR_4310,
    sim=True,
)
```

Simulation is valuable for dimension, API, model-compilation, FK/IK, and joint-limit tests, but it is not a motor
or policy-deployment digital twin:

- `command_joint_pos(...)` teleports MuJoCo position rather than simulating DAMIAO PD dynamics.
- `command_joint_state(...)` teleports position and stores velocity; it does not reproduce the real MIT loop.
- The factory does not start the simulation physics thread automatically.
- CAN latency, quantization, retries, timeouts, motor faults, thermal behavior, gripper calibration, force limiting,
  external perturbations, and auto-recovery are absent or synthetic.
- The factory replaces configured gravity multipliers with ones in simulation and ignores Coulomb friction.

Use the same action/observation adapter in sim and hardware, but test the adapter's timing and safety logic with a
deliberate hardware commissioning procedure.

## 10. Recording and lifecycle

### `start_recording(...)` and `stop_recording(...)`

These methods exist only when `get_yam_robot(...)` was given a `joint_state_saver_factory`. They delegate to that
external saver. For simple recording, the repository example samples `get_joint_pos()` into NumPy arrays instead.

### `close()`

Always call `close()` in `finally`. It stops the i2RT background loops and closes SocketCAN. In v1.2.4 it does
**not** send DAMIAO `motor_off` (`FF FF FF FF FF FF FF FD`) and does not first transmit a zero-torque or hold
command, despite its docstring/console message. The motor firmware timeout therefore remains important after a
process exits. Do not use process termination as a controlled stop.

## 11. Recommended RL policy adapter

A deployment boundary should perform, in order:

1. Verify `ArmType.YAM`, selected gripper, expected DOF, joint order, and model identity.
2. Read `get_observations()` and reject stale, missing, non-finite, faulted, or overheated state.
3. Convert the Isaac Lab action into explicit SI targets; keep the gripper as one normalized coordinate.
4. Clip to a **deployment safety envelope inside** the physical/model limits, not merely the library's expanded
   limits.
5. Limit position delta, velocity, acceleration, and jerk at the actual measured control period.
6. Use a bounded latest-setpoint buffer; do not replay a stale FIFO backlog after a slow policy step.
7. For position actions, call `command_joint_pos(...)`. For position-plus-velocity actions, call
   `command_joint_state(...)`; do not call the no-op real `command_target_vel(...)`.
8. On policy timeout or invalid action, command a prevalidated hold or gravity-compensation-idle transition, then
   use a separate supervisor/hardware stop when required.
9. Log desired/actual `q`, `qd`, effort, temperatures, action clipping, loop age/jitter, communication frequency,
   and every mode/fault transition.
10. Commission progressively: offline model tests, `SimRobot` API tests, motor-disabled dry run, supported arm,
    low gains/speed without payload, then bounded payload and perturbation tests.

## 12. API support summary

| API | Real YAM | `SimRobot` | Recommended use |
| --- | --- | --- | --- |
| `num_dofs` | Yes | Yes | Verify 6/7 action contract. |
| `get_joint_pos` | Yes, cached; returns internal array | Yes, copied | Prefer `.copy()` everywhere. |
| `get_joint_state` | Stub returns `None` | Yes | Avoid; use observations. |
| `get_observations` | Yes | Yes | Primary policy state. |
| `command_joint_pos` | MIT position/velocity PD + gravity FF | Teleport | Position policy. |
| `command_joint_state` | MIT position + velocity targets + gravity FF | Teleport/store velocity | Position/velocity policy with care. |
| `command_target_vel` | Stub/no-op | Stores velocity | Do not use as portable API. |
| `move_joints` | Yes | No | Slow hardware transition only. |
| `update_kp_kd` | Yes | No | Real default-gain update. |
| `enter_gravity_comp_idle` | Yes | No; sim has separate enable method | Return real arm to hand-guided idle. |
| `zero_torque_mode` | Not truly zero torque | No | Avoid as a stop API. |
| `get_motor_torques` | Calculated outbound FF | Calculated gravity torque | Diagnostics, not measured effort. |
| `close` | Stops threads/bus; no motor-off frame | Stops sim physics thread | Always call, but keep external safety. |

## 13. Source map

- Public protocol: [`i2rt/robots/robot.py`](../i2rt/robots/robot.py)
- YAM factory and configuration assembly: [`i2rt/robots/get_robot.py`](../i2rt/robots/get_robot.py)
- Real robot behavior: [`i2rt/robots/motor_chain_robot.py`](../i2rt/robots/motor_chain_robot.py)
- Simulation behavior: [`i2rt/robots/sim_robot.py`](../i2rt/robots/sim_robot.py)
- YAM hardware parameters: [`i2rt/robots/config/yam_v1.yml`](../i2rt/robots/config/yam_v1.yml)
- Gripper mapping/calibration/force limiting: [`i2rt/robots/utils.py`](../i2rt/robots/utils.py)
- Kinematics: [`i2rt/robots/kinematics.py`](../i2rt/robots/kinematics.py)
- Physical YAM properties: [`i2rt/robot_models/arm/yam/v1/README.md`](../i2rt/robot_models/arm/yam/v1/README.md)
