# Standard YAM API internals: user call to DAMIAO motor

This document traces the standard six-joint YAM API in release `v1.2.4` (`5d47b35`) from the user-facing Python
call to the DAMIAO motor firmware boundary and back. It excludes other arm products and mobile-base code.

The code in this repository is the authority for the claims below. Where a behavior is only suggested by a name
or comment, that uncertainty is stated explicitly.

For task-oriented examples and units, first read the [standard YAM API guide](yam-api-guide.md).

## 1. What this repository does—and does not do—to DAMIAO firmware

The normal YAM control path does **not** upload, replace, or toggle between DAMIAO firmware images. It communicates
with firmware already running inside each motor controller:

- It enables the controller with the eight-byte special command `FF FF FF FF FF FF FF FC`.
- It sends DAMIAO MIT-mode position, velocity, `kp`, `kd`, and torque-feedforward frames.
- It decodes the controller's position, velocity, torque, temperature, and error reply.
- Maintenance utilities can clear errors, disable a motor, save its present zero, and read/write/save firmware
  registers such as timeout.

The repository's actual firmware-flashing utility,
[`i2rt/utils/can_flash.py`](../i2rt/utils/can_flash.py#L1), explicitly targets the teaching handle's `ioheart`
passive encoder, not a DAMIAO arm motor. The CAN-adapter guide also discusses flashing adapter firmware, which is
again separate from the motor controller.

Therefore “toggling a DAMIAO” in the runtime path means toggling motor state or exercising an onboard control
mode through its CAN protocol—not reflashing its firmware.

## 2. Layer map

```text
Isaac Lab policy / trajectory / teleoperation code
                 |
                 | full q or {q, qd} target
                 v
Robot surface: get_yam_robot -> MotorChainRobot
                 |
                 | clip arm q; normalize/map gripper; select kp/kd
                 | add MuJoCo gravity torque and optional friction
                 v
Motor-chain surface: DMChainCanInterface
                 |
                 | apply motor sign and software offset
                 | serialize one command/reply per motor
                 v
DMSingleMotorCanInterface
                 |
                 | quantize and pack 8-byte DAMIAO MIT frame
                 v
CanInterface -> python-can -> Linux SocketCAN -> CAN adapter
                 |
                 v
DAMIAO onboard firmware and motor-control loops
                 |
                 | reply: error, q, qd, torque, MOS temp, rotor temp
                 v
decode -> unwrap -> offset/sign -> normalize gripper -> cached observations
```

This project is the host-side real-time bridge and compensation layer between a user's controller and the
DAMIAO firmware. It is not a policy runtime with action scaling/watchdogs, and it is not embedded motor-control
firmware.

## 3. Static YAM assembly

The standard YAM hardware YAML supplies the six arm entries read by
[`_load_arm_config`](../i2rt/robots/utils.py#L114):

| Public index | Joint | CAN ID | Motor type | Default `kp` | Default `kd` | Gravity multiplier |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 0 | `joint1` | `0x01` | DM4340 | 80 | 5.0 | 1.0 |
| 1 | `joint2` | `0x02` | DM4340 | 80 | 5.0 | 1.1 |
| 2 | `joint3` | `0x03` | DM4340 | 80 | 5.0 | 1.1 |
| 3 | `joint4` | `0x04` | DM4310 | 10 | 1.5 | 1.2 |
| 4 | `joint5` | `0x05` | DM4310 | 10 | 1.5 | 1.0 |
| 5 | `joint6` | `0x06` | DM4310 | 10 | 1.5 | 1.0 |

All six configured directions are `+1`. A motorized gripper is appended at public index 6 and CAN ID `0x07`.
Its type, gains, direction, raw limits, calibration requirement, mount, and force-limiter parameters come from
the selected gripper YAML. See [`yam.yml`](../i2rt/robots/config/yam.yml) and, for the default gripper,
[`linear_4310.yml`](../i2rt/robots/config/linear_4310.yml).

The hardware has one controlled gripper motor. The combined MuJoCo model may have two equal-coupled jaw joints;
model `nq` is therefore not the CAN motor count.

## 4. Startup and background execution

### `get_yam_robot(...)`: startup sequence

The real path in [`get_yam_robot`](../i2rt/robots/get_robot.py#L248) performs these operations:

1. Read `yam.yml` and the selected gripper YAML.
2. Compose an arm-plus-gripper MJCF in `/tmp`. This is the inverse-dynamics model used for gravity compensation.
3. Load the six arm joint limits from MJCF and expand each end by 0.15 rad.
4. Build the ordered motor list, directions, gains, gravity-idle damping, friction values, and zero software offsets.
5. Open `python-can` on SocketCAN at 1 Mbit/s through
   [`CanInterface`](../i2rt/motor_drivers/can_interface.py#L10).
6. Drain stale frames and enable every motor. `motor_on(...)` sends `FF FF FF FF FF FF FF FC` to the motor's
   arbitration ID, parses the response, clears any error with `... FB`, and retries enable until normal.
7. Decode an initial position for every motor and initialize the absolute-position unwrap accumulator.
8. Read those positions once. If a reported joint lies outside `[-pi, pi]`, shift its **software** offset by one
   revolution so the exposed starting coordinate is brought near that interval. This is not a firmware zero.
9. Start the motor-chain communication thread.
10. If required, auto-calibrate the gripper by applying torque in both directions.
11. Construct `MotorChainRobot`, load the combined model into `MuJoCoKDL`, verify measured arm positions against
    the expanded limits, and start the robot update thread.
12. If `zero_gravity_mode=False`, copy measured position into a first PD hold target.

Construction can therefore enable and command physical motors before returning to the caller.

### The two control threads

There are two host threads below a user API call:

| Thread | Owner | Main work |
| --- | --- | --- |
| Robot update | `MotorChainRobot.start_server` | Copy the latest user setpoint, calculate gravity/friction, apply gripper force limiting, publish a new chain command, and copy latest chain feedback. |
| Motor communication | `DMChainCanInterface._set_torques_and_update_state` | Serialize commands to motors 1–6/7, wait for each reply, decode it, update absolute position/state, and handle faults. |

The constant `CONTROL_FREQ = 250` is used as a CAN-bandwidth design/check value, but the communication loop is not
scheduled with an exact 4 ms period. Its real rate is determined by sequential send/reply latency and short sleeps;
inspect `motor_chain.comm_freq` rather than assuming 250 Hz. The robot update thread sleeps 1 ms per iteration but
also depends on locks and Python scheduling.

### Latest setpoint, not queued trajectory

User motion calls replace `MotorChainRobot._commands`; the robot thread copies the newest value. It then calls
`DMChainCanInterface.set_commands`, which replaces the motor thread's command list. There is no internal trajectory
FIFO. Repeated policy calls should therefore represent the latest desired setpoint. If an upstream network or
inference queue is used, bound it and drop stale entries, as the `minimum_gello` example does.

## 5. DAMIAO frame protocol used here

### Outbound MIT command

For the normal YAM path, `ControlMode.MIT` adds no arbitration-ID offset, so a command to motor `n` uses standard
CAN ID `n`. [`set_control`](../i2rt/motor_drivers/dm_driver.py#L229) clips and quantizes five fields using the
selected motor type's declared range:

| Field | Width | DM4340 range | DM4310 range |
| --- | ---: | ---: | ---: |
| position | 16 bits | `[-12.5, 12.5]` rad | `[-12.5, 12.5]` rad |
| velocity | 12 bits | `[-10, 10]` rad/s | `[-30, 30]` rad/s |
| `kp` | 12 bits | `[0, 500]` | `[0, 500]` |
| `kd` | 12 bits | `[0, 5]` | `[0, 5]` |
| torque feedforward | 12 bits | `[-28, 28]` N·m | `[-10, 10]` N·m |

The eight data bytes are packed as:

```text
byte 0  = position[15:8]
byte 1  = position[7:0]
byte 2  = velocity[11:4]
byte 3  = velocity[3:0] << 4 | kp[11:8]
byte 4  = kp[7:0]
byte 5  = kd[11:4]
byte 6  = kd[3:0] << 4 | torque[11:8]
byte 7  = torque[7:0]
```

`float_to_uint` saturates each value to its protocol range before quantizing. This protects the frame encoder from
overflow but is not a robot-level safety limit.

### Send/reply transaction

[`CanInterface._send_message_get_response`](../i2rt/motor_drivers/can_interface.py#L38) sends one frame and waits
up to 10 ms for a response. With the YAM's `ReceiveMode.p16`, motor `n` is expected to reply on CAN ID `n + 16`.
An unexpected frame is consumed and the call retries; normal control uses up to 15 attempts per motor. Exhaustion
raises an assertion, which stops the motor communication thread.

There is no CAN receive filter or response queue in the real YAM path; commands are serialized specifically so the
next frame should be the corresponding reply.

### Inbound feedback

[`parse_recv_message`](../i2rt/motor_drivers/dm_driver.py#L287) decodes:

```text
byte 0 high nibble = error/status code
bytes 1..2         = 16-bit position
byte 3 + byte 4 hi = 12-bit velocity
byte 4 lo + byte 5 = 12-bit torque
byte 6             = MOS temperature in deg C
byte 7             = rotor temperature in deg C
```

The code treats status `0x1` as normal and raises for disabled, voltage/current/temperature, communication, or
overload states unless error parsing is explicitly ignored during recovery. A source TODO says the extracted error
nibble still needs double-checking; do not build a safety case around this parser without validating it against the
installed DAMIAO firmware revision.

The motor-specific numeric ranges are used again to dequantize position, velocity, and torque. The chain then:

1. unwraps position across the motor protocol's `[-12.5, 12.5]` boundary;
2. applies `(raw_position - software_offset) * direction`;
3. multiplies velocity and torque feedback by `direction`; and
4. maps gripper position/velocity from raw angle to normalized public coordinates.

No gear-ratio conversion appears in this host path. The code assumes the decoded position already has the joint
coordinate convention expected by the model, apart from sign/offset and gripper normalization.

## 6. State and metadata APIs

### `num_dofs()`

**Trace:** `MotorChainRobot.num_dofs` -> `len(DMChainCanInterface)` -> length of ordered motor list.

No CAN traffic is caused. It reports six arm motors, plus one if a motorized gripper was appended by the factory.

### `get_joint_pos()`

**Trace:** latest parsed reply -> `DMChainCanInterface.read_states` -> `_motor_state_to_joint_state` -> cached
`JointStates.pos` -> user.

No CAN traffic is caused by the getter. The motor communication thread already produced the state. Arm positions
are unwrapped/sign-corrected radians; the gripper is normalized. The real getter returns the cached array itself,
so callers should copy it.

### `get_observations()`

**Trace:** same cached `JointStates` -> split at `gripper_index` -> arm/gripper dictionaries.

No CAN traffic is caused. `joint_eff` is parsed feedback torque, not the host's last gravity torque. Timestamps and
error codes are not exposed in this dictionary, and temperatures appear only when a non-factory recording flag is
enabled.

### `get_joint_state()`

**Trace on real YAM:** inherited `Robot.get_joint_state` stub -> `None`.

There is no DAMIAO effect and no real state returned. The method is implemented only by `SimRobot` in this release.

### `get_robot_info()`, `joint_pos_spec()`, and `joint_state_spec()`

These are local metadata calls. `get_robot_info()` returns the assembled configuration and current raw gripper
limits. The spec helpers encode only vector shape and dtype. None reads or writes a DAMIAO controller.

### `get_motor_torques()`

Returns the last host-calculated outbound feedforward vector saved before the chain update. It does not read the
motor and is not the same as reply torque. The next motor thread cycle quantizes/sends that vector as the MIT torque
field.

## 7. Motion-command APIs

### `command_joint_pos(joint_pos)`

**User-visible intent:** full-chain position target using configured stiffness and damping.

**Trace:**

1. [`command_joint_pos`](../i2rt/robots/motor_chain_robot.py#L543) clips arm entries to the loaded limits.
2. `JointMapper.to_robot_joint_pos_space` changes the gripper from normalized `[0, 1]` into calibrated raw motor
   angle. Arm entries pass through.
3. It clears torque/velocity fields, writes the mapped position, and copies configured `kp`/`kd` into
   `JointCommands`.
4. The robot thread calculates `g(q)` from MuJoCo and optional signed friction, then forms
   `tau_ff = 0 + gravity_factor * g(q) + friction`.
5. The gripper force limiter may replace the raw gripper position target; raw endpoints are clipped.
6. `DMChainCanInterface.set_commands` publishes the command to the motor thread.
7. The motor thread applies direction and offset, packs `position`, zero `velocity`, `kp`, `kd`, and `tau_ff`, then
   sends an MIT frame to every motor.
8. DAMIAO firmware closes the position/velocity loop and returns feedback.

For the arm, the effective requested law is approximately:

```text
tau_motor = gravity_factor * gravity_model(q)
          + optional_coulomb_friction * sign(qd)
          + kp * (q_target - q_measured)
          + kd * (0 - qd_measured)
```

The host does not wait for convergence. Shape and finite-value checks are absent. The 0.15 rad factory adjustment
**expands** model limits, despite being called a safety buffer, so deployment code should use stricter inner limits.

### `command_joint_state({"pos": ..., "vel": ...})`

**User-visible intent:** full-chain MIT position and velocity targets with optional per-command gains.

**Trace:** identical to `command_joint_pos`, except the supplied velocity passes through
`JointMapper.to_robot_joint_vel_space` (gripper normalized speed multiplied by raw span) and occupies the MIT
velocity field. Optional `kp` and `kd` are used for that active command. Host gravity/friction feedforward is still
added.

It does **not** select DAMIAO `ControlMode.VEL`; the chain remains in MIT mode. It also does not numerically
integrate velocity into a position target. With nonzero `kp`, both position and velocity errors contribute.

The real implementation requires both `pos` and `vel`; it has no default for a missing velocity key.

### `command_target_vel(joint_vel)`

**Trace on real YAM:** inherited `Robot.command_target_vel` stub -> return with no state change -> no CAN effect.

`DMSingleMotorCanInterface` contains a low-level `ControlMode.VEL` packer, but `get_yam_robot` constructs the chain
in MIT mode and `MotorChainRobot` does not connect the public method to it. A policy that calls this method can
appear to work in `SimRobot` while commanding nothing on hardware.

### `move_joints(target, time_interval_s)`

**Trace:** sample cached start -> compute 51 linear position waypoints -> repeatedly call `command_joint_pos` and
sleep -> same MIT path as above.

It adds host-side interpolation but no additional DAMIAO mode or trajectory primitive.

## 8. Gravity, gain, and idle APIs

### Gravity compensation inside every real command

[`MuJoCoKDL.compute_inverse_dynamics`](../i2rt/utils/mujoco_utils.py#L25) loads the runtime-composed model and calls
MuJoCo inverse dynamics using measured arm `q`, zero `qdot`, and zero `qddot`. The result is checked against a
hardcoded 25 N·m maximum and multiplied by configured factors before entering the MIT torque field. The gripper
gravity entry is explicitly zero.

This compensates modeled static gravity only. It does not use measured torque to estimate a held object's mass,
external contact wrench, or disturbance.

### `enter_gravity_comp_idle()`

**Trace:** replace active `JointCommands` with zero position/velocity/torque/`kp`; set only configured small
gravity-idle `kd` -> robot thread still adds modeled gravity -> MIT frames continue for every motor.

This is an actively energized hand-guiding mode, not motor-off.

### `update_kp_kd(kp, kd)`

**Trace:** update default arrays in `MotorChainRobot` -> no immediate chain/DAMIAO change -> next
`command_joint_pos` copies them into the active command -> subsequent MIT frames carry them.

An already-active command keeps its copied gains until another command call.

### `zero_torque_mode()`

**Trace:** clear command and zero stored `kp`/`kd` -> robot thread still adds gravity/friction -> MIT frames remain
active. No `motor_off` special frame is sent.

This method can also make later `command_joint_pos` calls use zero default gains. It is not a safe synonym for
disable.

## 9. Gripper APIs below the normalized coordinate

### Position mapping

For raw endpoints `[closed, open]`, the mapper implements:

```text
raw_position = normalized_position * (open - closed) + closed
raw_velocity = normalized_velocity * (open - closed)
```

Feedback applies the inverse formulas. This handles mechanisms whose motor angle decreases while the jaws open.
After mapping, motor direction/offset are applied by the same chain path as the arm.

### Calibration

[`detect_gripper_limits`](../i2rt/robots/utils.py#L682) calls `DMChainCanInterface.set_commands` with zero torque
for all other motors and `+test_torque`, then `-test_torque`, for the gripper. The motor thread sends these values
as MIT torque feedforward with zero position/velocity gains. Position is polled until three consecutive changes
fall below the threshold or the direction times out. The observed minimum/maximum is ordered by configured motor
direction and becomes the mapper's raw endpoints.

This procedure deliberately drives against mechanical stops; the DAMIAO firmware performs torque/current control,
while the i2RT host decides when motion has stopped.

### Force limiting

The robot thread examines a 0.1 s buffer of decoded gripper effort and current speed. Once high effort and low
speed indicate blockage, it converts the configured 50 N limit into motor torque using a mechanism model, then
backs out a position target expected to produce that torque through gripper `kp`. The DAMIAO motor still receives
an ordinary MIT position command. There is no direct force-control command and no fingertip force sensor in this
path.

## 10. Model and kinematics APIs

### `combine_arm_and_gripper_xml(...)`

This is a host-only operation; it sends no CAN traffic. Its output affects hardware indirectly because the
resulting masses, centers of mass, inertias, and poses determine the gravity torque placed in every arm motor's
MIT torque field.

The function overwrites the arm's terminal body's `pos`, `quat`, and joint axis from the selected gripper YAML,
then attaches the gripper body. A stale `last_joint_mount.yam` value can therefore change arm joint-6 geometry and
the gripper pose at runtime.

`ee_mass` replaces the gripper-body mass. `ee_inertia` is unusable in v1.2.4 because it writes invalid MJCF
attribute `ipos`; this was verified by compiling the generated XML with MuJoCo 3.8.1. The existing test only parses
the XML and therefore misses the schema error.

### `Kinematics.fk(...)` and `Kinematics.ik(...)`

These run only in the host process through Mink/MuJoCo. They do not read a motor or send a command. A caller must
explicitly pass a validated IK result into a motion API, which then follows the command trace above.

The lab-fork `Kinematics.ik_with_diagnostics(...)` follows the same numerical path but returns immutable causal
evidence. `limits=None` means Mink's model configuration limit is active; `use_model_joint_limits=False` selects
an explicit empty limit list and is recorded as `disabled`. It is a diagnostic ablation, not permission to issue
an out-of-envelope result. Solver damping and frame-task Levenberg-Marquardt damping are recorded separately.

## 11. Maintenance and special DAMIAO commands

These lower-level tools are not normal policy APIs, but they are the direct motor-firmware management surface in
this repository.

### `motor_on(motor_id, motor_type)`

Sends `FF FF FF FF FF FF FF FC` on the motor ID. It parses the reply while ignoring an initial error, clears faults
if needed, retries enable, then requires normal status. The YAM factory invokes this for all motors during startup.

### `clean_error(motor_id)`

Sends `FF FF FF FF FF FF FF FB` three times without waiting for a response. Auto-recovery may call it, pause, drain
one frame, re-enable the motor, then verify all current commands again. Default factory behavior is fail-fast;
automatic re-enable is opt-in because unexpected recovery can resume motion.

### `motor_off(motor_id)`

Sends `FF FF FF FF FF FF FF FD` and waits for the expected reply. The diagnostic `ping_motors.py` uses it. Normal
`MotorChainRobot.close()` does not.

### `save_zero_position(motor_id)` and `set_zero.py`

Sends `FF FF FF FF FF FF FF FE`, then sends a zero-gain/zero-torque MIT frame to check whether feedback is near
zero. This changes a persistent motor-controller zero. It is different from
`DMChainCanInterface.set_zero_position`, which changes only the host process's software offset.

The CLI defaults across IDs 1–7 and passes `MotorType.DM4310` even though standard YAM joints 1–3 are configured
as DM4340. The position ranges currently match, but the tool should be reviewed against the actual installed motor
types before maintenance use.

### `set_timeout.py`

The CLI first disables each selected motor, then uses a separate raw register protocol on CAN ID `0x7FF`:

| Operation | Payload pattern |
| --- | --- |
| Read register | `[motor_id, 0x00, 0x33, register_id, 0, 0, 0, 0]` |
| Write register | `[motor_id, 0x00, 0x55, register_id, value as little-endian u32]` |
| Save register | `[motor_id, 0x00, 0xAA, register_id, 0, 0, 0, 0]` |

Timeout is register ID 9. Without `--timeout`, the script writes raw value `0`; with `--timeout`, it writes raw
value `8000`, saves it to controller memory, and reads it back. The code does not document that register's unit,
so do not convert `8000` to milliseconds without the matching DAMIAO manual/firmware revision. The root README's
400 ms factory-default statement is not derived from this value.

The register map mentions `control_mode`, but `control_mode` has no register address in `register_addr_map`, and no
included maintenance command changes it. Runtime MIT mode is selected by how the host formats/arbitrates commands.

## 12. Faults, recovery, and shutdown

### Communication or motor error

Any normal reply status other than `0x1` raises `RuntimeError`. With auto-recovery disabled, the motor thread stops;
the robot thread detects `motor_chain.running=False` and raises. With recovery enabled, the chain attempts up to
three rounds of clear/re-enable/verify.

The public observation dictionary omits error codes and timestamps, so a robust policy supervisor needs access to
lower-level chain state or a new explicit health API rather than inferring freshness from repeated values.

### Firmware timeout

If host frames stop, the expected safe behavior is supplied by the controller's configured timeout. The repository
README says the factory setting enters damping after 400 ms. The host does not implement an independent motor
watchdog inside `command_joint_pos`; its threads simply continue sending the last setpoint until a failure.

### `close()`

`MotorChainRobot.close` stops and joins the robot thread, calls `DMChainCanInterface.close`, stops the motor thread,
and shuts down `python-can`. It does not issue zero command, hold-at-current, or `motor_off`. The last onboard
command may remain relevant until the DAMIAO timeout behavior takes effect.

## 13. Real-versus-simulation boundary

`get_yam_robot(..., sim=True)` uses the same runtime-composed model and public dimensions, but bypasses everything
from `DMChainCanInterface` downward. In particular:

- position commands teleport state;
- velocity targets are stored rather than passed through motor dynamics;
- no quantization, offsets, sequential CAN latency, special enable/disable frames, retries, faults, calibration, or
  motor timeout is exercised;
- force limiting and real friction feedforward are absent; and
- external load/perturbation behavior is not represented by the factory's default state container.

Passing a sim-only API test is therefore evidence of shape/model compatibility, not evidence that a DAMIAO motor
will receive the intended frame.

## 14. Per-API traceability index

| User API | Primary implementation | Last i2RT step before DAMIAO | Motor effect |
| --- | --- | --- | --- |
| `get_yam_robot` | [`get_robot.py`](../i2rt/robots/get_robot.py#L133) | `motor_on`, then start MIT loop | Enables all configured controllers and starts repeated commands. |
| `num_dofs` | [`motor_chain_robot.py`](../i2rt/robots/motor_chain_robot.py#L503) | None | None. |
| `get_joint_pos` | [`motor_chain_robot.py`](../i2rt/robots/motor_chain_robot.py#L515) | Read cached decoded reply | None. |
| `get_observations` | [`motor_chain_robot.py`](../i2rt/robots/motor_chain_robot.py#L580) | Read cached decoded reply | None. |
| `get_joint_state` | Protocol stub | None | None; returns `None` on real YAM. |
| `get_robot_info` / specs | Local metadata | None | None. |
| `get_motor_torques` | Local cached outbound value | None | None until regular loop sends it. |
| `command_joint_pos` | [`motor_chain_robot.py`](../i2rt/robots/motor_chain_robot.py#L543) | `DMSingleMotorCanInterface.set_control` | MIT `q`, zero `qd`, configured gains, gravity/friction torque FF. |
| `command_joint_state` | [`motor_chain_robot.py`](../i2rt/robots/motor_chain_robot.py#L556) | `set_control` | MIT `q`, `qd`, chosen gains, gravity/friction torque FF. |
| `command_target_vel` | Protocol stub | None | None on real YAM. |
| `move_joints` | [`motor_chain_robot.py`](../i2rt/robots/motor_chain_robot.py#L615) | Repeated position path | Sequence of MIT position setpoints. |
| `enter_gravity_comp_idle` | [`motor_chain_robot.py`](../i2rt/robots/motor_chain_robot.py#L640) | `set_control` | MIT zero stiffness, small damping, gravity FF. |
| `update_kp_kd` | [`motor_chain_robot.py`](../i2rt/robots/motor_chain_robot.py#L635) | Next position command | Changes future MIT gains. |
| `zero_torque_mode` | [`motor_chain_robot.py`](../i2rt/robots/motor_chain_robot.py#L573) | `set_control` | MIT zero PD but gravity/friction can remain. |
| `combine_arm_and_gripper_xml` | [`robots/utils.py`](../i2rt/robots/utils.py#L181) | Gravity model | Indirectly changes future MIT gravity torque. |
| `Kinematics.fk/ik` | [`kinematics.py`](../i2rt/robots/kinematics.py#L11) | None | None until caller sends result. |
| `close` | [`motor_chain_robot.py`](../i2rt/robots/motor_chain_robot.py#L627) | Close CAN bus | Stops host frames; no motor-off command. |

## 15. Confirmed gaps relevant to policy deployment

These are current code properties, not hypothetical recommendations:

1. Real `get_joint_state` returns `None`; real `command_target_vel` does nothing.
2. Motion commands do not validate length, shape, dtype, finite values, timestamp, action rate, or acceleration.
3. `get_joint_pos` exposes the mutable cached array; arm command clipping may mutate the caller's array.
4. The factory expands arm limits by 0.15 rad rather than shrinking to an inner safety region.
5. `zero_torque_mode` still allows modeled gravity/friction torque and zeroes defaults used later.
6. `close()` neither transmits a zero/hold command nor the DAMIAO motor-off special frame.
7. `ee_mass` replaces, rather than adds to, gripper-body mass; dynamic payload changes are not observed.
8. `ee_inertia` creates XML that MuJoCo rejects in v1.2.4.
9. The error-nibble parser has an unresolved source TODO.
10. The automated suite is simulation/model based; there is no hardware-in-the-loop DAMIAO validation in this
    repository.

These gaps do not prevent a carefully engineered deployment, but they define what the policy adapter and external
safety supervisor must supply or what the repository must implement before relying on those surfaces.
