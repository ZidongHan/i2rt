---
name: use-yam-api
description: Use, extend, review, or debug the standard six-joint i2RT YAM Python API and trace its real or simulated behavior through MotorChainRobot, gravity compensation, gripper mapping, the DAMIAO MIT driver, and CAN feedback. Use for YAM API integrations, examples, state/command adapters, fault analysis, real-versus-sim discrepancies, gripper DOF questions, or changes whose motor-level effect must be established. Exclude Big YAM, Flow Base, and unrelated products unless the user explicitly expands scope.
---

# Use the Standard YAM API

## Establish Scope

1. Read repository instructions and inspect the dirty worktree.
2. Fix the arm scope to `ArmType.YAM`; do not generalize results from another arm variant.
3. Identify the selected gripper, real versus `sim=True`, and whether the task reads state, commands motion, changes models, or diagnoses CAN behavior.
4. Read [the user guide](../../../docs/yam-api-guide.md). Read [the backend trace](../../../docs/yam-api-internals.md) whenever motor effect, timing, safety, faults, or implementation changes matter.

In the lab workspace, the vendor fork is `YAM_Deployment/i2rt/` and deployment code is in the sibling
`YAM_Deployment/yam-policy-deployment/`. Install the fork as an editable package and import `i2rt` normally.
Never use a parent-directory Python import, append the checkout to `sys.path`, or persist the absolute checkout
path in code, model configuration, or generated assets.

Distinguish model construction routes:

- Official grippers, including `LINEAR_4310`, retain the existing arm-plus-gripper composition route.
- Planned soft-finger/iPhone custom gripper types resolve to a generated complete YAM assembly plus generated
  model-interface metadata and bypass `combine_arm_and_gripper_xml`.

Do not treat the planned custom enum/model route as available until its implementation phase and tests exist.
## Preserve the Coordinate Contract

- Use action order `[joint1, ..., joint6]`, followed by one gripper coordinate only for a motorized gripper.
- Expect six public DOF without an active gripper and seven with one.
- Keep arm position/velocity in rad and rad/s.
- Keep the gripper as one normalized coordinate: `0 = closed`, `1 = open`; do not expose the two coupled jaw joints as independent policy actions.
- Do not infer hardware DOF from MuJoCo `nq`.

## Choose a Supported Surface

| Need | Use | Avoid |
| --- | --- | --- |
| Portable state | `get_observations()` | Real `get_joint_state()`, which returns `None` |
| Full position target | `command_joint_pos(q)` | Direct driver access when the robot layer suffices |
| Position and velocity targets | `command_joint_state({"pos": q, "vel": qd})` | Real `command_target_vel()`, which is a no-op |
| Return real arm to hand-guided idle | `enter_gravity_comp_idle()` | `zero_torque_mode()` as a disable operation |
| Read measured effort | `get_observations()["joint_eff"]` | `get_motor_torques()`, which is outbound feedforward |

Use complete six- or seven-element command vectors. Copy `get_joint_pos()` before editing it. Validate shape, finite values, timestamp/freshness, inner joint bounds, position delta, velocity, acceleration, and command age outside the library.

## Trace Motor Effects

For any command change, follow this exact chain:

1. Trace `get_yam_robot()` and confirm whether the selected type uses stock composition or a custom complete
   model and named coordinate adapter.
2. Locate the public method in `i2rt/robots/motor_chain_robot.py`.
3. Check arm clipping and `JointMapper` gripper conversion.
4. Check gravity, friction, and gripper-force modification in `MotorChainRobot.update()`.
5. Check shared-command replacement and the two independent background threads.
6. Check sign/offset conversion in `DMChainCanInterface._set_commands()`.
7. Check field clipping, quantization, and packing in `DMSingleMotorCanInterface.set_control()`.
8. Check SocketCAN retry/response-ID behavior and feedback decoding.
9. State whether the API sends a DAMIAO frame immediately, changes a future repeated frame, reads cached state,
   or has no hardware effect.

Do not describe this repository as flashing DAMIAO firmware. Normal control talks to existing motor firmware. `can_flash.py` flashes the teaching-handle encoder, not an arm motor.

## Respect Real-versus-Sim Differences

- Treat `SimRobot.command_joint_pos()` as teleportation, not DAMIAO PD dynamics.
- Treat sim velocity as stored state, not proof of real velocity control.
- Expect no CAN latency, quantization, timeout, motor error, calibration, force-limiter, or thermal fidelity in simulation.
- Use sim for API dimensions, composition, FK/IK, and bounded adapter tests; require staged hardware validation for motor behavior.

## Keep Safety Claims Precise

- Treat construction as active: it enables motors, starts loops, and may calibrate the gripper against its stops.
- Treat `enter_gravity_comp_idle()` as energized gravity compensation.
- Treat `zero_torque_mode()` as zero PD plus remaining gravity/friction, not motor-off.
- Treat `close()` as thread/bus shutdown without an explicit DAMIAO motor-off or final zero command.
- Preserve the firmware timeout and external emergency-stop assumptions.
- Flag that the factory expands model joint limits by 0.15 rad; impose stricter deployment limits.

## Validate Changes

Run the narrowest relevant tests first, then the full simulation suite:

```bash
uv run pytest i2rt/robots/tests/test_robot_variants.py -k "yam" -v
uv run pytest i2rt/robots/tests/test_control_interface.py -v
uv run pytest -n auto
ruff check .
git diff --check
```

Add focused tests for every corrected real/sim contract. Do not claim hardware validation from the existing sim-only suite.

## Report

Report the public coordinate/units, real and sim effects, exact DAMIAO path, tests run, and unresolved safety or hardware-validation gaps. Link claims to source lines or the two YAM documents.
