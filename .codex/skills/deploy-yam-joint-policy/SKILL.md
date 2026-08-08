---
name: deploy-yam-joint-policy
description: Build, review, or debug an Isaac Lab or other RL joint-action deployment adapter for the standard six-joint i2RT YAM with an optional one-coordinate parallel gripper. Use for real-robot position or position-plus-velocity policies, sim-to-real action and observation mapping, control timing, safety envelopes, watchdogs, latest-setpoint transport, payload-aware gravity compensation, external-perturbation handling, telemetry, or staged hardware commissioning. Exclude Big YAM and do not treat SimRobot as a motor-dynamics validation.
---

# Deploy a Standard YAM Joint Policy

## Establish the Deployment Contract

1. Read [the YAM API guide](../../../docs/yam-api-guide.md) and [backend trace](../../../docs/yam-api-internals.md).
2. Inspect repository instructions, status, the policy checkpoint metadata, Isaac Lab joint names/order, action scaling, control decimation, observation normalization, and training timestep.
3. Fix the hardware scope to `ArmType.YAM` and record the exact gripper.
4. Require an explicit mapping from training joint names to `[joint1, ..., joint6]` plus one normalized gripper coordinate when motorized. Never rely on coincidental array order.
5. Define whether actions mean absolute position, position delta, or position-plus-velocity target. Reject ambiguous checkpoints.

In the lab layout, `i2rt/` and `yam-policy-deployment/` are sibling repositories under `YAM_Deployment/`.
Install both into the deployment environment as editable packages and use ordinary absolute Python imports.
Never use filesystem-relative imports such as `from ../i2rt...`, add checkout paths to `sys.path`, or persist a
machine-specific workspace path in source/configuration.

For a custom soft-finger/iPhone assembly, require the exact custom `GripperType`, its generated complete MJCF,
and its generated model-interface metadata. The official `LINEAR_4310` route remains the stock composed model.
Reject a custom assembly type with a non-YAM arm or with missing/stale generated assets.

## Build One Explicit Adapter

Implement these stages in order:

1. Read `get_observations()` and assemble named arm/gripper state explicitly.
2. Apply the checkpoint's observation normalization without changing units silently.
3. Run inference with a measured timestamp.
4. Convert action normalization to SI arm targets and normalized gripper target.
5. Map training names to the seven public coordinates explicitly. When a custom complete model participates in
   inference, IK, or inverse dynamics, apply its generated named public/model mapping; do not infer model qpos
   order from array length.
6. Reject non-finite or stale state/action.
7. Clip to a configured inner safety envelope, then limit position delta, velocity, acceleration, and jerk using measured elapsed time.
8. Publish only the newest setpoint through a bounded latest-wins handoff.
9. Log raw/scaled/clipped actions, desired and measured state, action age, loop jitter, effort, temperature, communication rate, and mode/fault transitions.

Keep policy inference timing separate from the motor communication thread. Do not enqueue an unbounded FIFO of old setpoints.

## Select the Real Hardware API

- Use `command_joint_pos(q)` for position or integrated-delta policies.
- Use `command_joint_state({"pos": q, "vel": qd})` for position-plus-velocity MIT targets.
- Do not use `command_target_vel(qd)` on real YAM; it is an inherited no-op in v1.2.4.
- Do not expose a torque policy through this public surface; no public command-torque API exists.
- Preserve the gripper as one normalized position/velocity coordinate. Do not send MuJoCo's two finger coordinates.

Remember that `command_joint_state` remains MIT position/velocity PD with host gravity feedforward. A velocity target is not a standalone velocity servo while `kp` is nonzero.

## Design Mode and Failure Transitions

Define and test a finite state machine such as:

```text
DISABLED -> ALIGN/HOLD -> POLICY -> HOLD or GRAVITY_IDLE -> DISABLED
                         \-> FAULT -> hardware stop
```

- Initialize with a measured-pose hold and a slow, bounded move to the policy start region.
- Require explicit enable after model/physical alignment confirmation.
- On missed deadline, stale feedback, invalid action, joint-envelope violation, thermal/fault condition, or low communication rate, stop policy publication and execute a prevalidated transition.
- Do not use `zero_torque_mode()` as motor-off.
- Do not assume `close()` sends motor-off; retain the DAMIAO firmware timeout and external stop.
- Add an explicit health/freshness surface if public observations are insufficient; they omit timestamps and motor error codes.

## Account for Loads and Perturbations

- Model every persistent end-effector fixture before hardware deployment.
- On the official stock-composition route, treat `ee_mass` as replacing the entire gripper-body mass, not adding
  payload mass, and do not use `ee_inertia` in v1.2.4 until its invalid `ipos` generation is fixed and
  compile-tested.
- On a custom complete-assembly route, change the canonical complete URDF and regenerate. Do not layer
  `ee_mass`/`ee_inertia` overrides or stock gripper inertials on top of it.
- Represent held objects with validated payload/model variants or a deliberately conservative controller; this layer does not estimate payload from motor effort.
- Do not claim disturbance rejection from gravity compensation. It computes static modeled gravity with zero velocity/acceleration and has no contact-wrench observer.
- Bound actions and gains for the worst credible payload and external perturbation; monitor effort and temperature.

## Preserve Sim-to-Real Honesty

Use `SimRobot` to validate dimensions, mappings, clipping, finite checks, watchdog logic, model compilation, and FK/IK. Add a more faithful motor/network simulation or hardware-in-the-loop test for timing and closed-loop claims because `SimRobot` teleports positions and bypasses DAMIAO/CAN behavior.

## Commission Progressively

1. Unit-test name/order, action scaling, limiters, latest-wins behavior, and timeout transitions.
2. Run the repository model/API suite with `ArmType.YAM` and the selected gripper. For a custom assembly, compile
   the selected complete MJCF directly and verify the seven-public/eight-model named mapping, including joint-6
   and coupled-jaw signs.
3. Replay recorded observations with injected NaN, stale timestamps, inference overruns, and dropped commands.
4. Perform a motor-disabled output inspection if the hardware setup supports it.
5. Validate a low-gain current-pose hold, then small joint motions without payload.
6. Validate bounded motions across the intended workspace while logging gravity/model residuals.
7. Add the known fixture, then representative held-object loads.
8. Introduce bounded perturbations only under an approved physical test plan and reachable emergency stop.

Never expand limits, disable timeouts, enable automatic fault recovery, or raise gains merely to make a commissioning test pass.

## Verify and Report

Run relevant focused tests, the full sim suite, lint, and `git diff --check`. Report action/observation schemas, rates, limits, transition logic, payload assumptions, sim coverage, hardware stages actually completed, and every remaining gap. Never label a sim-only pass as real-robot validation.
