---
name: update-yam-payload-dynamics
description: Update, review, or validate standard i2RT YAM end-effector, fixture, gripper, or payload properties used by URDF, stock runtime-composed MJCF, custom complete MJCF, MuJoCo gravity compensation, simulation, and policy deployment. Use when adding a non-negligible fixture, changing gripper geometry or inertials, representing held objects, correcting mass, center of mass, or inertia, diagnosing gravity-compensation error, or keeping YAM models aligned. Exclude Big YAM and require measured properties or explicitly accepted engineering estimates with recorded provenance rather than invented dynamics.
---

# Update Standard YAM Payload Dynamics

## Establish the Physical Contract

1. Read [the YAM API guide](../../../docs/yam-api-guide.md), [backend trace](../../../docs/yam-api-internals.md), and [YAM physical properties](../../../i2rt/robot_models/arm/yam/README.md).
2. Read the repository's [`align-urdf-mjcf`](../align-urdf-mjcf/SKILL.md) skill before changing URDF, arm MJCF, terminal frames, or shared mounts; use it for those edits.
3. Fix scope to `ArmType.YAM` and identify the exact gripper/config/composed model.
4. Separate:
   - a persistent rigid fixture that always belongs in the robot model;
   - a selectable tool that needs explicit model variants; and
   - a transient grasped object whose pose/mass may change during a task.
5. Require mass, COM location and frame, inertia tensor about the COM and its frame, and the rigid attachment
   transform. Record whether each value is measured, CAD-derived, or an explicitly accepted uniform-density or
   scaled-inertia estimate. Stop rather than inventing missing dynamics from appearance.

For the lab workspace, expect `i2rt/` and `yam-policy-deployment/` as siblings under `YAM_Deployment/`. Resolve
repository and package resources without storing the machine-specific workspace path in source models or code.

## Audit the Runtime Path

Identify the model route before tracing runtime behavior:

- For the official stock YAM route, trace `get_yam_robot` -> `combine_arm_and_gripper_xml` -> `MuJoCoKDL` ->
  `MotorChainRobot._compute_gravity_compensation` -> MIT torque feedforward.
- For a YAM_Deployment custom assembly, trace its custom `GripperType` -> generated complete MJCF and
  model-interface metadata -> named public/model coordinate adapter -> `MuJoCoKDL` -> gravity compensation ->
  MIT torque feedforward. This route must bypass `combine_arm_and_gripper_xml`.

Confirm which body receives every inertial and whether each change replaces or adds to existing properties.


Remember:

- The official arm MJCF is composed with an external gripper at runtime; a custom complete assembly is not.

- `last_joint_mount.yam` from every gripper YAML overwrites the arm terminal `pos`, `quat`, and joint axis.
- `ee_mass` replaces the composed gripper-body mass; supply a combined total only when that representation is physically valid.
- `ee_inertia` is broken in v1.2.4 because it writes unsupported MJCF attribute `ipos`; fix it to `pos`, validate input, and compile the generated model before allowing use.
- Gravity compensation evaluates static inverse dynamics with zero `qdot` and `qddot`; it does not identify a held object or estimate external wrench.

## Combine Rigid-Body Properties Correctly

Express every component in one chosen gripper/tool frame. For component `i` with mass `m_i`, COM `c_i`, and COM-frame inertia rotated into the common frame as `I_i`:

```text
M = sum(m_i)
c = sum(m_i * c_i) / M
I_total_at_c = sum(I_i + m_i * ((d_i . d_i) * eye(3) - outer(d_i, d_i)))
d_i = c_i - c
```

Apply each component's rigid rotation to its inertia before the parallel-axis term. Verify units are kg, m, and kg*m^2. Verify symmetry, positive eigenvalues, principal-moment triangle inequalities, and a normalized orientation.

For MJCF, write `<inertial pos="..." mass="..." quat="w x y z" diaginertia="...">`. Diagonalize the common-frame tensor with a symmetric eigendecomposition, enforce a right-handed eigenvector basis, and reconstruct the tensor to verify the conversion. Do not confuse MuJoCo `wxyz` with `xyzw`.

## Choose the Durable Representation

- Put permanent arm-link fixtures in the authoritative URDF link with measured visual/collision/inertial data, then regenerate and align the arm MJCF.
- Keep the official six-joint YAM arm MJCF arm-only; do not copy gripper/finger/tool bodies into its terminal
  placeholder.
- For a custom complete assembly, put permanent gripper, finger, mount, case, and installed-phone properties in
  its canonical complete URDF and regenerate the complete MJCF literally. Do not apply `ee_mass`, copy stock
  inertials, or hand-edit the generated MJCF.
- Put permanent gripper/tool hardware in the appropriate gripper/tool model and preserve one public gripper action.
- Use explicit payload/tool variants when held-object dynamics are known and discrete.
- Avoid a mass-only override for production when COM or inertia also changes materially.
- Never mutate a loaded model and assume the active real robot's gravity solver has reloaded it; reconstruct the robot/model through a controlled disabled transition.

## Preserve Kinematics and Public DOF

- Keep YAM arm joints exactly `joint1` through `joint6` in order.
- Keep terminal mount transforms and axes consistent across all standard-YAM gripper configs affected by an arm-frame change.
- Preserve `tcp_site`/`grasp_site` placement unless the physical TCP changes.
- Keep motorized gripper hardware at seven public DOF even when the model contains two coupled finger joints.
- Compare transforms as matrices and inertias as reconstructed tensors, not Euler angles or raw quaternion signs.

## Validate Before Hardware

1. Compile the edited standalone and runtime-composed MJCF with the supported MuJoCo version.
2. Add a regression that compiles any `ee_mass`/`ee_inertia` override path; XML parsing alone is insufficient.
3. Run standard-YAM URDF/MJCF home and posed alignment tests when arm data changes.
4. Run assembly, robot-variant, control-interface, kinematics, and gravity-torque tests for `ArmType.YAM` and every affected gripper.
5. Numerically compare old/new mass, COM, tensor, TCP transform, joint frames, and gravity torque over representative configurations.
6. Check gravity torque against the relevant DAMIAO motor limits with conservative margin; do not rely only on the repository's broad threshold.
7. Run the full sim suite, lint, and `git diff --check`.

For a custom complete assembly, also verify that changing only an accepted URDF mass/tensor and regenerating
changes the compiled model and representative gravity torques while leaving kinematics and named-coordinate
mapping unchanged. Evaluate inverse dynamics at the current mapped jaw opening, not an implicit default pose.

Use commands such as:

```bash
uv run pytest i2rt/robots/tests/test_assembly.py -v
uv run pytest i2rt/robots/tests/test_urdf_mjcf_alignment.py -k "yam" -v
uv run pytest i2rt/robots/tests/test_urdf_mjcf_posed_alignment.py -k "yam" -v
uv run pytest i2rt/robots/tests/test_gravity_comp.py -k "yam" -v
uv run pytest -n auto
ruff check .
git diff --check
```

## Commission the Physical Change

Require a physical test plan with the arm supported, low gains/speed, reachable emergency stop, and live position/velocity/effort/temperature/communication logging. Validate unloaded fixture behavior first, then known payloads, then bounded perturbations. Compare measured holding effort against model predictions and stop on persistent residuals, heating, oscillation, or saturation.

## Report

Report source measurements and frames, combined mass/COM/tensor, files and bodies changed, public/model DOF, maximum kinematic and inertial residuals, gravity-torque deltas/margins, tests run, hardware stages actually completed, and any transient-payload behavior still outside the model.
