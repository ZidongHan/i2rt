# Lab YAM deployment fork

This paired-checkout fork is used by the outer YAM_Deployment repository through a pinned gitlink and editable
installation. It is not an upstream hardware-startup migration. Do not push without explicit authorization;
the official upstream remote remains fetch-only.

## Patch ledger

- `1276f63`: upstream base.
- `4200bb1`, `3099506`: lab agent/API workflow documentation.
- `4c4e9e4`: complete custom YAM models, explicit public/model coordinate mapping and custom model routing.
- `6ab6e6e`: fixed-base geometry retention.
- `61fcb38`: opt-in diagnostic IK with immutable numerical evidence.
- Tracking correctness review: shared private FrameTask/iteration implementation behind `ik()` and
  `ik_with_diagnostics()`. No gain, weight, integration step, limit default, canonical model or motor change.

`ik()` continues to raise native solver exceptions and returns a mutable NumPy result. Installed Mink 1.1.0
returns a detached copy for `Configuration.q`; this is not an alias of the next solver state.
`ik_with_diagnostics()` converts `NoSolutionFound` to a failure result with immutable tuple snapshots.
Both preserve input seeds, and distinguish `limits=None` from an explicit empty list.

Exactly the three lab custom assemblies are deployment-supported. The official stock route remains a regression
path. No ROS/MoveIt, generic assembly registry, upstream startup changes or unrelated robot migration is included.
