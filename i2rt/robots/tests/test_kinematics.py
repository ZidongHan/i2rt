import time

import mink
import numpy as np
import pytest

from i2rt.robots.kinematics import IKDiagnosticOptions, Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml


@pytest.fixture
def kinematics_yam() -> Kinematics:
    combined_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER)
    return Kinematics(combined_path, "grasp_site")


def test_fk(kinematics_yam: Kinematics) -> None:
    q = np.zeros(6)
    pose = kinematics_yam.fk(q)
    assert pose.shape == (4, 4), "FK should return a 4x4 matrix"

    # Add more assertions based on expected pose values
    rotation = pose[:3, :3]
    translation = pose[:3, 3]

    start_rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    start_trans = np.array([0.1105973, 0.0000010, 0.1735018])
    np.testing.assert_allclose(rotation, start_rot, atol=1e-5)
    np.testing.assert_allclose(translation, start_trans, atol=1e-5)


def test_expired_diagnostic_ik_does_not_call_qp(kinematics_yam: Kinematics, monkeypatch: pytest.MonkeyPatch) -> None:
    seed = np.zeros(6)
    pose = kinematics_yam.fk(seed)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("expired diagnostic solve must not enter a QP")

    monkeypatch.setattr(mink, "solve_ik", forbidden)
    result = kinematics_yam.ik_with_diagnostics(pose, "grasp_site", init_q=seed, deadline_at=time.monotonic() - 1)
    assert not result.success and result.failure_reason == "deadline_exceeded"
    assert result.iterations == 0
    np.testing.assert_array_equal(result.solution, seed)


def test_late_qp_result_is_not_integrated_or_reported_converged(
    kinematics_yam: Kinematics, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = np.zeros(6)
    pose = kinematics_yam.fk(seed)

    def slow(*args: object, **kwargs: object) -> np.ndarray:
        time.sleep(0.02)
        return np.ones(6)

    monkeypatch.setattr(mink, "solve_ik", slow)
    result = kinematics_yam.ik_with_diagnostics(pose, "grasp_site", init_q=seed, deadline_at=time.monotonic() + 0.005)
    assert not result.success and result.failure_reason == "deadline_exceeded"
    np.testing.assert_array_equal(result.solution, seed)


def test_ik_smoke(kinematics_yam: Kinematics) -> None:
    q = np.ones(6)
    pose = kinematics_yam.fk(q)
    success, q_ik = kinematics_yam.ik(pose, "grasp_site")
    assert success, "IK should succeed"
    assert q_ik.shape == (6,), "IK should return a joint configuration of size 6"


def test_cycle(kinematics_yam: Kinematics) -> None:
    for _ in range(10):
        q = np.random.uniform(0, np.pi / 2, 6)
        pose = kinematics_yam.fk(q)
        q_init_for_ik = q + np.random.uniform(-0.1, 0.1, 6)
        success, q_ik = kinematics_yam.ik(pose, "grasp_site", init_q=q_init_for_ik)
        assert success, f"IK failed for target pose {pose}, init_q: {q_init_for_ik}"
        pose_reconstructed = kinematics_yam.fk(q_ik)
        np.testing.assert_allclose(pose, pose_reconstructed, atol=1e-4)


def test_diagnostic_defaults_preserve_legacy_ik_solution() -> None:
    combined_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER)
    target_kinematics = Kinematics(combined_path, "grasp_site")
    target = target_kinematics.fk(np.full(6, 0.2))
    seed = np.full(6, 0.25)
    legacy = Kinematics(combined_path, "grasp_site")
    diagnostic = Kinematics(combined_path, "grasp_site")
    success, solution = legacy.ik(target, "grasp_site", init_q=seed)
    result = diagnostic.ik_with_diagnostics(target, "grasp_site", init_q=seed)
    assert success is result.success is True
    assert result.failure_reason == "converged"
    assert result.limits_mode == "model_default"
    np.testing.assert_array_equal(solution, np.asarray(result.solution))
    assert result.position_residual_norm <= result.position_threshold
    assert result.orientation_residual_norm <= result.orientation_threshold
    assert result.iterations > 0


@pytest.mark.parametrize(
    "gripper",
    (
        GripperType.LINEAR_4310_STOCK,
        GripperType.LINEAR_4310_SOFT,
        GripperType.LINEAR_4310_SOFT_IPHONE_15_PRO,
        GripperType.LINEAR_4310_SOFT_IPHONE_15_PRO_MAX,
    ),
)
def test_diagnostics_cover_complete_custom_yam_models(gripper: GripperType) -> None:
    model_path, _interface_path = gripper.get_complete_model_paths(ArmType.YAM)
    kinematics = Kinematics(model_path, "grasp_site")
    seed = kinematics._configuration.model.qpos0.copy()
    target = kinematics.fk(seed)
    result = kinematics.ik_with_diagnostics(target, "grasp_site", init_q=seed)
    assert result.success
    assert len(result.solution) == 8
    assert result.jacobian_minimum_singular_value > 0.0
    assert result.jacobian_condition_estimate is not None
    assert result.minimum_joint_limit_margin is not None


def test_diagnostic_failure_and_limit_ablation_are_explicit(kinematics_yam: Kinematics) -> None:
    seed = np.zeros(6)
    target = kinematics_yam.fk(seed)
    target[0, 3] += 0.1
    result = kinematics_yam.ik_with_diagnostics(
        target,
        "grasp_site",
        init_q=seed,
        options=IKDiagnosticOptions(max_iters=1, use_model_joint_limits=False),
    )
    assert result.success is False
    assert result.failure_reason == "maximum_iterations"
    assert result.iterations == 1
    assert result.limits_mode == "disabled"
    assert result.position_residual_norm > result.position_threshold
    assert len(result.seed) == len(result.solution) == len(result.joint_delta) == 6


@pytest.mark.parametrize("limits", [None, []])
def test_legacy_and_diagnostic_ownership_and_explicit_limits(kinematics_yam: Kinematics, limits: object) -> None:
    seed = np.full(6, 0.25)
    original = seed.copy()
    target = kinematics_yam.fk(np.full(6, 0.2))
    target[0, 3] += 0.3
    success, live_solution = kinematics_yam.ik(target, "grasp_site", init_q=seed, limits=limits, max_iters=2)
    snapshot = live_solution.copy()
    result = kinematics_yam.ik_with_diagnostics(
        target, "grasp_site", init_q=seed, limits=limits, options=IKDiagnosticOptions(max_iters=2)
    )
    assert success is result.success is False
    np.testing.assert_array_equal(snapshot, result.solution)
    np.testing.assert_array_equal(seed, original)
    assert not np.shares_memory(live_solution, kinematics_yam._configuration.data.qpos)
    kinematics_yam.fk(np.zeros(6))
    np.testing.assert_array_equal(snapshot, live_solution)
    np.testing.assert_array_equal(snapshot, result.solution)
    assert result.limits_mode == ("model_default" if limits is None else "disabled")


def test_solver_failure_preserves_distinct_public_contracts(
    kinematics_yam: Kinematics, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = kinematics_yam.fk(np.zeros(6))

    def fail(*args: object, **kwargs: object) -> None:
        raise mink.NoSolutionFound("quadprog")

    monkeypatch.setattr(mink, "solve_ik", fail)
    with pytest.raises(mink.NoSolutionFound):
        kinematics_yam.ik(target, "grasp_site")
    diagnostic = kinematics_yam.ik_with_diagnostics(target, "grasp_site")
    assert diagnostic.success is False
    assert diagnostic.failure_reason == "qp_no_solution"
    assert diagnostic.iterations == 0
