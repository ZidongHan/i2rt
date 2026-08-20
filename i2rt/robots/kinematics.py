import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mink
import mujoco
import numpy as np

from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml


@dataclass(frozen=True)
class IKDiagnosticOptions:
    """Numerical choices for the lab-fork diagnostic IK entry point."""

    dt: float = 0.01
    solver: str = "quadprog"
    pos_threshold: float = 1e-4
    ori_threshold: float = 1e-4
    damping: float = 1e-4
    frame_task_lm_damping: float = 1.0
    position_cost: float = 1.0
    orientation_cost: float = 1.0
    max_iters: int = 200
    use_model_joint_limits: bool = True

    def validate(self) -> None:
        finite_positive = {
            "dt": self.dt,
            "pos_threshold": self.pos_threshold,
            "ori_threshold": self.ori_threshold,
            "position_cost": self.position_cost,
            "orientation_cost": self.orientation_cost,
        }
        for name, value in finite_positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"IK diagnostic option {name} must be positive and finite")
        for name, value in {
            "damping": self.damping,
            "frame_task_lm_damping": self.frame_task_lm_damping,
        }.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"IK diagnostic option {name} must be non-negative and finite")
        if not isinstance(self.solver, str) or not self.solver:
            raise ValueError("IK diagnostic option solver must be a non-empty string")
        if isinstance(self.max_iters, bool) or not isinstance(self.max_iters, int) or self.max_iters <= 0:
            raise ValueError("IK diagnostic option max_iters must be a positive integer")
        if type(self.use_model_joint_limits) is not bool:
            raise ValueError("IK diagnostic option use_model_joint_limits must be boolean")


@dataclass(frozen=True)
class IKDiagnosticResult:
    """Immutable causal trace for one differential-IK solve."""

    success: bool
    failure_reason: str
    iterations: int
    solver: str
    dt: float
    solver_damping: float
    frame_task_lm_damping: float
    position_cost: float
    orientation_cost: float
    position_threshold: float
    orientation_threshold: float
    maximum_iterations: int
    limits_mode: str
    seed: Tuple[float, ...]
    solution: Tuple[float, ...]
    joint_delta: Tuple[float, ...]
    position_residual: Tuple[float, ...]
    orientation_residual: Tuple[float, ...]
    position_residual_norm: float
    orientation_residual_norm: float
    jacobian_minimum_singular_value: float
    jacobian_condition_estimate: Optional[float]
    minimum_joint_limit_margin: Optional[float]


class Kinematics:
    def __init__(self, xml_path: str, site_name: Optional[str]):
        """Initialize the Kinematics object.

        Args:
            xml_path (str): Path to the MuJoCo XML model file.
            site_name (Optional[str]): Name of the site for which to compute the forward kinematics.
        """
        model = mujoco.MjModel.from_xml_path(xml_path)
        self._configuration = mink.Configuration(model)
        self._site_name = site_name

    def fk(self, q: np.ndarray, site_name: Optional[str] = None) -> np.ndarray:
        """Compute the forward kinematics for the given joint configuration.

        Args:
            q (np.ndarray): The joint configuration.
            site_name (Optional[str]): Name of the site for which to compute the forward kinematics.
                       If not provided, the default site name is used.

        Returns:
            (np.ndarray): Site frame in world frame. Shape: (4, 4)
        """
        self._configuration.update(q)
        site_name = site_name or self._site_name
        assert site_name is not None, "site_name must be provided"
        return self._configuration.get_transform_frame_to_world(site_name, "site").as_matrix()

    def ik(
        self,
        target_pose: np.ndarray,
        site_name: str,
        init_q: Optional[np.ndarray] = None,
        limits: Optional[List[mink.Limit]] = None,
        dt: float = 0.01,
        solver: str = "quadprog",
        pos_threshold: float = 1e-4,
        ori_threshold: float = 1e-4,
        damping: float = 1e-4,
        max_iters: int = 200,
        verbose: bool = False,
    ) -> Tuple[bool, np.ndarray]:
        """Differential ik solver, leverging mink.

        Args:
            target_pose (np.ndarray): The target pose to reach.
            site_name (str): Name of the desired site.
            limits (List[mink.Limit]): List of limits to enforce.
            init_q (Optional[np.ndarray]): Initial joint configuration.
            dt (float): Integration timestep in [s].
            solver (str): Quadratic program solver.
            pos_threshold (float): Position threshold for convergence.
            ori_threshold (float): Orientation threshold for convergence.
            damping (float): Levenberg-Marquardt damping.
            max_iters (int): Maximum number of iterations.
            verbose (bool): Whether to print debug information.

        Returns:
            Tuple[bool, np.ndarray]: Success flag and the converged joint configuration.
        """
        if init_q is not None:
            self._configuration.update(init_q)

        end_effector_task = mink.FrameTask(
            frame_name=site_name,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )

        end_effector_task.set_target(mink.SE3.from_matrix(target_pose))
        tasks = [end_effector_task]

        start_time = time.time()  # Start timing

        for j in range(max_iters):
            vel = mink.solve_ik(self._configuration, tasks, dt, solver, damping=damping, limits=limits)
            self._configuration.integrate_inplace(vel, dt)
            err = end_effector_task.compute_error(self._configuration)

            pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
            if pos_achieved and ori_achieved:
                end_time = time.time()  # End timing
                elapsed_time = end_time - start_time
                if verbose:
                    print(
                        f"Exiting after {j} iterations, configuration: {self._configuration.q}, time taken: {elapsed_time:.4f} seconds"
                    )
                return True, self._configuration.q

        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        if verbose:
            print(
                f"Failed to converge after {max_iters} iterations, time taken: {elapsed_time:.4f} seconds, pos_err: {err[:3]}, rot_err: {err[3:]}"
            )
        return False, self._configuration.q

    def ik_with_diagnostics(
        self,
        target_pose: np.ndarray,
        site_name: str,
        init_q: Optional[np.ndarray] = None,
        limits: Optional[List[mink.Limit]] = None,
        options: Optional[IKDiagnosticOptions] = None,
    ) -> IKDiagnosticResult:
        """Solve IK and retain immutable numerical diagnostics.

        This deliberately named lab-fork extension does not replace :meth:`ik`.
        Its default options reproduce the legacy method's numerical defaults.
        Passing ``limits=None`` uses Mink's model configuration limits when
        ``use_model_joint_limits`` is true; an empty effective list is the
        explicit diagnostic no-limit ablation.
        """
        options = IKDiagnosticOptions() if options is None else options
        options.validate()
        target = np.asarray(target_pose, dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("IK diagnostic target_pose must be a finite 4x4 transform")
        if init_q is not None:
            self._configuration.update(np.asarray(init_q, dtype=float))
        seed = self._configuration.q.copy()
        effective_limits: Optional[List[mink.Limit]]
        if limits is None:
            effective_limits = None if options.use_model_joint_limits else []
            limits_mode = "model_default" if options.use_model_joint_limits else "disabled"
        else:
            effective_limits = limits
            limits_mode = "disabled" if len(limits) == 0 else "explicit"

        end_effector_task = mink.FrameTask(
            frame_name=site_name,
            frame_type="site",
            position_cost=options.position_cost,
            orientation_cost=options.orientation_cost,
            lm_damping=options.frame_task_lm_damping,
        )
        end_effector_task.set_target(mink.SE3.from_matrix(target))
        tasks = [end_effector_task]
        success = False
        failure_reason = "maximum_iterations"
        iterations = 0
        for iteration in range(options.max_iters):
            try:
                velocity = mink.solve_ik(
                    self._configuration,
                    tasks,
                    options.dt,
                    options.solver,
                    damping=options.damping,
                    limits=effective_limits,
                )
            except mink.NoSolutionFound:
                failure_reason = "qp_no_solution"
                break
            self._configuration.integrate_inplace(velocity, options.dt)
            iterations = iteration + 1
            error = end_effector_task.compute_error(self._configuration)
            if (
                np.linalg.norm(error[:3]) <= options.pos_threshold
                and np.linalg.norm(error[3:]) <= options.ori_threshold
            ):
                success = True
                failure_reason = "converged"
                break

        error = end_effector_task.compute_error(self._configuration)
        jacobian = end_effector_task.compute_jacobian(self._configuration)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        minimum_singular = float(singular_values[-1])
        condition = None if minimum_singular <= np.finfo(float).eps else float(singular_values[0] / minimum_singular)
        solution = self._configuration.q.copy()
        joint_delta = solution - seed
        margin = self._minimum_joint_limit_margin(solution)
        return IKDiagnosticResult(
            success=success,
            failure_reason=failure_reason,
            iterations=iterations,
            solver=options.solver,
            dt=options.dt,
            solver_damping=options.damping,
            frame_task_lm_damping=options.frame_task_lm_damping,
            position_cost=options.position_cost,
            orientation_cost=options.orientation_cost,
            position_threshold=options.pos_threshold,
            orientation_threshold=options.ori_threshold,
            maximum_iterations=options.max_iters,
            limits_mode=limits_mode,
            seed=tuple(float(value) for value in seed),
            solution=tuple(float(value) for value in solution),
            joint_delta=tuple(float(value) for value in joint_delta),
            position_residual=tuple(float(value) for value in error[:3]),
            orientation_residual=tuple(float(value) for value in error[3:]),
            position_residual_norm=float(np.linalg.norm(error[:3])),
            orientation_residual_norm=float(np.linalg.norm(error[3:])),
            jacobian_minimum_singular_value=minimum_singular,
            jacobian_condition_estimate=condition,
            minimum_joint_limit_margin=margin,
        )

    def _minimum_joint_limit_margin(self, configuration: np.ndarray) -> Optional[float]:
        model = self._configuration.model
        limited_joint_ids = np.flatnonzero(model.jnt_limited)
        if len(limited_joint_ids) == 0:
            return None
        margins = []
        for joint_id in limited_joint_ids:
            qpos_address = model.jnt_qposadr[joint_id]
            lower, upper = model.jnt_range[joint_id]
            value = configuration[qpos_address]
            margins.append(min(value - lower, upper - value))
        return float(min(margins))


def main() -> None:
    combined_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER)
    mj_model = Kinematics(combined_path, "grasp_site")
    q = np.zeros(6)
    pose = mj_model.fk(q)
    print(pose)

    pose[0, 3] -= 0.1
    pose[2, 3] += 0.1
    print(pose)
    q_ik = mj_model.ik(pose, "grasp_site")
    print(q_ik)


if __name__ == "__main__":
    main()
