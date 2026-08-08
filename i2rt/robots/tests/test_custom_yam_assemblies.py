"""Complete custom YAM assembly routing and coordinate-contract regressions."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

import i2rt.robots.get_robot as get_robot_module
from i2rt.motor_drivers.utils import MotorType
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.model_coordinates import ModelCoordinateAdapter
from i2rt.robots.sim_robot import SimRobot
from i2rt.robots.utils import ArmType, GripperType, _load_arm_config, _load_gripper_config
from i2rt.utils.mujoco_utils import MuJoCoKDL

CUSTOM_GRIPPERS = (
    GripperType.LINEAR_4310_SOFT,
    GripperType.LINEAR_4310_SOFT_IPHONE_15_PRO,
    GripperType.LINEAR_4310_SOFT_IPHONE_15_PRO_MAX,
)


@pytest.mark.parametrize("gripper", CUSTOM_GRIPPERS)
def test_custom_assembly_reuses_stock_hardware_config(gripper: GripperType) -> None:
    assert gripper.hardware_config_name == GripperType.LINEAR_4310.value
    assert _load_gripper_config(gripper.value, ArmType.YAM) == _load_gripper_config(
        GripperType.LINEAR_4310.value, ArmType.YAM
    )


@pytest.mark.parametrize("gripper", CUSTOM_GRIPPERS)
def test_complete_models_compile_with_exact_interface(gripper: GripperType) -> None:
    xml_path, interface_path = gripper.get_complete_model_paths(ArmType.YAM)
    model = mujoco.MjModel.from_xml_path(xml_path)
    adapter = ModelCoordinateAdapter.from_path(interface_path, xml_path)

    assert model.nq == model.nv == model.njnt == 8
    assert adapter.public_names == ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper")
    assert Path(interface_path).parent == Path(xml_path).parent
    root = Path(xml_path).read_text()
    assert 'file="/' not in root


@pytest.mark.parametrize("gripper", CUSTOM_GRIPPERS)
def test_named_coordinate_round_trip_and_effort_sign(gripper: GripperType) -> None:
    xml_path, interface_path = gripper.get_complete_model_paths(ArmType.YAM)
    adapter = ModelCoordinateAdapter.from_path(interface_path, xml_path)
    public = np.array([0.2, 0.8, 0.6, -0.2, 0.15, 0.3, 0.5])
    model = adapter.public_position_to_model(public)

    np.testing.assert_allclose(model, [0.2, 0.8, 0.6, -0.2, 0.15, -0.3, -0.02375, -0.02375])
    np.testing.assert_allclose(adapter.model_position_to_public(model), public)
    np.testing.assert_allclose(adapter.model_velocity_to_public(adapter.public_velocity_to_model(public)), public)
    np.testing.assert_allclose(
        adapter.model_effort_to_public(np.arange(1.0, 9.0)), [1.0, 2.0, 3.0, 4.0, 5.0, -6.0, 0.0]
    )


def test_shared_jaw_coordinates_must_agree() -> None:
    xml_path, interface_path = GripperType.LINEAR_4310_SOFT.get_complete_model_paths(ArmType.YAM)
    adapter = ModelCoordinateAdapter.from_path(interface_path, xml_path)
    model = adapter.public_position_to_model(np.zeros(7))
    model[7] = -0.01
    with pytest.raises(ValueError, match="shared public coordinate 'gripper' disagree"):
        adapter.model_position_to_public(model)


def test_named_public_replacement_updates_every_mapped_jaw() -> None:
    xml_path, interface_path = GripperType.LINEAR_4310_SOFT.get_complete_model_paths(ArmType.YAM)
    adapter = ModelCoordinateAdapter.from_path(interface_path, xml_path)
    model = np.arange(8.0)
    replaced = adapter.set_public_position_in_model(model, "gripper", 0.5)
    np.testing.assert_array_equal(replaced[:6], model[:6])
    np.testing.assert_allclose(replaced[6:], [-0.02375, -0.02375])
    with pytest.raises(ValueError, match="unknown public coordinate"):
        adapter.set_public_position_in_model(model, "jaw", 0.5)


@pytest.mark.parametrize("gripper", CUSTOM_GRIPPERS)
def test_custom_sim_robot_has_seven_public_and_eight_model_coordinates(gripper: GripperType) -> None:
    robot = get_yam_robot(arm_type=ArmType.YAM, gripper_type=gripper, sim=True)
    assert isinstance(robot, SimRobot)
    target = np.array([0.2, 0.8, 0.6, -0.2, 0.15, 0.3, 0.5])
    robot.command_joint_pos(target)

    assert robot.num_dofs() == 7
    assert robot._model.nq == 8
    np.testing.assert_allclose(robot.get_joint_pos(), target)
    np.testing.assert_allclose(robot._data.qpos, [0.2, 0.8, 0.6, -0.2, 0.15, -0.3, -0.02375, -0.02375])
    assert robot.get_motor_torques() is not None
    assert robot.get_motor_torques().shape == (7,)
    assert robot.get_motor_torques()[6] == 0.0
    robot.close()


def test_custom_gravity_uses_current_jaw_position() -> None:
    robot = get_yam_robot(arm_type=ArmType.YAM, gripper_type=GripperType.LINEAR_4310_SOFT, sim=True)
    closed = np.array([0.2, 0.8, 0.6, -0.2, 0.15, 0.3, 0.0])
    opened = closed.copy()
    opened[-1] = 1.0
    robot.command_joint_pos(closed)
    closed_torque = robot.get_motor_torques().copy()
    robot.command_joint_pos(opened)
    opened_torque = robot.get_motor_torques().copy()
    assert not np.allclose(closed_torque[:6], opened_torque[:6], rtol=0.0, atol=1.0e-8)
    robot.close()


@pytest.mark.parametrize("gripper", CUSTOM_GRIPPERS)
def test_custom_gravity_feedforward_stays_within_declared_motor_torque(gripper: GripperType) -> None:
    xml_path, interface_path = gripper.get_complete_model_paths(ArmType.YAM)
    adapter = ModelCoordinateAdapter.from_path(interface_path, xml_path)
    kdl = MuJoCoKDL(xml_path)
    limits = adapter.public_position_limits(kdl.model)
    hardware = _load_arm_config(ArmType.YAM)
    torque_limits = np.array(
        [MotorType.get_motor_constants(motor_type).TORQUE_MAX for _can_id, motor_type in hardware.motor_list]
    )
    rng = np.random.default_rng(42)
    for _ in range(100):
        public = rng.uniform(limits[:, 0], limits[:, 1])
        model = adapter.public_position_to_model(public)
        model_torque = kdl.compute_inverse_dynamics(model, np.zeros(8), np.zeros(8))
        public_feedforward = adapter.model_effort_to_public(model_torque)[:6] * hardware.gravity_comp_factor
        assert np.all(np.abs(public_feedforward) < torque_limits), (
            f"gravity feedforward exceeds declared motor torque: {public_feedforward=} {torque_limits=}"
        )


@pytest.mark.parametrize("gripper", CUSTOM_GRIPPERS)
def test_custom_assemblies_reject_nonstandard_arm_and_runtime_inertial_override(gripper: GripperType) -> None:
    with pytest.raises(ValueError, match=r"supports only ArmType.YAM"):
        get_yam_robot(arm_type=ArmType.YAM_PRO, gripper_type=gripper, sim=True)
    with pytest.raises(ValueError, match="do not support runtime end-effector inertial overrides"):
        get_yam_robot(arm_type=ArmType.YAM, gripper_type=gripper, ee_mass=0.1, sim=True)


def test_stock_linear_route_still_uses_composition(monkeypatch: pytest.MonkeyPatch) -> None:
    original = get_robot_module.combine_arm_and_gripper_xml
    calls: list[tuple[ArmType, GripperType]] = []

    def recording_combine(arm: ArmType, gripper: GripperType, **kwargs: object) -> str:
        calls.append((arm, gripper))
        return original(arm, gripper, **kwargs)

    monkeypatch.setattr(get_robot_module, "combine_arm_and_gripper_xml", recording_combine)
    stock = get_yam_robot(arm_type=ArmType.YAM, gripper_type=GripperType.LINEAR_4310, sim=True)
    assert calls == [(ArmType.YAM, GripperType.LINEAR_4310)]
    assert stock.get_robot_info()["model_coordinate_schema"] is None
    stock.close()


def test_malformed_interface_is_rejected() -> None:
    xml_path, interface_path = GripperType.LINEAR_4310_SOFT.get_complete_model_paths(ArmType.YAM)
    metadata = json.loads(Path(interface_path).read_text())
    metadata["model_qpos"].pop("joint8")
    model = mujoco.MjModel.from_xml_path(xml_path)
    with pytest.raises(ValueError, match="map every model joint exactly once"):
        ModelCoordinateAdapter(metadata, model)
