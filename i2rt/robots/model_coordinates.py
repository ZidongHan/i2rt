"""Named public/model coordinate mapping for complete assembled robot models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import mujoco
import numpy as np


@dataclass(frozen=True)
class _PositionEntry:
    model_name: str
    public_name: str
    scale: float
    offset: float
    qpos_address: int
    dof_address: int


@dataclass(frozen=True)
class _EffortEntry:
    model_name: str
    public_name: str
    scale: float
    dof_address: int


class ModelCoordinateAdapter:
    """Map an exact named public vector to and from an exact named MuJoCo model.

    A many-to-one public mapping is supported for coupled jaws.  Model-to-public
    conversion verifies that every model coordinate which shares a public source
    agrees; it never selects one silently.
    """

    SCHEMA_VERSION = "yam-coordinate-map/v1"

    def __init__(self, metadata: Mapping[str, Any], model: mujoco.MjModel):
        if metadata.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(f"unsupported model coordinate schema: {metadata.get('schema_version')!r}")
        self.metadata = dict(metadata)
        self.assembly = str(metadata.get("assembly", ""))
        if not self.assembly:
            raise ValueError("model coordinate metadata needs an assembly")
        self.public_names = tuple(metadata.get("public_names", ()))
        self.model_joint_names = tuple(metadata.get("model_joint_names", ()))
        if not self.public_names or len(self.public_names) != len(set(self.public_names)):
            raise ValueError("public coordinate names must be non-empty and unique")
        actual_model_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or "" for index in range(model.njnt)
        )
        if self.model_joint_names != actual_model_names:
            raise ValueError(
                f"model joint order mismatch: metadata={self.model_joint_names}, model={actual_model_names}"
            )
        if len(actual_model_names) != len(set(actual_model_names)):
            raise ValueError("model joint names must be unique")
        if model.nq != model.njnt or model.nv != model.njnt:
            raise ValueError("named coordinate adapter requires one scalar position and velocity per model joint")

        raw_positions = metadata.get("model_qpos")
        if not isinstance(raw_positions, Mapping) or set(raw_positions) != set(actual_model_names):
            raise ValueError("model_qpos must map every model joint exactly once")
        position_entries = []
        entries_by_public: dict[str, list[_PositionEntry]] = {name: [] for name in self.public_names}
        for model_name in actual_model_names:
            item = raw_positions[model_name]
            public_name = str(item.get("source", ""))
            if public_name not in entries_by_public:
                raise ValueError(f"model joint {model_name!r} names unknown public source {public_name!r}")
            scale = self._finite(item.get("scale"), label=f"{model_name}.scale")
            offset = self._finite(item.get("offset"), label=f"{model_name}.offset")
            if scale == 0.0:
                raise ValueError(f"{model_name}.scale must not be zero")
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, model_name)
            entry = _PositionEntry(
                model_name=model_name,
                public_name=public_name,
                scale=scale,
                offset=offset,
                qpos_address=int(model.jnt_qposadr[joint_id]),
                dof_address=int(model.jnt_dofadr[joint_id]),
            )
            position_entries.append(entry)
            entries_by_public[public_name].append(entry)
        missing_public = [name for name, entries in entries_by_public.items() if not entries]
        if missing_public:
            raise ValueError(f"public coordinates have no model position source: {missing_public}")

        raw_efforts = metadata.get("model_effort_to_public")
        if not isinstance(raw_efforts, Mapping):
            raise ValueError("model_effort_to_public must be an object")
        effort_entries = []
        effort_targets: set[str] = set()
        for model_name, item in raw_efforts.items():
            if model_name not in actual_model_names:
                raise ValueError(f"effort map names absent model joint {model_name!r}")
            public_name = str(item.get("target", ""))
            if public_name not in self.public_names or public_name in effort_targets:
                raise ValueError(f"invalid or duplicate public effort target {public_name!r}")
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, model_name)
            effort_entries.append(
                _EffortEntry(
                    model_name=model_name,
                    public_name=public_name,
                    scale=self._finite(item.get("scale"), label=f"{model_name}.effort_scale"),
                    dof_address=int(model.jnt_dofadr[joint_id]),
                )
            )
            effort_targets.add(public_name)
        self._position_entries = tuple(position_entries)
        self._entries_by_public = {name: tuple(entries) for name, entries in entries_by_public.items()}
        self._effort_entries = tuple(effort_entries)
        self._public_index = {name: index for index, name in enumerate(self.public_names)}
        self.model_nq = int(model.nq)
        self.model_nv = int(model.nv)

    @staticmethod
    def _finite(value: Any, *, label: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must be finite") from error
        if not np.isfinite(result):
            raise ValueError(f"{label} must be finite")
        return result

    @classmethod
    def from_path(cls, metadata_path: str | Path, xml_path: Optional[str | Path] = None) -> "ModelCoordinateAdapter":
        path = Path(metadata_path).expanduser().resolve(strict=True)
        metadata = json.loads(path.read_text())
        if xml_path is None:
            model_name = metadata.get("model_xml")
            if not isinstance(model_name, str) or Path(model_name).name != model_name:
                raise ValueError("model_interface model_xml must be a sibling filename")
            xml_path = path.parent / model_name
        model = mujoco.MjModel.from_xml_path(str(Path(xml_path).expanduser().resolve(strict=True)))
        return cls(metadata, model)

    def _public_vector(self, value: np.ndarray, *, label: str) -> np.ndarray:
        result = np.asarray(value, dtype=float)
        if result.shape != (len(self.public_names),) or not np.all(np.isfinite(result)):
            raise ValueError(f"{label} must be {len(self.public_names)} finite values in {self.public_names} order")
        return result

    def _model_vector(self, value: np.ndarray, *, velocity: bool, label: str) -> np.ndarray:
        size = self.model_nv if velocity else self.model_nq
        result = np.asarray(value, dtype=float)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{label} must be {size} finite named model values")
        return result

    def public_position_to_model(self, public_position: np.ndarray) -> np.ndarray:
        public = self._public_vector(public_position, label="public_position")
        model = np.zeros(self.model_nq)
        for entry in self._position_entries:
            model[entry.qpos_address] = entry.scale * public[self._public_index[entry.public_name]] + entry.offset
        return model

    def public_velocity_to_model(self, public_velocity: np.ndarray) -> np.ndarray:
        public = self._public_vector(public_velocity, label="public_velocity")
        model = np.zeros(self.model_nv)
        for entry in self._position_entries:
            model[entry.dof_address] = entry.scale * public[self._public_index[entry.public_name]]
        return model

    def model_position_to_public(self, model_position: np.ndarray, *, atol: float = 1.0e-8) -> np.ndarray:
        model = self._model_vector(model_position, velocity=False, label="model_position")
        public = np.empty(len(self.public_names))
        for public_name, entries in self._entries_by_public.items():
            values = np.array([(model[entry.qpos_address] - entry.offset) / entry.scale for entry in entries])
            if not np.allclose(values, values[0], rtol=0.0, atol=atol):
                raise ValueError(f"model positions for shared public coordinate {public_name!r} disagree: {values}")
            public[self._public_index[public_name]] = values[0]
        return public

    def set_public_position_in_model(
        self, model_position: np.ndarray, public_name: str, public_value: float
    ) -> np.ndarray:
        """Return a model vector with every named coordinate for one public DOF replaced."""
        model = self._model_vector(model_position, velocity=False, label="model_position").copy()
        if public_name not in self._entries_by_public:
            raise ValueError(f"unknown public coordinate {public_name!r}; expected one of {self.public_names}")
        value = self._finite(public_value, label=f"{public_name}.position")
        for entry in self._entries_by_public[public_name]:
            model[entry.qpos_address] = entry.scale * value + entry.offset
        return model

    def model_velocity_to_public(self, model_velocity: np.ndarray, *, atol: float = 1.0e-8) -> np.ndarray:
        model = self._model_vector(model_velocity, velocity=True, label="model_velocity")
        public = np.empty(len(self.public_names))
        for public_name, entries in self._entries_by_public.items():
            values = np.array([model[entry.dof_address] / entry.scale for entry in entries])
            if not np.allclose(values, values[0], rtol=0.0, atol=atol):
                raise ValueError(f"model velocities for shared public coordinate {public_name!r} disagree: {values}")
            public[self._public_index[public_name]] = values[0]
        return public

    def model_effort_to_public(self, model_effort: np.ndarray) -> np.ndarray:
        model = self._model_vector(model_effort, velocity=True, label="model_effort")
        public = np.zeros(len(self.public_names))
        for entry in self._effort_entries:
            public[self._public_index[entry.public_name]] = entry.scale * model[entry.dof_address]
        return public

    def public_position_limits(self, model: mujoco.MjModel) -> np.ndarray:
        """Return the intersection of mapped model ranges in public coordinates."""
        limits = np.empty((len(self.public_names), 2), dtype=float)
        for public_name, entries in self._entries_by_public.items():
            lower, upper = -np.inf, np.inf
            for entry in entries:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry.model_name)
                model_lower, model_upper = model.jnt_range[joint_id]
                candidates = np.array(
                    [(model_lower - entry.offset) / entry.scale, (model_upper - entry.offset) / entry.scale]
                )
                lower = max(lower, float(np.min(candidates)))
                upper = min(upper, float(np.max(candidates)))
            if not lower < upper:
                raise ValueError(f"mapped model range for {public_name!r} is empty")
            limits[self._public_index[public_name]] = [lower, upper]
        return limits

    def model_effort_with_public_factors(self, model_effort: np.ndarray, public_factors: np.ndarray) -> np.ndarray:
        """Scale mapped arm model forces using factors declared in public order."""
        effort = self._model_vector(model_effort, velocity=True, label="model_effort").copy()
        factors = self._public_vector(public_factors, label="public_factors")
        effort[:] = 0.0
        raw = np.asarray(model_effort, dtype=float)
        for entry in self._effort_entries:
            effort[entry.dof_address] = raw[entry.dof_address] * factors[self._public_index[entry.public_name]]
        return effort
