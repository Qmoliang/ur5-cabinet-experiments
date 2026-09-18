"""Extracted original v4.3 ellipsoid dependency; see docs/source_manifest.json."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import mujoco
import numpy as np
DT = 0.02
JOINT_NAMES = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')
ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class BoxObstacle:
    name: str
    center: tuple[float, float, float]
    half_size: tuple[float, float, float]
    material: str = 'obstacle'

@dataclass(frozen=True)
class SceneDefinition:
    name: str
    description: str
    boxes: tuple[BoxObstacle, ...]
    q0: tuple[float, ...]
    waypoints: tuple[tuple[float, float, float], ...]
    duration: float

@dataclass(frozen=True)
class CertificateSphere:
    body_id: int
    body_name: str
    source_geom_id: int
    local_center: np.ndarray
    radius: float

def build_model(scene: SceneDefinition) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_xml(scene))

def set_configuration(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> None:
    data.qpos[:len(JOINT_NAMES)] = np.asarray(q, dtype=float)
    data.qvel[:] = 0.0
    if model.nu:
        data.ctrl[:] = data.qpos[:model.nu]
    mujoco.mj_forward(model, data)

def attachment_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return data.site_xpos[model.site('attachment_site').id].copy()

def _quat_matrix(quat: np.ndarray) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)

def build_robot_certificate(model: mujoco.MjModel, axial_spacing: float=0.035) -> list[CertificateSphere]:
    certificate: list[CertificateSphere] = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id == 0:
            continue
        geom_type = int(model.geom_type[geom_id])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            half = np.asarray(model.geom_size[geom_id], dtype=float)
            counts = np.maximum(1, np.ceil(2.0 * half / 0.024).astype(int))
            step = 2.0 * half / counts
            rotation = _quat_matrix(model.geom_quat[geom_id])
            body_name = model.body(body_id).name
            axes = [np.linspace(-half[axis] + 0.5 * step[axis], half[axis] - 0.5 * step[axis], counts[axis]) for axis in range(3)]
            cover_radius = float(np.linalg.norm(0.5 * step))
            for x in axes[0]:
                for y in axes[1]:
                    for z in axes[2]:
                        local = model.geom_pos[geom_id] + rotation @ np.array([x, y, z])
                        certificate.append(CertificateSphere(body_id=body_id, body_name=body_name, source_geom_id=geom_id, local_center=local.copy(), radius=cover_radius))
            continue
        if geom_type not in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            continue
        radius = float(model.geom_size[geom_id, 0])
        half_length = float(model.geom_size[geom_id, 1])
        count = max(2, int(np.ceil(2.0 * half_length / axial_spacing)) + 1)
        axial = np.linspace(-half_length, half_length, count)
        spacing = 0.0 if count == 1 else float(axial[1] - axial[0])
        cover_radius = float(np.sqrt(radius * radius + 0.25 * spacing * spacing))
        rotation = _quat_matrix(model.geom_quat[geom_id])
        body_name = model.body(body_id).name
        for z in axial:
            local = model.geom_pos[geom_id] + rotation @ np.array([0.0, 0.0, z])
            certificate.append(CertificateSphere(body_id=body_id, body_name=body_name, source_geom_id=geom_id, local_center=local.copy(), radius=cover_radius))
    return certificate

def certificate_world_state(model: mujoco.MjModel, data: mujoco.MjData, certificate: list[CertificateSphere]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(certificate)
    positions = np.zeros((count, 3))
    jacobians = np.zeros((count, 3, model.nv))
    radii = np.fromiter((sphere.radius for sphere in certificate), dtype=float)
    jac_rot = np.zeros((3, model.nv))
    for index, sphere in enumerate(certificate):
        rotation = data.xmat[sphere.body_id].reshape(3, 3)
        position = data.xpos[sphere.body_id] + rotation @ sphere.local_center
        positions[index] = position
        mujoco.mj_jac(model, data, jacobians[index], jac_rot, position, sphere.body_id)
    return (positions, jacobians, radii)

def certificate_world_positions(data: mujoco.MjData, certificate: list[CertificateSphere]) -> np.ndarray:
    positions = np.empty((len(certificate), 3), dtype=float)
    for index, sphere in enumerate(certificate):
        rotation = data.xmat[sphere.body_id].reshape(3, 3)
        positions[index] = data.xpos[sphere.body_id] + rotation @ sphere.local_center
    return positions

def build_xml(scene):
    return (ROOT / 'assets' / 'drawer.xml').read_text(encoding='utf-8')
