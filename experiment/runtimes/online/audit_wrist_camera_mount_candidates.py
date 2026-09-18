"""Evaluate physically external wrist-camera mounts before freezing a scene.

This is a sensor/hardware audit only. It does not run a controller, search a
path, alter any task target, or expose box geometry to proxy construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from depth_camera_perception import UR5MountedDepthCamera, environment_endpoint_mask
from model import (
    DT,
    JOINT_NAMES,
    build_model,
    build_robot_certificate,
    certificate_world_positions,
    set_configuration,
)
from native_occupancy import NativeIncrementalOccupancyMap
from protocol_drawer_scene import (
    CAMERA_HZ,
    INITIAL_CALIBRATED_FREE_PADDING,
    protocol_drawer_v4_scene,
)
from protocol_liuqp_controller import ProtocolLiuQPController
from run_protocol_v3_online_ablation import (
    MAP_VOXEL_SIZE,
    _first_nonfree_swept_voxel,
)


ROOT = Path(__file__).resolve().parent
STATIC_GATE = (
    ROOT / "formal_results" / "protocol_v4" / "static_geometry_gate"
    / "static_geometry_gate.json"
)
CAMERA_TO_HOUSING_FRONT = 0.012575
CANDIDATE_HOUSING_LOCAL_Y = (0.030, 0.040, 0.050, 0.060, 0.070)
CANDIDATE_LOCAL_Z = (-0.065,)


def _strict_initial_step(model, data, scene, observations) -> dict:
    """Check the task-directed leading shell using only the initial depth map."""

    robot = build_robot_certificate(model)
    radii = np.asarray([sphere.radius for sphere in robot], dtype=float)
    before = certificate_world_positions(data, robot)
    occupancy = NativeIncrementalOccupancyMap(MAP_VOXEL_SIZE)
    try:
        occupancy.integrate(observations)
        occupancy.mark_current_robot_free(
            before, radii, padding=INITIAL_CALIBRATED_FREE_PADDING
        )
        controller = ProtocolLiuQPController(
            model,
            data,
            scene,
            robot,
            np.zeros((0, 3), dtype=float),
            representation="sphere",
            obstacle_radii=np.zeros(0, dtype=float),
            proxy_ids=np.zeros(0, dtype=np.int64),
        )
        qdot, metrics = controller.solve(np.asarray(scene.waypoints[-1], dtype=float))
        q0 = data.qpos[: len(JOINT_NAMES)].copy()
        scale = 1.0
        certificate = None
        after = before
        for trial in range(1, 9):
            set_configuration(model, data, q0 + scale * qdot * DT)
            after = certificate_world_positions(data, robot)
            set_configuration(model, data, q0)
            certificate = occupancy.certify_swept_spheres(before, after, radii)
            if certificate.safe:
                return {
                    "safe": True,
                    "accepted_scale": scale,
                    "trials": trial,
                    "unknown_voxels": certificate.unknown_voxels,
                    "occupied_voxels": certificate.occupied_voxels,
                    "qdot_norm": float(np.linalg.norm(qdot)),
                    "qp_status": metrics.status,
                }
            scale *= 0.5
        blockers = []
        for index, sphere in enumerate(robot):
            item = occupancy.certify_swept_spheres(
                before[index : index + 1],
                after[index : index + 1],
                radii[index : index + 1],
            )
            if not item.safe:
                blockers.append(
                    {
                        "sphere_index": index,
                        "body_name": sphere.body_name,
                        "unknown_voxels": item.unknown_voxels,
                        "occupied_voxels": item.occupied_voxels,
                    }
                )
        return {
            "safe": False,
            "accepted_scale": 0.0,
            "trials": 8,
            "unknown_voxels": certificate.unknown_voxels,
            "occupied_voxels": certificate.occupied_voxels,
            "qdot_norm": float(np.linalg.norm(qdot)),
            "qp_status": metrics.status,
            "blocking_spheres": blockers,
        }
    finally:
        occupancy.close()


def _strict_open_space_rollout(model, data, scene, camera, cycles: int = 20) -> dict:
    """Sensor observability gate with LiuQP task motion and no obstacles.

    Removing obstacle proxies here cannot manufacture success in the formal
    comparison: this audit asks only whether locally measured free space can
    certify the physical robot's leading shell.  The chosen mount is frozen
    before sphere/ellipsoid obstacle trials.
    """

    set_configuration(model, data, np.asarray(scene.q0, dtype=float))
    robot = build_robot_certificate(model)
    radii = np.asarray([sphere.radius for sphere in robot], dtype=float)
    occupancy = NativeIncrementalOccupancyMap(MAP_VOXEL_SIZE)
    controller = ProtocolLiuQPController(
        model,
        data,
        scene,
        robot,
        np.zeros((0, 3), dtype=float),
        representation="sphere",
        obstacle_radii=np.zeros(0, dtype=float),
        proxy_ids=np.zeros(0, dtype=np.int64),
    )
    last_camera_tick = -1
    accepted = 0
    minimum_scale = 1.0
    first_blocker = None
    try:
        for cycle in range(cycles):
            before = certificate_world_positions(data, robot)
            occupancy.mark_current_robot_free(
                before,
                radii,
                padding=(
                    INITIAL_CALIBRATED_FREE_PADDING if cycle == 0 else 0.0
                ),
            )
            camera_tick = int(np.floor(cycle * DT * CAMERA_HZ + 1.0e-12))
            if camera_tick != last_camera_tick:
                occupancy.integrate(camera.capture(data))
                occupancy.mark_current_robot_free(
                    before,
                    radii,
                    padding=(
                        INITIAL_CALIBRATED_FREE_PADDING if cycle == 0 else 0.0
                    ),
                )
                last_camera_tick = camera_tick
            qdot, metrics = controller.solve(
                np.asarray(scene.waypoints[-1], dtype=float)
            )
            q0 = data.qpos[: len(JOINT_NAMES)].copy()
            scale = 1.0
            certificate = None
            after = before
            for trial in range(1, 9):
                set_configuration(model, data, q0 + scale * qdot * DT)
                after = certificate_world_positions(data, robot)
                set_configuration(model, data, q0)
                certificate = occupancy.certify_swept_spheres(
                    before, after, radii
                )
                if certificate.safe:
                    break
                scale *= 0.5
            if certificate is None or not certificate.safe:
                blockers = []
                for index, sphere in enumerate(robot):
                    item = occupancy.certify_swept_spheres(
                        before[index : index + 1],
                        after[index : index + 1],
                        radii[index : index + 1],
                    )
                    if not item.safe:
                        blockers.append(
                            {
                                "sphere_index": index,
                                "body_name": sphere.body_name,
                                "unknown_voxels": item.unknown_voxels,
                                "occupied_voxels": item.occupied_voxels,
                                "first_nonfree_voxel": (
                                    _first_nonfree_swept_voxel(
                                        occupancy,
                                        before[index],
                                        after[index],
                                        radii[index],
                                    )
                                ),
                            }
                        )
                first_blocker = {
                    "cycle": cycle,
                    "unknown_voxels": certificate.unknown_voxels,
                    "occupied_voxels": certificate.occupied_voxels,
                    "blocking_spheres": blockers,
                }
                break
            minimum_scale = min(minimum_scale, scale)
            set_configuration(model, data, q0 + scale * qdot * DT)
            accepted += 1
        return {
            "requested_cycles": cycles,
            "accepted_cycles": accepted,
            "all_cycles_safe": accepted == cycles,
            "minimum_accepted_scale": minimum_scale,
            "first_blocker": first_blocker,
            "final_error_m": float(
                np.linalg.norm(
                    np.asarray(scene.waypoints[-1], dtype=float)
                    - data.site_xpos[model.site("attachment_site").id]
                )
            ),
            "last_qp_status": metrics.status,
        }
    finally:
        occupancy.close()


def main() -> None:
    gate = json.loads(STATIC_GATE.read_text(encoding="utf-8"))
    scene = protocol_drawer_v4_scene()
    configurations = [
        ("q0", np.asarray(scene.q0, dtype=float)),
        ("mandatory_crossing", np.asarray(gate["ellipsoid_crossing"]["q"])),
    ]
    configurations.extend(
        (f"goal_{index}", np.asarray(row["q"], dtype=float))
        for index, row in enumerate(gate["goal_ik"])
    )
    reports = []
    for housing_local_y in CANDIDATE_HOUSING_LOCAL_Y:
        for local_z in CANDIDATE_LOCAL_Z:
            model = build_model(scene)
            data = mujoco.MjData(model)
            camera_id = model.camera("ur5_depth_wrist").id
            housing_id = model.geom("protocol_d405_camera_housing").id
            camera_local_y = housing_local_y + CAMERA_TO_HOUSING_FRONT
            model.cam_pos[camera_id] = np.array([0.0, camera_local_y, local_z])
            model.geom_pos[housing_id] = np.array(
                [0.0, housing_local_y, local_z]
            )
            poses = []
            strict_initial_step = None
            with UR5MountedDepthCamera(
                model,
                camera_names=("ur5_depth_wrist", "ur5_depth_shoulder"),
                width=160,
                height=120,
                pixel_stride=2,
                optical_depth_error_bound=0.003,
                occluding_self_filter=True,
                native_raycast=True,
            ) as camera:
                for label, q in configurations:
                    set_configuration(model, data, q)
                    observations = camera.capture(data)
                    counts = {}
                    for observation in observations:
                        mask = environment_endpoint_mask(observation)
                        counts[observation.camera_name] = {
                            "environment_first_hits": int(np.count_nonzero(mask)),
                            "self_first_hits": int(
                                len(mask) - np.count_nonzero(mask)
                            ),
                        }
                    poses.append(
                        {
                            "configuration": label,
                            "penetrating_contacts": int(
                                sum(
                                    1
                                    for contact in data.contact
                                    if contact.dist < 0.0
                                )
                            ),
                            "cameras": counts,
                        }
                    )
                    if label == "q0":
                        strict_initial_step = _strict_initial_step(
                            model, data, scene, observations
                        )
                strict_rollout = _strict_open_space_rollout(
                    model, data, scene, camera
                )
            reports.append(
                {
                    "wrist_camera_local_y_m": camera_local_y,
                    "wrist_housing_local_y_m": housing_local_y,
                    "wrist_local_z_m": local_z,
                    "strict_initial_step": strict_initial_step,
                    "strict_open_space_rollout": strict_rollout,
                    "poses": poses,
                }
            )
    print(json.dumps({"candidates": reports}, indent=2))


if __name__ == "__main__":
    main()
