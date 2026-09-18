"""Visibility-only roll audit for the two physical D405 mounts.

The wrist elevation and forearm optical target come from earlier frozen
collision/visibility screens.  This audit changes only the forearm camera's
roll about its own optical axis, so neither optical center nor center ray is
moved.  Completed sphere/ellipsoid trajectories are read-only inputs and QP
success is never used for ranking.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from audit_formal_two_trajectory_camera_mount import (
    D405_VERTICAL_FOV_DEG,
    _trajectory_context,
    _visibility_row,
    _wrist_quaternion,
)
from audit_secondary_shoulder_mount_grid import (
    LENS_PROTRUSION_FROM_HOUSING_CENTER,
    _camera_quaternion_for_target,
    _has_penetration,
)
from model import build_model, set_configuration
from protocol_drawer_scene import (
    DRAWER_Y,
    DRAWER_Z,
    formal_drawer_two_camera_scene,
)


def _rolled_quaternion(quaternion: np.ndarray, degrees: float) -> np.ndarray:
    flat = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(flat, np.asarray(quaternion, dtype=float))
    rotation = flat.reshape(3, 3).copy()
    angle = math.radians(float(degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    x_axis = rotation[:, 0].copy()
    y_axis = rotation[:, 1].copy()
    rotation[:, 0] = cosine * x_axis + sine * y_axis
    rotation[:, 1] = -sine * x_axis + cosine * y_axis
    result = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(result, rotation.reshape(-1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ellipsoid_run_dir", type=Path)
    parser.add_argument("sphere_run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    scene = formal_drawer_two_camera_scene()
    contexts = {
        "ellipsoid": _trajectory_context(args.ellipsoid_run_dir, scene),
        "sphere": _trajectory_context(args.sphere_run_dir, scene),
    }
    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(scene.q0, dtype=float))
    wrist_id = model.camera("ur5_depth_wrist").id
    forearm_id = model.camera("ur5_depth_forearm").id
    wrist_housing_id = model.geom("formal_d405_wrist_housing").id
    forearm_housing_id = model.geom("formal_d405_forearm_housing").id
    forearm_body_id = model.body("forearm_link").id
    body_rotation = data.xmat[forearm_body_id].reshape(3, 3).copy()
    body_position = data.xpos[forearm_body_id].copy()
    wrist_position = model.cam_pos[wrist_id].copy()
    local_position = np.array([0.0, 0.075, 0.31])
    target = np.array([0.35, DRAWER_Y - 0.18, DRAWER_Z - 0.06])
    base_forearm_quat, optical_local = _camera_quaternion_for_target(
        body_rotation, body_position, local_position, target
    )
    option = mujoco.MjvOption()
    option.geomgroup[1] = 1
    geomgroup = np.asarray(option.geomgroup, dtype=np.uint8)
    model.cam_fovy[wrist_id] = D405_VERTICAL_FOV_DEG
    model.cam_fovy[forearm_id] = D405_VERTICAL_FOV_DEG

    rows = []
    for wrist_yaw in (20.0, 25.0, 30.0, 35.0, 40.0):
        wrist_quat, wrist_optical = _wrist_quaternion(wrist_yaw, 15.0)
        model.cam_quat[wrist_id] = wrist_quat
        model.geom_quat[wrist_housing_id] = wrist_quat
        model.geom_pos[wrist_housing_id] = (
            wrist_position
            - LENS_PROTRUSION_FROM_HOUSING_CENTER * wrist_optical
        )
        for roll in (-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0):
            forearm_quat = _rolled_quaternion(base_forearm_quat, roll)
            model.cam_pos[forearm_id] = local_position
            model.cam_quat[forearm_id] = forearm_quat
            model.geom_pos[forearm_housing_id] = (
                local_position
                - LENS_PROTRUSION_FROM_HOUSING_CENTER * optical_local
            )
            model.geom_quat[forearm_housing_id] = forearm_quat
            collision = {}
            collision_free = True
            for name, context in contexts.items():
                hit, cycle, distance = _has_penetration(
                    model, data, context["trajectory"]
                )
                collision[name] = {
                    "collision_free": not hit,
                    "first_contact_cycle": cycle,
                    "minimum_contact_distance_m": distance,
                }
                collision_free &= not hit
            visibility = {}
            for name, context in contexts.items():
                visibility[name] = (
                    _visibility_row(
                        model,
                        data,
                        context,
                        (wrist_id, forearm_id),
                        geomgroup,
                    )
                    if collision_free
                    else {
                        "risk_samples": int(len(context["risk_indices"])),
                        "visible_before_deadline": 0,
                        "missed_before_deadline": int(
                            len(context["risk_indices"])
                        ),
                    }
                )
            missed = [
                visibility[name]["missed_before_deadline"]
                for name in ("ellipsoid", "sphere")
            ]
            rows.append(
                {
                    "wrist_yaw_deg": wrist_yaw,
                    "wrist_pitch_deg": 15.0,
                    "wrist_quaternion_wxyz": wrist_quat.tolist(),
                    "wrist_housing_center_m": model.geom_pos[
                        wrist_housing_id
                    ].tolist(),
                    "forearm_target_m": target.tolist(),
                    "forearm_roll_deg": roll,
                    "forearm_quaternion_wxyz": forearm_quat.tolist(),
                    "forearm_housing_center_m": model.geom_pos[
                        forearm_housing_id
                    ].tolist(),
                    "collision_free_on_both_trajectories": collision_free,
                    "collision": collision,
                    "visibility": visibility,
                    "maximum_missed_before_deadline": max(missed),
                    "total_missed_before_deadline": sum(missed),
                }
            )

    rows.sort(
        key=lambda row: (
            not row["collision_free_on_both_trajectories"],
            row["maximum_missed_before_deadline"],
            row["total_missed_before_deadline"],
        )
    )
    report = {
        "audit_role": "formal_two_trajectory_d405_forearm_roll_screen",
        "controller_success_used_for_ranking": False,
        "truth_feedback_to_control": False,
        "vertical_fov_deg": D405_VERTICAL_FOV_DEG,
        "candidate_count": len(rows),
        "source_runs": {
            "ellipsoid": str(args.ellipsoid_run_dir.resolve()),
            "sphere": str(args.sphere_run_dir.resolve()),
        },
        "ranking": rows,
        "elapsed_s": time.perf_counter() - started,
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
