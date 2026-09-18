"""Visibility/collision audit for translating the forearm D405 mount only."""

from __future__ import annotations

import argparse
import json
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
    _has_penetration,
)
from model import build_model, set_configuration
from protocol_drawer_scene import formal_drawer_two_camera_scene


FOREARM_QUATERNION = np.array(
    [
        0.5106926718694792,
        0.4259945316165838,
        0.5010282490438904,
        -0.5537981108595377,
    ]
)


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
    wrist_position = model.cam_pos[wrist_id].copy()
    wrist_quat, wrist_optical = _wrist_quaternion(20.0, 15.0)
    model.cam_quat[wrist_id] = wrist_quat
    model.geom_quat[wrist_housing_id] = wrist_quat
    model.geom_pos[wrist_housing_id] = (
        wrist_position - LENS_PROTRUSION_FROM_HOUSING_CENTER * wrist_optical
    )
    rotation_flat = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(rotation_flat, FOREARM_QUATERNION)
    forearm_optical = -rotation_flat.reshape(3, 3)[:, 2]
    model.cam_quat[forearm_id] = FOREARM_QUATERNION
    model.geom_quat[forearm_housing_id] = FOREARM_QUATERNION
    model.cam_fovy[wrist_id] = D405_VERTICAL_FOV_DEG
    model.cam_fovy[forearm_id] = D405_VERTICAL_FOV_DEG
    option = mujoco.MjvOption()
    option.geomgroup[1] = 1
    geomgroup = np.asarray(option.geomgroup, dtype=np.uint8)

    rows = []
    for local_x in (-0.06, -0.03, 0.0, 0.03, 0.06):
        for local_y in (0.055, 0.075, 0.095):
            for local_z in (0.25, 0.28, 0.31, 0.34, 0.37):
                position = np.array([local_x, local_y, local_z])
                model.cam_pos[forearm_id] = position
                model.geom_pos[forearm_housing_id] = (
                    position
                    - LENS_PROTRUSION_FROM_HOUSING_CENTER * forearm_optical
                )
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
                        "forearm_local_position_m": position.tolist(),
                        "forearm_housing_center_m": model.geom_pos[
                            forearm_housing_id
                        ].tolist(),
                        "forearm_quaternion_wxyz": FOREARM_QUATERNION.tolist(),
                        "wrist_yaw_deg": 20.0,
                        "wrist_pitch_deg": 15.0,
                        "wrist_quaternion_wxyz": wrist_quat.tolist(),
                        "wrist_housing_center_m": model.geom_pos[
                            wrist_housing_id
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
        "audit_role": "formal_two_trajectory_d405_forearm_translation_screen",
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
