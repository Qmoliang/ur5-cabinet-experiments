"""Visibility-only two-camera redesign audit for uniform-radius trajectories."""

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
    _camera_quaternion_for_target,
    _has_penetration,
)
from model import build_model, set_configuration
from protocol_drawer_scene import DRAWER_Y, DRAWER_Z, formal_drawer_two_camera_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ellipsoid_run_dir", type=Path)
    parser.add_argument("sphere_run_dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
    model.cam_fovy[wrist_id] = D405_VERTICAL_FOV_DEG
    model.cam_fovy[forearm_id] = D405_VERTICAL_FOV_DEG
    option = mujoco.MjvOption()
    option.geomgroup[1] = 1
    geomgroup = np.asarray(option.geomgroup, dtype=np.uint8)

    rows: list[dict] = []
    for wrist_yaw_deg in (15.0, 20.0, 25.0, 30.0, 35.0):
        for wrist_pitch_deg in (0.0, 10.0, 15.0, 20.0, 30.0):
            wrist_quat, wrist_optical = _wrist_quaternion(
                wrist_yaw_deg, wrist_pitch_deg
            )
            model.cam_quat[wrist_id] = wrist_quat
            model.geom_quat[wrist_housing_id] = wrist_quat
            model.geom_pos[wrist_housing_id] = (
                wrist_position
                - LENS_PROTRUSION_FROM_HOUSING_CENTER * wrist_optical
            )
            for mount_z in (0.28, 0.30, 0.31, 0.32):
                local_position = np.array([0.0, 0.075, mount_z])
                for target_y_offset in (-0.20, -0.18, -0.16, -0.14):
                    for target_z_offset in (-0.08, -0.06, -0.04, -0.02):
                        target = np.array(
                            [
                                0.35,
                                DRAWER_Y + target_y_offset,
                                DRAWER_Z + target_z_offset,
                            ]
                        )
                        forearm_quat, optical_local = _camera_quaternion_for_target(
                            body_rotation,
                            body_position,
                            local_position,
                            target,
                        )
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
                                    "risk_samples": int(
                                        len(context["risk_indices"])
                                    ),
                                    "visible_before_deadline": 0,
                                    "missed_before_deadline": int(
                                        len(context["risk_indices"])
                                    ),
                                }
                            )
                        rows.append(
                            {
                                "wrist_yaw_deg": wrist_yaw_deg,
                                "wrist_pitch_deg": wrist_pitch_deg,
                                "wrist_quaternion_wxyz": wrist_quat.tolist(),
                                "wrist_housing_center_m": model.geom_pos[
                                    wrist_housing_id
                                ].tolist(),
                                "forearm_local_position_m": local_position.tolist(),
                                "forearm_target_m": target.tolist(),
                                "forearm_target_y_offset_m": target_y_offset,
                                "forearm_target_z_offset_m": target_z_offset,
                                "forearm_quaternion_wxyz": forearm_quat.tolist(),
                                "forearm_housing_center_m": model.geom_pos[
                                    forearm_housing_id
                                ].tolist(),
                                "collision_free_on_both_trajectories": collision_free,
                                "collision": collision,
                                "visibility": visibility,
                                "maximum_missed_before_deadline": max(
                                    visibility[name]["missed_before_deadline"]
                                    for name in visibility
                                ),
                                "total_missed_before_deadline": sum(
                                    visibility[name]["missed_before_deadline"]
                                    for name in visibility
                                ),
                            }
                        )
    rows.sort(
        key=lambda row: (
            not row["collision_free_on_both_trajectories"],
            row["maximum_missed_before_deadline"],
            row["total_missed_before_deadline"],
            abs(row["wrist_yaw_deg"] - 20.0)
            + abs(row["wrist_pitch_deg"] - 15.0),
        )
    )
    report = {
        "audit_role": "uniform_radius_two_camera_visibility_only_redesign",
        "controller_success_used_for_ranking": False,
        "truth_feedback_to_control": False,
        "source_runs": {
            key: str(path.resolve())
            for key, path in {
                "ellipsoid": args.ellipsoid_run_dir,
                "sphere": args.sphere_run_dir,
            }.items()
        },
        "candidate_count": len(rows),
        "zero_miss_candidate_count": sum(
            row["collision_free_on_both_trajectories"]
            and row["maximum_missed_before_deadline"] == 0
            for row in rows
        ),
        "ranking": rows,
        "elapsed_s": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_count": len(rows),
                "zero_miss_candidate_count": report["zero_miss_candidate_count"],
                "top": rows[:5],
                "elapsed_s": report["elapsed_s"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
