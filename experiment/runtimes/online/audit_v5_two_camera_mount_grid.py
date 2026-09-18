"""Fine visibility/collision-only audit for the two physical v5 D405 mounts.

The search consumes both preserved trajectories, but never reads success,
error, QP state, target velocity, or proxy representation when ranking.  It
minimizes observed-before-risk misses, then freezes the lexicographically first
collision-free mount.  This is an instrumentation design audit, not a control
or path-planning module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from audit_formal_forearm_roll import _rolled_quaternion
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    option = mujoco.MjvOption()
    option.geomgroup[1] = 1
    geomgroup = np.asarray(option.geomgroup, dtype=np.uint8)
    model.cam_fovy[wrist_id] = D405_VERTICAL_FOV_DEG
    model.cam_fovy[forearm_id] = D405_VERTICAL_FOV_DEG

    rows = []
    # Predeclared local refinement around the best coarse visibility-only
    # mount: +/-20 mm target translation at 5 mm spacing and +/-5 degree roll.
    wrist_yaw = 20.0
    wrist_pitch = 15.0
    wrist_quat, wrist_optical = _wrist_quaternion(wrist_yaw, wrist_pitch)
    model.cam_quat[wrist_id] = wrist_quat
    model.geom_quat[wrist_housing_id] = wrist_quat
    model.geom_pos[wrist_housing_id] = (
        wrist_position
        - LENS_PROTRUSION_FROM_HOUSING_CENTER * wrist_optical
    )
    for target_y_offset in np.arange(-0.20, -0.14 + 1.0e-12, 0.005):
        for target_z_offset in np.arange(-0.08, -0.04 + 1.0e-12, 0.005):
            target = np.array(
                [
                    0.35,
                    DRAWER_Y + float(target_y_offset),
                    DRAWER_Z + float(target_z_offset),
                ]
            )
            base_quat, optical_local = _camera_quaternion_for_target(
                body_rotation, body_position, local_position, target
            )
            for roll in (-20.0, -17.5, -15.0, -12.5, -10.0):
                forearm_quat = _rolled_quaternion(base_quat, roll)
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
                        "wrist_pitch_deg": wrist_pitch,
                        "wrist_quaternion_wxyz": wrist_quat.tolist(),
                        "wrist_housing_center_m": model.geom_pos[
                            wrist_housing_id
                        ].tolist(),
                        "forearm_local_position_m": local_position.tolist(),
                        "forearm_target_m": target.tolist(),
                        "forearm_target_y_offset_m": float(target_y_offset),
                        "forearm_target_z_offset_m": float(target_z_offset),
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
            row["forearm_target_y_offset_m"],
            row["forearm_target_z_offset_m"],
            row["forearm_roll_deg"],
        )
    )
    report = {
        "audit_role": "protocol_v5_two_physical_d405_visibility_screen",
        "formal_evidence": False,
        "controller_success_used_for_ranking": False,
        "truth_feedback_to_control": False,
        "candidate_grid_frozen_before_execution": True,
        "vertical_fov_deg": D405_VERTICAL_FOV_DEG,
        "candidate_count": len(rows),
        "source_runs": {
            "ellipsoid": str(args.ellipsoid_run_dir.resolve()),
            "sphere": str(args.sphere_run_dir.resolve()),
        },
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
                "candidate_count": len(rows),
                "best": rows[0],
                "zero_miss_candidates": int(
                    sum(
                        row["collision_free_on_both_trajectories"]
                        and row["maximum_missed_before_deadline"] == 0
                        for row in rows
                    )
                ),
                "elapsed_s": report["elapsed_s"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
