"""Visibility-only forearm D405 aim screen for the v4.3 cabinet-left blind spot.

The wrist camera, robot trajectory, scene, camera FOV and physical forearm
mount are frozen.  Candidates change only the forearm optical/housing
orientation and are ranked by collision-free mounting plus visibility before
risk.  Controller success is neither loaded nor used for ranking.
"""

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
)
from audit_secondary_shoulder_mount_grid import (
    LENS_PROTRUSION_FROM_HOUSING_CENTER,
    _camera_quaternion_for_target,
    _has_penetration,
)
from model import build_model, set_configuration
from protocol_drawer_scene import formal_drawer_camera_quarter_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()

    scene = formal_drawer_camera_quarter_scene()
    context = _trajectory_context(args.run_dir, scene)
    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(scene.q0, dtype=float))
    wrist_id = model.camera("ur5_depth_wrist").id
    forearm_id = model.camera("ur5_depth_forearm").id
    housing_id = model.geom("formal_d405_forearm_housing").id
    body_id = model.body("forearm_link").id
    body_rotation = data.xmat[body_id].reshape(3, 3).copy()
    body_position = data.xpos[body_id].copy()
    model.cam_fovy[wrist_id] = D405_VERTICAL_FOV_DEG
    model.cam_fovy[forearm_id] = D405_VERTICAL_FOV_DEG
    option = mujoco.MjvOption()
    option.geomgroup[1] = 1
    geomgroup = np.asarray(option.geomgroup, dtype=np.uint8)

    rows: list[dict] = []
    for local_x in (-0.04, 0.0, 0.04):
        for local_z in (0.28, 0.31, 0.34):
            local_position = np.array([local_x, 0.075, local_z], dtype=float)
            for target_y in (0.02, 0.08, 0.14, 0.20, 0.26):
                for target_z in (0.34, 0.40, 0.46, 0.52, 0.58):
                    target = np.array([0.35, target_y, target_z], dtype=float)
                    quaternion, optical_local = _camera_quaternion_for_target(
                        body_rotation,
                        body_position,
                        local_position,
                        target,
                    )
                    model.cam_pos[forearm_id] = local_position
                    model.cam_quat[forearm_id] = quaternion
                    model.geom_pos[housing_id] = (
                        local_position
                        - LENS_PROTRUSION_FROM_HOUSING_CENTER * optical_local
                    )
                    model.geom_quat[housing_id] = quaternion
                    collides, first_contact, minimum_contact = _has_penetration(
                        model, data, context["trajectory"]
                    )
                    visibility = (
                        _visibility_row(
                            model,
                            data,
                            context,
                            (wrist_id, forearm_id),
                            geomgroup,
                        )
                        if not collides
                        else {
                            "risk_samples": int(len(context["risk_indices"])),
                            "visible_before_deadline": 0,
                            "missed_before_deadline": int(
                                len(context["risk_indices"])
                            ),
                            "missed_by_box": {},
                            "missed_point_bounds_m": None,
                        }
                    )
                    rows.append(
                        {
                            "forearm_local_position_m": local_position.tolist(),
                            "forearm_target_m": target.tolist(),
                            "forearm_quaternion_wxyz": quaternion.tolist(),
                            "forearm_housing_center_m": model.geom_pos[
                                housing_id
                            ].tolist(),
                            "collision_free_on_source_trajectory": not collides,
                            "first_contact_cycle": first_contact,
                            "minimum_contact_distance_m": minimum_contact,
                            **visibility,
                        }
                    )
    rows.sort(
        key=lambda row: (
            not row["collision_free_on_source_trajectory"],
            row["missed_before_deadline"],
            sum(row.get("missed_by_box", {}).values()),
        )
    )
    report = {
        "audit_role": "v4_3_forearm_blindspot_visibility_only_screen",
        "controller_success_loaded": False,
        "controller_success_used_for_ranking": False,
        "truth_feedback_to_control": False,
        "wrist_camera_frozen": True,
        "source_run": str(args.run_dir.resolve()),
        "candidate_count": len(rows),
        "ranking": rows,
        "elapsed_s": time.perf_counter() - started,
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "ranking"
            }
            | {"best": rows[0]},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
