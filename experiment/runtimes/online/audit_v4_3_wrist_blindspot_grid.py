"""Visibility/collision-only wrist D405 aim screen for the v4.3 blind spot.

The successful source trajectory is used only to define when truth-surface
witnesses first become risky.  Controller success is never loaded or used for
ranking.  The forearm camera, D405 FOV, wrist optical-center position, robot,
scene and task remain frozen.  Each candidate changes only the wrist optical
axis and the matching physical housing orientation.
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
    housing_id = model.geom("formal_d405_wrist_housing").id
    body_id = int(model.cam_bodyid[wrist_id])
    body_rotation = data.xmat[body_id].reshape(3, 3).copy()
    body_position = data.xpos[body_id].copy()
    local_position = model.cam_pos[wrist_id].copy()
    model.cam_fovy[wrist_id] = D405_VERTICAL_FOV_DEG
    model.cam_fovy[forearm_id] = D405_VERTICAL_FOV_DEG
    option = mujoco.MjvOption()
    option.geomgroup[1] = 1
    geomgroup = np.asarray(option.geomgroup, dtype=np.uint8)

    rows: list[dict] = []
    for target_x in (0.35, 0.45):
        for target_y in (0.02, 0.08, 0.14, 0.20, 0.26):
            for target_z in (0.34, 0.42, 0.50, 0.58, 0.66):
                target = np.array(
                    [target_x, target_y, target_z], dtype=float
                )
                quaternion, optical_local = _camera_quaternion_for_target(
                    body_rotation,
                    body_position,
                    local_position,
                    target,
                )
                model.cam_quat[wrist_id] = quaternion
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
                        "wrist_local_position_m": local_position.tolist(),
                        "wrist_target_m": target.tolist(),
                        "wrist_quaternion_wxyz": quaternion.tolist(),
                        "wrist_housing_center_m": model.geom_pos[
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
        "audit_role": "v4_3_wrist_blindspot_visibility_collision_screen",
        "controller_success_loaded": False,
        "controller_success_used_for_ranking": False,
        "truth_feedback_to_control": False,
        "forearm_camera_frozen": True,
        "wrist_optical_center_frozen": True,
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
