"""Wrist D405 compensation screen with forearm fixed on the cabinet blind spot.

Candidate wrist axes are bounded slerps from the frozen wrist view toward a
small grid of drawer-surface targets.  Ranking uses only visibility before
risk and camera-housing collision on the frozen source trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

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


FOREARM_BLINDSPOT_WXYZ = np.array(
    [0.5311666153160751, 0.4004262832359864, 0.5152439122715451, -0.54041144448891]
)


def _xyzw(wxyz: np.ndarray) -> np.ndarray:
    return np.r_[wxyz[1:], wxyz[0]]


def _wxyz(xyzw: np.ndarray) -> np.ndarray:
    return np.r_[xyzw[3], xyzw[:3]]


def _optical_axis_local(quaternion: np.ndarray) -> np.ndarray:
    rotation_flat = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(rotation_flat, quaternion)
    return -rotation_flat.reshape(3, 3)[:, 2]


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
    wrist_housing_id = model.geom("formal_d405_wrist_housing").id
    forearm_housing_id = model.geom("formal_d405_forearm_housing").id
    wrist_body_id = int(model.cam_bodyid[wrist_id])
    wrist_body_rotation = data.xmat[wrist_body_id].reshape(3, 3).copy()
    wrist_body_position = data.xpos[wrist_body_id].copy()
    wrist_position = model.cam_pos[wrist_id].copy()
    forearm_position = model.cam_pos[forearm_id].copy()
    wrist_start = model.cam_quat[wrist_id].copy()
    model.cam_fovy[wrist_id] = D405_VERTICAL_FOV_DEG
    model.cam_fovy[forearm_id] = D405_VERTICAL_FOV_DEG
    option = mujoco.MjvOption()
    option.geomgroup[1] = 1
    geomgroup = np.asarray(option.geomgroup, dtype=np.uint8)

    model.cam_quat[forearm_id] = FOREARM_BLINDSPOT_WXYZ
    model.geom_quat[forearm_housing_id] = FOREARM_BLINDSPOT_WXYZ
    model.geom_pos[forearm_housing_id] = (
        forearm_position
        - LENS_PROTRUSION_FROM_HOUSING_CENTER
        * _optical_axis_local(FOREARM_BLINDSPOT_WXYZ)
    )

    targets = [
        np.array([x, y, z], dtype=float)
        for x in (0.35, 0.45)
        for y in (0.20, 0.32, 0.44)
        for z in (0.48, 0.57, 0.66)
    ]
    rows: list[dict] = []
    for target in targets:
        endpoint, _ = _camera_quaternion_for_target(
            wrist_body_rotation,
            wrist_body_position,
            wrist_position,
            target,
        )
        slerp = Slerp(
            [0.0, 1.0],
            Rotation.from_quat(
                np.vstack((_xyzw(wrist_start), _xyzw(endpoint)))
            ),
        )
        for fraction in np.linspace(0.0, 0.50, 11):
            quaternion = _wxyz(slerp([fraction]).as_quat()[0])
            model.cam_quat[wrist_id] = quaternion
            model.geom_quat[wrist_housing_id] = quaternion
            model.geom_pos[wrist_housing_id] = (
                wrist_position
                - LENS_PROTRUSION_FROM_HOUSING_CENTER
                * _optical_axis_local(quaternion)
            )
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
                    "wrist_fraction": float(fraction),
                    "wrist_target_m": target.tolist(),
                    "wrist_quaternion_wxyz": quaternion.tolist(),
                    "forearm_quaternion_wxyz": (
                        FOREARM_BLINDSPOT_WXYZ.tolist()
                    ),
                    "wrist_housing_center_m": model.geom_pos[
                        wrist_housing_id
                    ].tolist(),
                    "forearm_housing_center_m": model.geom_pos[
                        forearm_housing_id
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
            row["wrist_fraction"],
        )
    )
    report = {
        "audit_role": "v4_3_wrist_drawer_compensation_visibility_collision_screen",
        "controller_success_loaded": False,
        "controller_success_used_for_ranking": False,
        "truth_feedback_to_control": False,
        "camera_optical_centers_frozen": True,
        "camera_fov_frozen": True,
        "forearm_blindspot_axis_frozen": True,
        "source_run": str(args.run_dir.resolve()),
        "candidate_count": len(rows),
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
