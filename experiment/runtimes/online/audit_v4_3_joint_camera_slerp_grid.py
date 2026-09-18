"""Joint two-D405 visibility/collision screen with frozen optical centers.

The grid interpolates between predeclared, collision-screened optical axes.
It ranks only observed-before-risk visibility and physical housing collision
on a frozen source trajectory; controller success is never loaded.
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
    _has_penetration,
)
from model import build_model, set_configuration
from protocol_drawer_scene import formal_drawer_camera_quarter_scene


FOREARM_BLINDSPOT_WXYZ = np.array(
    [0.5311666153160751, 0.4004262832359864, 0.5152439122715451, -0.54041144448891]
)
WRIST_BLINDSPOT_WXYZ = np.array(
    [0.46853029060547596, -0.3559533326369886, 0.600134872365962, -0.5418622765427206]
)


def _xyzw(wxyz: np.ndarray) -> np.ndarray:
    return np.r_[wxyz[1:], wxyz[0]]


def _wxyz(xyzw: np.ndarray) -> np.ndarray:
    return np.r_[xyzw[3], xyzw[:3]]


def _optical_axis_local(quaternion: np.ndarray) -> np.ndarray:
    rotation_flat = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(rotation_flat, quaternion)
    return -rotation_flat.reshape(3, 3)[:, 2]


def _slerp(start_wxyz: np.ndarray, end_wxyz: np.ndarray) -> Slerp:
    return Slerp(
        [0.0, 1.0],
        Rotation.from_quat(
            np.vstack((_xyzw(start_wxyz), _xyzw(end_wxyz)))
        ),
    )


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
    wrist_position = model.cam_pos[wrist_id].copy()
    forearm_position = model.cam_pos[forearm_id].copy()
    model.cam_fovy[wrist_id] = D405_VERTICAL_FOV_DEG
    model.cam_fovy[forearm_id] = D405_VERTICAL_FOV_DEG
    option = mujoco.MjvOption()
    option.geomgroup[1] = 1
    geomgroup = np.asarray(option.geomgroup, dtype=np.uint8)

    wrist_start = model.cam_quat[wrist_id].copy()
    forearm_start = model.cam_quat[forearm_id].copy()
    wrist_slerp = _slerp(wrist_start, WRIST_BLINDSPOT_WXYZ)
    forearm_slerp = _slerp(forearm_start, FOREARM_BLINDSPOT_WXYZ)

    rows: list[dict] = []
    for wrist_fraction in np.linspace(0.0, 0.30, 11):
        wrist_quaternion = _wxyz(
            wrist_slerp([wrist_fraction]).as_quat()[0]
        )
        model.cam_quat[wrist_id] = wrist_quaternion
        model.geom_quat[wrist_housing_id] = wrist_quaternion
        model.geom_pos[wrist_housing_id] = (
            wrist_position
            - LENS_PROTRUSION_FROM_HOUSING_CENTER
            * _optical_axis_local(wrist_quaternion)
        )
        for forearm_fraction in np.linspace(0.0, 1.0, 11):
            forearm_quaternion = _wxyz(
                forearm_slerp([forearm_fraction]).as_quat()[0]
            )
            model.cam_quat[forearm_id] = forearm_quaternion
            model.geom_quat[forearm_housing_id] = forearm_quaternion
            model.geom_pos[forearm_housing_id] = (
                forearm_position
                - LENS_PROTRUSION_FROM_HOUSING_CENTER
                * _optical_axis_local(forearm_quaternion)
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
                    "wrist_fraction": float(wrist_fraction),
                    "forearm_fraction": float(forearm_fraction),
                    "wrist_quaternion_wxyz": wrist_quaternion.tolist(),
                    "forearm_quaternion_wxyz": forearm_quaternion.tolist(),
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
            row["wrist_fraction"] + row["forearm_fraction"],
        )
    )
    report = {
        "audit_role": "v4_3_joint_two_d405_slerp_visibility_collision_screen",
        "controller_success_loaded": False,
        "controller_success_used_for_ranking": False,
        "truth_feedback_to_control": False,
        "camera_optical_centers_frozen": True,
        "camera_fov_frozen": True,
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
