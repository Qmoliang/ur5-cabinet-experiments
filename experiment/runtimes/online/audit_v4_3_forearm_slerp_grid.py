"""Visibility/collision/initial-load screen between two forearm D405 aims."""

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
from depth_camera_perception import UR5MountedDepthCamera, environment_endpoint_mask
from model import build_model, set_configuration
from protocol_drawer_scene import formal_drawer_camera_quarter_scene


QUARTER_WXYZ = np.array(
    [0.5059250092158937, 0.4319329943648225, 0.514929627115348, -0.5406674139873654]
)
BLINDSPOT_WXYZ = np.array(
    [0.5311666153160751, 0.4004262832359864, 0.5152439122715451, -0.54041144448891]
)


def _xyzw(wxyz: np.ndarray) -> np.ndarray:
    return np.r_[wxyz[1:], wxyz[0]]


def _wxyz(xyzw: np.ndarray) -> np.ndarray:
    return np.r_[xyzw[3], xyzw[:3]]


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
    camera_position = model.cam_pos[forearm_id].copy()
    model.cam_fovy[wrist_id] = D405_VERTICAL_FOV_DEG
    model.cam_fovy[forearm_id] = D405_VERTICAL_FOV_DEG
    option = mujoco.MjvOption()
    option.geomgroup[1] = 1
    geomgroup = np.asarray(option.geomgroup, dtype=np.uint8)

    key_rots = Rotation.from_quat(
        np.vstack((_xyzw(QUARTER_WXYZ), _xyzw(BLINDSPOT_WXYZ)))
    )
    slerp = Slerp([0.0, 1.0], key_rots)
    rows: list[dict] = []
    for fraction in np.linspace(0.0, 1.0, 21):
        quaternion = _wxyz(slerp([fraction]).as_quat()[0])
        rotation_flat = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(rotation_flat, quaternion)
        optical_local = -rotation_flat.reshape(3, 3)[:, 2]
        model.cam_quat[forearm_id] = quaternion
        model.geom_quat[housing_id] = quaternion
        model.geom_pos[housing_id] = (
            camera_position
            - LENS_PROTRUSION_FROM_HOUSING_CENTER * optical_local
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
                "missed_before_deadline": int(len(context["risk_indices"])),
                "missed_by_box": {},
                "missed_point_bounds_m": None,
            }
        )
        set_configuration(model, data, np.asarray(scene.q0, dtype=float))
        observations = UR5MountedDepthCamera(
            model,
            camera_names=("ur5_depth_wrist", "ur5_depth_forearm"),
            width=320,
            height=180,
        ).capture(data)
        environment_points = sum(
            int(np.count_nonzero(environment_endpoint_mask(item)))
            for item in observations
        )
        workspace_batches = []
        for item in observations:
            keep = environment_endpoint_mask(item)
            points = np.asarray(item.points, dtype=float)[keep]
            keep_workspace = np.all(points >= [-0.20, -0.20, 0.10], axis=1)
            keep_workspace &= np.all(points <= [1.00, 1.00, 1.20], axis=1)
            workspace_batches.append(points[keep_workspace])
        workspace = (
            np.vstack(workspace_batches)
            if workspace_batches
            else np.empty((0, 3), dtype=float)
        )
        center_voxels = (
            0
            if not len(workspace)
            else len(
                np.unique(
                    np.floor((workspace - [-1.2, -1.2, 0.0]) / 0.0075).astype(
                        np.int64
                    ),
                    axis=0,
                )
            )
        )
        rows.append(
            {
                "fraction": float(fraction),
                "forearm_quaternion_wxyz": quaternion.tolist(),
                "forearm_housing_center_m": model.geom_pos[housing_id].tolist(),
                "collision_free_on_source_trajectory": not collides,
                "first_contact_cycle": first_contact,
                "minimum_contact_distance_m": minimum_contact,
                "initial_environment_depth_points": environment_points,
                "initial_center_voxels_estimate": int(center_voxels),
                **visibility,
            }
        )
    rows.sort(
        key=lambda row: (
            not row["collision_free_on_source_trajectory"],
            row["missed_before_deadline"],
            row["initial_center_voxels_estimate"],
        )
    )
    report = {
        "audit_role": "v4_3_forearm_slerp_visibility_collision_load_screen",
        "controller_success_loaded": False,
        "controller_success_used_for_ranking": False,
        "truth_feedback_to_control": False,
        "source_run": str(args.run_dir.resolve()),
        "candidate_count": len(rows),
        "ranking": rows,
        "elapsed_s": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({**report, "ranking": rows[:5]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
