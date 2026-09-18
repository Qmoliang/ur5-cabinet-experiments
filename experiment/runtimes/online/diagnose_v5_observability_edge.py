"""Diagnose the invariant drawer-edge visibility witness with exact MuJoCo rays."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_formal_forearm_roll import _rolled_quaternion
from audit_formal_two_trajectory_camera_mount import (
    _trajectory_context,
    _wrist_quaternion,
)
from audit_camera_yaw_grid import CAMERA_MAXIMUM_RANGE, CAMERA_MINIMUM_RANGE
from audit_secondary_shoulder_mount_grid import (
    LENS_PROTRUSION_FROM_HOUSING_CENTER,
    _camera_quaternion_for_target,
)
from model import build_model, set_configuration
from protocol_drawer_scene import (
    DRAWER_Y,
    DRAWER_Z,
    formal_drawer_two_camera_scene,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--point", default="0.35,0.454108688,0.48"
    )
    args = parser.parse_args()
    point = np.asarray([float(value) for value in args.point.split(",")])

    scene = formal_drawer_two_camera_scene()
    context = _trajectory_context(args.run_dir, scene)
    truth = context["truth"]
    truth_index = int(np.argmin(np.linalg.norm(truth - point, axis=1)))
    risk_local = int(
        np.flatnonzero(context["risk_indices"] == truth_index)[0]
    )
    deadline = int(context["deadlines"][risk_local])

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

    wrist_quat, wrist_optical = _wrist_quaternion(20.0, 15.0)
    model.cam_quat[wrist_id] = wrist_quat
    model.geom_quat[wrist_housing_id] = wrist_quat
    model.geom_pos[wrist_housing_id] = (
        wrist_position
        - LENS_PROTRUSION_FROM_HOUSING_CENTER * wrist_optical
    )
    local_position = np.array([0.0, 0.075, 0.31])
    target = np.array([0.35, DRAWER_Y - 0.18, DRAWER_Z - 0.06])
    base_quat, optical_local = _camera_quaternion_for_target(
        body_rotation, body_position, local_position, target
    )
    forearm_quat = _rolled_quaternion(base_quat, -15.0)
    model.cam_pos[forearm_id] = local_position
    model.cam_quat[forearm_id] = forearm_quat
    model.geom_pos[forearm_housing_id] = (
        local_position
        - LENS_PROTRUSION_FROM_HOUSING_CENTER * optical_local
    )
    model.geom_quat[forearm_housing_id] = forearm_quat
    model.cam_fovy[wrist_id] = 58.0
    model.cam_fovy[forearm_id] = 58.0

    option = mujoco.MjvOption()
    option.geomgroup[1] = 1
    geomgroup = np.asarray(option.geomgroup, dtype=np.uint8)
    rows = []
    for cycle in context["camera_cycles"]:
        if cycle > deadline:
            break
        set_configuration(model, data, context["trajectory"][cycle])
        for camera_id in (wrist_id, forearm_id):
            position = data.cam_xpos[camera_id]
            rotation = data.cam_xmat[camera_id].reshape(3, 3)
            displacement = truth[truth_index] - position
            target_range = float(np.linalg.norm(displacement))
            local = displacement @ rotation
            depth = -float(local[2])
            half_y = 0.5 * math.radians(float(model.cam_fovy[camera_id]))
            half_x = math.atan(math.tan(half_y) * (320.0 / 180.0))
            horizontal = math.atan2(float(local[0]), depth)
            vertical = math.atan2(float(local[1]), depth)
            in_frustum = bool(
                depth > 0.0
                and abs(horizontal) <= half_x
                and abs(vertical) <= half_y
                and CAMERA_MINIMUM_RANGE <= target_range <= CAMERA_MAXIMUM_RANGE
            )
            hit_geom = np.array([-1], dtype=np.int32)
            hit_range = (
                float(
                    mujoco.mj_ray(
                        model,
                        data,
                        position,
                        displacement / target_range,
                        geomgroup,
                        True,
                        -1,
                        hit_geom,
                    )
                )
                if in_frustum
                else -1.0
            )
            geom_name = (
                None
                if hit_geom[0] < 0
                else model.geom(int(hit_geom[0])).name
            )
            hit_point = (
                None
                if hit_range < 0.0
                else (
                    position + hit_range * displacement / target_range
                ).tolist()
            )
            rows.append(
                {
                    "cycle": int(cycle),
                    "camera": model.camera(camera_id).name,
                    "in_frustum": in_frustum,
                    "target_range_m": target_range,
                    "first_hit_range_m": hit_range,
                    "first_hit_minus_target_m": (
                        None if hit_range < 0.0 else hit_range - target_range
                    ),
                    "first_hit_geom": geom_name,
                    "first_hit_point_m": hit_point,
                }
            )
    in_frustum_hits = [
        row
        for row in rows
        if row["in_frustum"] and row["first_hit_range_m"] >= 0.0
    ]
    closest = min(
        in_frustum_hits,
        key=lambda row: abs(row["first_hit_minus_target_m"]),
    )
    result = {
        "run_dir": str(args.run_dir.resolve()),
        "requested_point_m": point.tolist(),
        "nearest_truth_point_m": truth[truth_index].tolist(),
        "truth_point_error_m": float(
            np.linalg.norm(truth[truth_index] - point)
        ),
        "truth_index": truth_index,
        "deadline_cycle": deadline,
        "closest_first_hit": closest,
        "same_physical_obstacle_first_hit": bool(
            closest["first_hit_geom"] == "drawer_bottom"
        ),
        "rows": rows,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
