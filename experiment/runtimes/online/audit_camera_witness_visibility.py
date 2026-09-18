"""Post-hoc frustum and first-hit audit for selected truth witness points."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from model import build_model, set_configuration
from protocol_drawer_scene import protocol_drawer_v6_scene


CAMERAS = (
    "ur5_depth_wrist",
    "ur5_depth_wrist_right",
    "ur5_depth_shoulder",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--point",
        action="append",
        required=True,
        help="comma-separated world XYZ; may be repeated",
    )
    args = parser.parse_args()

    points = [
        np.asarray([float(value) for value in item.split(",")], dtype=float)
        for item in args.point
    ]
    if any(point.shape != (3,) for point in points):
        raise ValueError("every witness point must contain exactly three coordinates")

    scene = protocol_drawer_v6_scene()
    model = build_model(scene)
    data = mujoco.MjData(model)
    q_history = np.load(args.run_dir / "q_history.npy")
    q_trajectory = np.vstack((np.asarray(scene.q0, dtype=float), q_history))
    with (args.run_dir / "perception_frames.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        source_cycles = [int(row["source_cycle"]) for row in csv.DictReader(stream)]

    option = mujoco.MjvOption()
    option.geomgroup[1] = 1
    rows: list[dict] = []
    for source_cycle in source_cycles:
        set_configuration(model, data, q_trajectory[source_cycle])
        for point_index, point in enumerate(points):
            for camera_name in CAMERAS:
                camera_id = model.camera(camera_name).id
                position = data.cam_xpos[camera_id].copy()
                rotation = data.cam_xmat[camera_id].reshape(3, 3).copy()
                local = rotation.T @ (point - position)
                depth = float(-local[2])
                fovy = math.radians(float(model.cam_fovy[camera_id]))
                half_y = math.atan(math.tan(0.5 * fovy))
                half_x = math.atan(math.tan(0.5 * fovy) * (240.0 / 180.0))
                horizontal = math.atan2(float(local[0]), depth)
                vertical = math.atan2(float(local[1]), depth)
                distance = float(np.linalg.norm(point - position))
                in_frustum = (
                    depth > 0.0
                    and abs(horizontal) <= half_x
                    and abs(vertical) <= half_y
                )
                direction = (point - position) / distance
                geom_id = np.array([-1], dtype=np.int32)
                first_hit = float(
                    mujoco.mj_ray(
                        model,
                        data,
                        position,
                        direction,
                        np.asarray(option.geomgroup, dtype=np.uint8),
                        True,
                        -1,
                        geom_id,
                    )
                )
                hit_name = (
                    None
                    if int(geom_id[0]) < 0
                    else mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id[0])
                    )
                )
                rows.append(
                    {
                        "source_cycle": source_cycle,
                        "point_index": point_index,
                        "camera": camera_name,
                        "range_m": distance,
                        "depth_m": depth,
                        "horizontal_angle_deg": math.degrees(horizontal),
                        "vertical_angle_deg": math.degrees(vertical),
                        "in_frustum": in_frustum,
                        "above_d405_minimum_range": distance >= 0.070,
                        "first_hit_range_m": first_hit,
                        "first_hit_geom": hit_name,
                        "truth_point_is_first_hit": (
                            in_frustum
                            and first_hit >= 0.0
                            and abs(first_hit - distance) <= 0.012
                        ),
                    }
                )
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
