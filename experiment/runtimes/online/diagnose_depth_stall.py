"""Diagnose the tight support planes in a sensor-driven drawer run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

from ellipsoid_model import ellipsoid_world_state, spheres_as_isotropic_ellipsoids
from ellipsoid_qp_controller import EllipsoidLiuQPController
from model import build_model, build_robot_certificate, set_configuration
from shelf_drawer_scene import shelf_drawer_certified_scene


ROOT = Path(__file__).resolve().parent


def latest_trajectory() -> Path:
    matches = sorted(
        (ROOT / "results_geometry_ablation").glob(
            "shelf_drawer_certified_ellipsoid_*wrist_depth_dur030s/trajectory.csv"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError("no 30 s wrist-depth ellipsoid trajectory found")
    return matches[0]


def box_distance(point: np.ndarray, center: np.ndarray, half: np.ndarray) -> float:
    delta = np.abs(point - center) - half
    outside = np.maximum(delta, 0.0)
    if np.any(delta > 0.0):
        return float(np.linalg.norm(outside))
    return float(-np.min(-delta))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument(
        "--proxy-set",
        type=Path,
        default=ROOT / "results_depth_camera" / "wrist_depth_proxy_set.npz",
    )
    parser.add_argument("--count", type=int, default=25)
    args = parser.parse_args()
    trajectory = latest_trajectory() if args.trajectory is None else args.trajectory
    with trajectory.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    final = rows[-1]
    q = np.asarray([float(final[f"q{i}_rad"]) for i in range(1, 7)])

    proxy_data = np.load(args.proxy_set)
    centers = proxy_data["centers"]
    shapes = proxy_data["ellipsoid_shapes"]
    offsets = proxy_data["proxy_offset_radii"]

    scene = shelf_drawer_certified_scene()
    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, q)
    robot = spheres_as_isotropic_ellipsoids(build_robot_certificate(model))
    controller = EllipsoidLiuQPController(
        model,
        data,
        scene,
        robot,
        centers,
        shapes,
        obstacle_offsets=offsets,
        obstacle_index=None,
    )
    positions, _, _, robot_shapes = ellipsoid_world_state(model, data, robot)
    records = []
    for robot_index, (position, robot_shape) in enumerate(
        zip(positions, robot_shapes)
    ):
        for plane in controller.prune_redundant_obstacle_ellipsoids(
            position, robot_shape, robot_index=robot_index
        ):
            obstacle_index = plane.obstacle_index
            eigenvalues, rotation = np.linalg.eigh(shapes[obstacle_index])
            box_distances = [
                box_distance(
                    centers[obstacle_index],
                    np.asarray(box.center),
                    np.asarray(box.half_size),
                )
                for box in scene.boxes
            ]
            nearest_box = int(np.argmin(box_distances))
            records.append(
                (
                    plane.clearance,
                    robot_index,
                    robot[robot_index].body_name,
                    obstacle_index,
                    scene.boxes[nearest_box].name,
                    centers[obstacle_index],
                    np.sqrt(np.maximum(eigenvalues, 0.0)),
                    rotation,
                    offsets[obstacle_index],
                    plane.normal_to_obstacle,
                )
            )
    records.sort(key=lambda item: item[0])
    print(f"trajectory={trajectory}")
    print(f"final_error_m={float(final['final_target_error_m']):.9f}")
    print(f"active_planes={len(records)}")
    for record in records[: args.count]:
        (
            clearance,
            robot_index,
            body_name,
            obstacle_index,
            box_name,
            center,
            axes,
            rotation,
            offset,
            normal,
        ) = record
        smallest_axis = rotation[:, int(np.argmin(axes))]
        print(
            "clearance={:.6f} robot={}:{} obstacle={} box={} "
            "center={} axes={} offset={:.6f} normal={} thin_axis={}".format(
                clearance,
                robot_index,
                body_name,
                obstacle_index,
                box_name,
                np.round(center, 5),
                np.round(axes, 5),
                offset,
                np.round(normal, 5),
                np.round(smallest_axis, 5),
            )
        )


if __name__ == "__main__":
    main()
