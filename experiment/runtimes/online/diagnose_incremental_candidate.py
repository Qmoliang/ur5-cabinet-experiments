"""Oracle-only geometry diagnostics before freezing incremental drawer v2.

This script may read analytic box surfaces because it is a scene-design audit.
Its output is never accepted by the online controller and is marked
``oracle_development_only``.  A formal scene is frozen only after a candidate
has both a strict sphere cross-section barrier and positive ellipsoid clearance.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math

import numpy as np

from incremental_drawer_scene import (
    BACK_X,
    CENTER_X,
    DRAWER_Y,
    DRAWER_Z,
    FRONT_X,
    HALF_X,
    WALL,
    incremental_drawer_scene,
)
from iris_taskspace_planner import EllipsoidToolFreeSpace
from model import BoxObstacle, build_model, build_robot_certificate
from pointcloud_proxy import build_shelf_drawer_proxy_set


def candidate_scene(inner_half: float):
    base = incremental_drawer_scene()
    boxes = []
    for box in base.boxes:
        if box.name == "drawer_bottom":
            box = BoxObstacle(
                box.name,
                (CENTER_X, DRAWER_Y, DRAWER_Z - inner_half - WALL),
                (HALF_X, inner_half + 0.045, WALL),
                box.material,
            )
        elif box.name == "drawer_left":
            box = BoxObstacle(
                box.name,
                (CENTER_X, DRAWER_Y - inner_half - WALL, DRAWER_Z),
                (HALF_X, WALL, inner_half),
                box.material,
            )
        elif box.name == "drawer_right":
            box = BoxObstacle(
                box.name,
                (CENTER_X, DRAWER_Y + inner_half + WALL, DRAWER_Z),
                (HALF_X, WALL, inner_half),
                box.material,
            )
        elif box.name == "drawer_ceiling":
            box = BoxObstacle(
                box.name,
                (CENTER_X, DRAWER_Y, DRAWER_Z + inner_half + WALL),
                (HALF_X, 0.32, WALL),
                box.material,
            )
        elif box.name == "drawer_back":
            box = BoxObstacle(
                box.name,
                (BACK_X + WALL, DRAWER_Y, DRAWER_Z),
                (WALL, inner_half + 0.045, inner_half + WALL),
                box.material,
            )
        boxes.append(box)
    return replace(
        base,
        name=f"incremental_drawer_candidate_{int(round(2000*inner_half)):03d}mm",
        boxes=tuple(boxes),
    )


def diagnose(inner_half: float, cluster_size: float) -> dict:
    scene = candidate_scene(inner_half)
    model = build_model(scene)
    robot = build_robot_certificate(model)
    wrist_radius = max(
        item.radius
        for item in robot
        if item.body_name in {"wrist_2_link", "wrist_3_link"}
    )
    proxies = build_shelf_drawer_proxy_set(
        scene.boxes,
        point_spacing=0.008,
        filter_size=0.006,
        cluster_size=cluster_size,
        maximum_aabb_overshoot=0.025,
    )
    offsets = (
        proxies.proxy_offset_radii
        if proxies.proxy_offset_radii is not None
        else np.full(len(proxies.centers), proxies.surface_cover_radius)
    )
    safety = 0.006
    center_half = inner_half - wrist_radius - safety
    ys = np.linspace(DRAWER_Y - center_half, DRAWER_Y + center_half, 101)
    zs = np.linspace(DRAWER_Z - center_half, DRAWER_Z + center_half, 101)
    yz = np.array(np.meshgrid(ys, zs, indexing="ij")).reshape(2, -1).T
    points = np.c_[np.full(len(yz), FRONT_X), yz]
    inflated = proxies.sphere_radii + offsets + wrist_radius + safety
    sampled_max = -np.inf
    for chunk in np.array_split(points, 32):
        clearance = (
            np.linalg.norm(
                chunk[:, None, :] - proxies.centers[None, :, :], axis=2
            )
            - inflated[None, :]
        )
        sampled_max = max(sampled_max, float(np.max(np.min(clearance, axis=1))))
    grid_cover = math.sqrt(2.0) * 0.5 * float(ys[1] - ys[0])
    free_space = EllipsoidToolFreeSpace(
        proxies.centers,
        proxies.ellipsoid_shapes,
        offsets,
        wrist_radius,
        safety,
    )
    center_clearance = free_space.clearance(
        np.array([FRONT_X, DRAWER_Y, DRAWER_Z])
    )
    return {
        "evidence_role": "oracle_development_only_not_controller_input",
        "inner_opening_m": 2.0 * inner_half,
        "cluster_size_m": cluster_size,
        "proxy_count": len(proxies.centers),
        "wrist_certificate_radius_m": wrist_radius,
        "physical_center_half_width_m": center_half,
        "sphere_sampled_max_clearance_m": sampled_max,
        "sphere_continuous_upper_bound_m": sampled_max + grid_cover,
        "sphere_section_closed": sampled_max + grid_cover < 0.0,
        "ellipsoid_center_exact_clearance_m": center_clearance,
    }


def main() -> None:
    rows = [
        diagnose(inner_half, cluster_size)
        for inner_half in (0.095, 0.100, 0.105, 0.110, 0.115, 0.120)
        for cluster_size in (0.080, 0.100, 0.120)
    ]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
