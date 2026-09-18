"""Audit exact, sphere-proxy, and ellipsoid-proxy feasibility on one IK path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from ellipsoid_model import closest_point_normal_on_ellipsoid, support_radius
from model import (
    JOINT_NAMES,
    attachment_position,
    build_model,
    build_robot_certificate,
    certificate_world_state,
    set_configuration,
)
from multilevel_voxel_table import MultilevelVoxelTable
from pointcloud_proxy import build_shelf_drawer_proxy_set
from run_simulation import minimum_mujoco_self_contact, minimum_mujoco_world_contact
from shelf_drawer_scene import (
    shelf_drawer_certified_scene,
    shelf_drawer_narrow_scene,
    shelf_drawer_scene,
)


ROOT = Path(__file__).resolve().parent


def orientation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    error_matrix = target @ current.T
    return 0.5 * np.array(
        [
            error_matrix[2, 1] - error_matrix[1, 2],
            error_matrix[0, 2] - error_matrix[2, 0],
            error_matrix[1, 0] - error_matrix[0, 1],
        ]
    )


def solve_pose(model, data, seed, target_position, target_rotation):
    lower = np.array([model.jnt_range[model.joint(n).id, 0] for n in JOINT_NAMES])
    upper = np.array([model.jnt_range[model.joint(n).id, 1] for n in JOINT_NAMES])
    site_id = model.site("attachment_site").id

    def residual(q):
        set_configuration(model, data, q)
        current_rotation = data.site_xmat[site_id].reshape(3, 3)
        return np.r_[
            attachment_position(model, data) - target_position,
            0.5 * orientation_error(target_rotation, current_rotation),
        ]

    result = least_squares(
        residual,
        np.asarray(seed),
        bounds=(lower + 1.0e-5, upper - 1.0e-5),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=400,
    )
    set_configuration(model, data, result.x)
    return result.x.copy(), float(np.linalg.norm(residual(result.x)[:3]))


def sphere_clearance(positions, robot_radii, centers, radii, margin):
    distance = np.linalg.norm(positions[:, None, :] - centers[None, :, :], axis=2)
    return float(np.min(distance - robot_radii[:, None] - radii[None, :] - margin))


def ellipsoid_clearance(
    positions, robot_radii, centers, shapes, table, margin, obstacle_offset=0.0
):
    best = np.inf
    offsets = np.broadcast_to(np.asarray(obstacle_offset, dtype=float), (len(centers),))
    for position, radius in zip(positions, robot_radii):
        candidates = table.query_sphere(position, float(radius))
        for obstacle_index in candidates:
            center = centers[obstacle_index]
            shape = shapes[obstacle_index]
            normal = closest_point_normal_on_ellipsoid(center, shape, position)
            clearance = (
                float(normal @ (center - position))
                - float(radius)
                - support_radius(shape, normal)
                - offsets[obstacle_index]
                - margin
            )
            best = min(best, clearance)
    return float(best)


def sphere_cross_section_barrier(
    scene,
    model,
    robot,
    obstacle_centers,
    obstacle_radii,
    surface_cover_radius,
    x_plane: float = 0.620,
    grid_size: int = 201,
):
    """Upper-bound the best sphere clearance on a mandatory drawer section.

    The minimum of signed sphere distances is 1-Lipschitz.  Consequently the
    largest sampled clearance plus the half-cell diagonal is a continuous
    upper bound over the complete rectangular center domain.
    """

    tool_proxies = [
        item
        for item in robot
        if model.geom(item.source_geom_id).name == "drawer_retrieval_tool"
    ]
    tool_proxy_radius = min(item.radius for item in tool_proxies)
    tool_geom_id = model.geom("drawer_retrieval_tool").id
    exact_tool_radius = float(model.geom_size[tool_geom_id, 0])
    left = next(box for box in scene.boxes if box.name == "open_drawer_left_wall")
    inner_half = abs(left.center[1] - scene.waypoints[-1][1]) - left.half_size[1]
    center_half = inner_half - exact_tool_radius
    ys = np.linspace(scene.waypoints[-1][1] - center_half, scene.waypoints[-1][1] + center_half, grid_size)
    zs = np.linspace(scene.waypoints[-1][2] - center_half, scene.waypoints[-1][2] + center_half, grid_size)
    yz = np.array(np.meshgrid(ys, zs, indexing="ij")).reshape(2, -1).T
    points = np.c_[np.full(len(yz), x_plane), yz]
    sampled_best = -np.inf
    for chunk in np.array_split(points, 24):
        clearance = (
            np.linalg.norm(chunk[:, None, :] - obstacle_centers[None, :, :], axis=2)
            - obstacle_radii[None, :]
            - surface_cover_radius
            - tool_proxy_radius
            - 0.006
        )
        sampled_best = max(sampled_best, float(np.max(np.min(clearance, axis=1))))
    grid_step = 2.0 * center_half / max(grid_size - 1, 1)
    lipschitz_correction = grid_step / np.sqrt(2.0)
    return {
        "barrier_plane_x_m": x_plane,
        "cross_section_grid_size": grid_size,
        "exact_tool_radius_m": exact_tool_radius,
        "tool_certificate_sphere_radius_m": tool_proxy_radius,
        "center_domain_half_width_m": center_half,
        "sampled_maximum_sphere_clearance_m": sampled_best,
        "continuous_maximum_clearance_upper_bound_m": sampled_best
        + lipschitz_correction,
        "lipschitz_grid_correction_m": lipschitz_correction,
        "sphere_certificate_cross_section_closed": bool(
            sampled_best + lipschitz_correction < 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-size", type=float, default=0.100)
    parser.add_argument("--maximum-overshoot", type=float, default=0.025)
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--narrow", action="store_true")
    parser.add_argument("--certified", action="store_true")
    parser.add_argument("--point-spacing", type=float, default=0.012)
    parser.add_argument("--filter-size", type=float, default=0.012)
    args = parser.parse_args()

    if args.certified:
        scene = shelf_drawer_certified_scene()
    elif args.narrow:
        scene = shelf_drawer_narrow_scene()
    else:
        scene = shelf_drawer_scene()
    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(scene.q0))
    site_id = model.site("attachment_site").id
    target_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    proxies = build_shelf_drawer_proxy_set(
        scene.boxes,
        point_spacing=args.point_spacing,
        filter_size=args.filter_size,
        cluster_size=args.cluster_size,
        maximum_aabb_overshoot=args.maximum_overshoot,
    )
    robot = build_robot_certificate(model)
    table = MultilevelVoxelTable.from_ellipsoids(
        proxies.centers,
        proxies.ellipsoid_shapes,
        voxel_size=max(item.radius for item in robot) + 0.003,
        query_padding=0.16,
        offset_radii=proxies.surface_cover_radius,
    )

    q = np.asarray(scene.q0)
    xs = np.linspace(scene.waypoints[0][0], scene.waypoints[-1][0], args.samples)
    rows = []
    for x in xs:
        target = np.array([x, scene.waypoints[-1][1], scene.waypoints[-1][2]])
        q, ik_error = solve_pose(model, data, q, target, target_rotation)
        positions, _, robot_radii = certificate_world_state(model, data, robot)
        rows.append(
            {
                "x_m": float(x),
                "ik_position_error_m": ik_error,
                "exact_world_contact_distance_m": minimum_mujoco_world_contact(data, model),
                "exact_self_contact_distance_m": minimum_mujoco_self_contact(data, model),
                "sphere_proxy_clearance_m": sphere_clearance(
                    positions,
                    robot_radii,
                    proxies.centers,
                    proxies.sphere_radii + proxies.surface_cover_radius,
                    0.006,
                ),
                "ellipsoid_proxy_clearance_m": ellipsoid_clearance(
                    positions,
                    robot_radii,
                    proxies.centers,
                    proxies.ellipsoid_shapes,
                    table,
                    0.006,
                    proxies.surface_cover_radius,
                ),
            }
        )

    result = {
        "path_source": "sequential_fixed-orientation_IK_audit_not_controller_guidance",
        "samples": len(rows),
        "cluster_size_m": args.cluster_size,
        "maximum_overshoot_limit_m": args.maximum_overshoot,
        "surface_cover_radius_m": proxies.surface_cover_radius,
        "maximum_ik_position_error_m": max(r["ik_position_error_m"] for r in rows),
        "exact_path_collision_free": bool(
            all(not np.isfinite(r["exact_world_contact_distance_m"]) for r in rows)
            and all(not np.isfinite(r["exact_self_contact_distance_m"]) for r in rows)
        ),
        "minimum_sphere_proxy_clearance_m": min(
            r["sphere_proxy_clearance_m"] for r in rows
        ),
        "minimum_ellipsoid_proxy_clearance_m": min(
            r["ellipsoid_proxy_clearance_m"] for r in rows
        ),
        "goal_sphere_proxy_clearance_m": rows[-1]["sphere_proxy_clearance_m"],
        "goal_ellipsoid_proxy_clearance_m": rows[-1]["ellipsoid_proxy_clearance_m"],
        "sphere_cross_section_barrier": sphere_cross_section_barrier(
            scene,
            model,
            robot,
            proxies.centers,
            proxies.sphere_radii,
            proxies.surface_cover_radius,
        ),
        "rows": rows,
    }
    output = (
        ROOT
        / "results_geometry_ablation"
        / f"{scene.name}_pca_graph_mahal_cluster100mm_over025mm_feasibility.json"
    )
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
