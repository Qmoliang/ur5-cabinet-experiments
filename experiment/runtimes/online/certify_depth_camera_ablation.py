"""Offline feasibility certificate for the frozen UR5 wrist-depth ablation.

The controller never receives drawer box geometry.  This audit is deliberately
offline: exact scene dimensions define the mandatory drawer cross-section,
while the obstacle certificate is the same frozen depth-derived proxy set used
by both LiuQP runs.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np

from analyze_shelf_drawer_feasibility import (
    ellipsoid_clearance,
    solve_pose,
    sphere_clearance,
    sphere_cross_section_barrier,
)
from model import (
    attachment_position,
    build_model,
    build_robot_certificate,
    certificate_world_state,
    set_configuration,
)
from multilevel_voxel_table import MultilevelVoxelTable
from run_depth_camera_ablation import RESULTS_ROOT, load_sensor_proxy_set
from run_simulation import minimum_mujoco_self_contact, minimum_mujoco_world_contact
from shelf_drawer_scene import shelf_drawer_certified_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=401)
    parser.add_argument("--ik-samples", type=int, default=81)
    parser.add_argument("--proxy-set", type=Path, default=None)
    parser.add_argument("--long-tool", action="store_true")
    parser.add_argument("--goal-x", type=float, default=None)
    parser.add_argument("--barrier-x", type=float, default=0.620)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    scene = shelf_drawer_certified_scene()
    if args.long_tool:
        scene = replace(scene, name="shelf_drawer_strict_long_tool")
    proxies = load_sensor_proxy_set(args.proxy_set)
    if proxies.proxy_offset_radii is None:
        raise RuntimeError("sensor proxy set is missing per-proxy uncertainty radii")
    offsets = np.asarray(proxies.proxy_offset_radii, dtype=float)
    effective_sphere_radii = np.asarray(proxies.sphere_radii, dtype=float) + offsets

    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(scene.q0, dtype=float))
    start = attachment_position(model, data)
    goal_x = float(scene.waypoints[-1][0] if args.goal_x is None else args.goal_x)
    goal = (goal_x, float(start[1]), float(start[2]))
    scene = replace(scene, waypoints=(tuple(start), goal))
    robot = build_robot_certificate(model)
    barrier = sphere_cross_section_barrier(
        scene,
        model,
        robot,
        proxies.centers,
        effective_sphere_radii,
        0.0,
        x_plane=args.barrier_x,
        grid_size=args.grid_size,
    )

    table = MultilevelVoxelTable.from_ellipsoids(
        proxies.centers,
        proxies.ellipsoid_shapes,
        voxel_size=max(item.radius for item in robot) + 0.003,
        query_padding=0.16,
        offset_radii=offsets,
    )
    site_id = model.site("attachment_site").id
    target_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    q = np.asarray(scene.q0, dtype=float)
    rows = []
    xs = np.linspace(scene.waypoints[0][0], scene.waypoints[-1][0], args.ik_samples)
    for x in xs:
        target = np.array([x, scene.waypoints[-1][1], scene.waypoints[-1][2]])
        q, ik_error = solve_pose(model, data, q, target, target_rotation)
        positions, _, radii = certificate_world_state(model, data, robot)
        rows.append(
            {
                "x_m": float(x),
                "ik_position_error_m": ik_error,
                "exact_world_contact_distance_m": minimum_mujoco_world_contact(data, model),
                "exact_self_contact_distance_m": minimum_mujoco_self_contact(data, model),
                "sphere_proxy_clearance_m": sphere_clearance(
                    positions,
                    radii,
                    proxies.centers,
                    effective_sphere_radii,
                    0.006,
                ),
                "ellipsoid_proxy_clearance_m": ellipsoid_clearance(
                    positions,
                    radii,
                    proxies.centers,
                    proxies.ellipsoid_shapes,
                    table,
                    0.006,
                    offsets,
                ),
                "attachment_error_m": float(np.linalg.norm(attachment_position(model, data) - target)),
            }
        )

    finite_world = [row["exact_world_contact_distance_m"] for row in rows if np.isfinite(row["exact_world_contact_distance_m"])]
    finite_self = [row["exact_self_contact_distance_m"] for row in rows if np.isfinite(row["exact_self_contact_distance_m"])]
    result = {
        "scene": scene.name,
        "proxy_source": "frozen UR5 wrist-mounted depth scan; XYZ only for fitting",
        "controller_uses_ground_truth_geometry": False,
        "offline_audit_uses_exact_drawer_dimensions": True,
        "matched_proxy_count": int(len(proxies.centers)),
        "raw_depth_point_count": int(len(proxies.raw_points)),
        "centervox_point_count": int(len(proxies.filtered_points)),
        "mean_proxy_uncertainty_m": float(np.mean(offsets)),
        "maximum_proxy_uncertainty_m": float(np.max(offsets)),
        "sphere_cross_section_barrier": barrier,
        "fixed_orientation_ik_audit": {
            "samples": len(rows),
            "maximum_ik_position_error_m": float(max(row["ik_position_error_m"] for row in rows)),
            "minimum_sphere_proxy_clearance_m": float(min(row["sphere_proxy_clearance_m"] for row in rows)),
            "minimum_ellipsoid_proxy_clearance_m": float(min(row["ellipsoid_proxy_clearance_m"] for row in rows)),
            "exact_world_collision_free": not finite_world,
            "exact_self_collision_free": not finite_self,
            "interpretation": "kinematic existence audit only; these poses are not supplied to either QP controller",
        },
        "rows": rows,
    }
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    output = args.output or RESULTS_ROOT / "wrist_depth_feasibility_certificate.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
