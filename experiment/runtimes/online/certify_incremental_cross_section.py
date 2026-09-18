"""Continuous front-section certificate from one causal planning snapshot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from ellipsoid_model import build_robot_ellipsoid_certificate, ellipsoid_world_state
from iris_taskspace_planner import EllipsoidToolFreeSpace
from model import (
    attachment_position,
    build_model,
    build_robot_certificate,
    certificate_world_state,
    set_configuration,
)
from incremental_drawer_scene import incremental_drawer_v12_certified_candidate_scene


SAFETY_MARGIN = 0.001


def _physical_attachment_domain(
    opening_lower_yz: np.ndarray,
    opening_upper_yz: np.ndarray,
    tool_offsets: np.ndarray,
    tool_half_extents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.max(
        opening_lower_yz[None, :] - tool_offsets[:, 1:3] + tool_half_extents[:, 1:3],
        axis=0,
    )
    upper = np.min(
        opening_upper_yz[None, :] - tool_offsets[:, 1:3] - tool_half_extents[:, 1:3],
        axis=0,
    )
    if np.any(lower >= upper):
        raise RuntimeError("tool certificate has an empty physical opening domain")
    return lower, upper


def _grid(lower: np.ndarray, upper: np.ndarray, grid_size: int, plane_x: float):
    ys = np.linspace(lower[0], upper[0], grid_size)
    zs = np.linspace(lower[1], upper[1], grid_size)
    yz = np.array(np.meshgrid(ys, zs, indexing="ij")).reshape(2, -1).T
    points = np.c_[np.full(len(yz), plane_x), yz]
    step_y = float(ys[1] - ys[0])
    step_z = float(zs[1] - zs[0])
    cover = 0.5 * math.sqrt(step_y**2 + step_z**2)
    return ys, zs, points, cover


def _sphere_chain_clearance(
    points: np.ndarray,
    obstacle_centers: np.ndarray,
    obstacle_radii: np.ndarray,
    tool_offsets: np.ndarray,
    tool_radii: np.ndarray,
) -> np.ndarray:
    result = np.full(len(points), np.inf)
    for tool_offset, tool_radius in zip(tool_offsets, tool_radii):
        tool_centers = points + tool_offset
        for indices in np.array_split(np.arange(len(points)), 32):
            values = np.min(
                np.linalg.norm(
                    tool_centers[indices, None, :]
                    - obstacle_centers[None, :, :],
                    axis=2,
                )
                - obstacle_radii[None, :]
                - float(tool_radius)
                - SAFETY_MARGIN,
                axis=1,
            )
            result[indices] = np.minimum(result[indices], values)
    return result


def certify(snapshot_path: Path, grid_size: int = 51) -> dict:
    snapshot = np.load(snapshot_path)
    centers = np.asarray(snapshot["centers"], dtype=float)
    sphere_radii = np.asarray(snapshot["sphere_radii"], dtype=float)
    shapes = np.asarray(snapshot["ellipsoid_shapes"], dtype=float)
    uncertainty_shapes = np.asarray(
        snapshot["proxy_uncertainty_shapes"], dtype=float
    )
    pruning_shapes = np.asarray(snapshot["ellipsoid_outer_shapes"], dtype=float)
    offsets = np.asarray(snapshot["proxy_offsets"], dtype=float)

    scene = incremental_drawer_v12_certified_candidate_scene()
    boxes = {box.name: box for box in scene.boxes}
    left, right = boxes["drawer_left"], boxes["drawer_right"]
    bottom, ceiling = boxes["drawer_bottom"], boxes["drawer_ceiling"]
    opening_lower_yz = np.array(
        [left.center[1] + left.half_size[1],
         bottom.center[2] + bottom.half_size[2]],
        dtype=float,
    )
    opening_upper_yz = np.array(
        [right.center[1] - right.half_size[1],
         ceiling.center[2] - ceiling.half_size[2]],
        dtype=float,
    )
    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(scene.q0, dtype=float))
    attachment = attachment_position(model, data)
    attachment_body = int(model.site_bodyid[model.site("attachment_site").id])

    sphere_tool = build_robot_certificate(model)
    sphere_positions, _, sphere_tool_radii = certificate_world_state(
        model, data, sphere_tool
    )
    sphere_mask = np.asarray(
        [item.body_id == attachment_body for item in sphere_tool], dtype=bool
    )
    sphere_tool_offsets = sphere_positions[sphere_mask] - attachment
    sphere_tool_radii = sphere_tool_radii[sphere_mask]
    sphere_half_extents = np.repeat(sphere_tool_radii[:, None], 3, axis=1)
    sphere_lower, sphere_upper_domain = _physical_attachment_domain(
        opening_lower_yz, opening_upper_yz,
        sphere_tool_offsets, sphere_half_extents
    )
    # Fixed, auditable selection rule: scan every 10 mm plane from 0.30 m to
    # the goal with a 31x31 grid and choose the first plane whose 1-Lipschitz
    # upper bound is strictly negative.  The selected plane is then recomputed
    # at the requested verification resolution for both representations.
    scan_records = []
    selected_plane = None
    goal_x = float(scene.waypoints[-1][0])
    for plane_x in np.arange(0.30, goal_x + 1.0e-7, 0.01):
        _, _, scan_points, scan_cover = _grid(
            sphere_lower, sphere_upper_domain, 31, float(plane_x)
        )
        scan_clearance = _sphere_chain_clearance(
            scan_points,
            centers,
            sphere_radii + offsets,
            sphere_tool_offsets,
            sphere_tool_radii,
        )
        continuous_upper = float(np.max(scan_clearance) + scan_cover)
        scan_records.append(
            {
                "plane_x_m": float(plane_x),
                "sphere_continuous_upper_m": continuous_upper,
            }
        )
        if selected_plane is None and continuous_upper < 0.0:
            selected_plane = float(plane_x)
    if selected_plane is None:
        best_scan = min(
            scan_records, key=lambda row: row["sphere_continuous_upper_m"]
        )
        failed = {
            "snapshot": str(snapshot_path.resolve()),
            "evidence_role": "failed_continuous_cross_section_gate",
            "plane_selection_rule": f"first 10 mm scan plane in [0.30,{goal_x:.2f}] with negative 31x31-plus-Lipschitz sphere upper bound",
            "sphere_plane_scan": scan_records,
            "sphere_section_strictly_closed": False,
            "best_scanned_plane_x_m": best_scan["plane_x_m"],
            "best_sphere_continuous_upper_bound_m": best_scan[
                "sphere_continuous_upper_m"
            ],
        }
        output = snapshot_path.parent / "continuous_cross_section"
        output.mkdir(exist_ok=True)
        (output / "failed_certificate.json").write_text(
            json.dumps(failed, indent=2), encoding="utf-8"
        )
        return failed
    sphere_ys, sphere_zs, sphere_points, sphere_cover = _grid(
        sphere_lower, sphere_upper_domain, grid_size, selected_plane
    )
    sphere_clearance = _sphere_chain_clearance(
        sphere_points,
        centers,
        sphere_radii + offsets,
        sphere_tool_offsets,
        sphere_tool_radii,
    )

    ellipsoid_tool = build_robot_ellipsoid_certificate(model)
    ellipsoid_positions, _, _, ellipsoid_tool_shapes = ellipsoid_world_state(
        model, data, ellipsoid_tool
    )
    ellipsoid_mask = np.asarray(
        [item.body_id == attachment_body for item in ellipsoid_tool], dtype=bool
    )
    ellipsoid_tool_offsets = ellipsoid_positions[ellipsoid_mask] - attachment
    ellipsoid_tool_shapes = ellipsoid_tool_shapes[ellipsoid_mask]
    ellipsoid_half_extents = np.sqrt(
        np.maximum(
            np.diagonal(ellipsoid_tool_shapes, axis1=1, axis2=2), 0.0
        )
    )
    ellipsoid_lower_domain, ellipsoid_upper_domain = _physical_attachment_domain(
        opening_lower_yz, opening_upper_yz,
        ellipsoid_tool_offsets, ellipsoid_half_extents
    )
    ellipsoid_ys, ellipsoid_zs, ellipsoid_points, ellipsoid_cover = _grid(
        ellipsoid_lower_domain,
        ellipsoid_upper_domain,
        grid_size,
        selected_plane,
    )
    free_space = EllipsoidToolFreeSpace(
        centers,
        shapes,
        offsets,
        0.0,
        SAFETY_MARGIN,
        tool_shapes=ellipsoid_tool_shapes,
        tool_offsets=ellipsoid_tool_offsets,
        obstacle_uncertainty_shapes=uncertainty_shapes,
        obstacle_pruning_shapes=pruning_shapes,
    )
    ellipsoid_clearance = np.asarray(
        [free_space.clearance(point) for point in ellipsoid_points]
    )

    sphere_sampled_max = float(np.max(sphere_clearance))
    sphere_continuous_upper = sphere_sampled_max + sphere_cover
    ellipsoid_best_index = int(np.argmax(ellipsoid_clearance))
    ellipsoid_best = ellipsoid_points[ellipsoid_best_index]
    ellipsoid_sampled_max = float(ellipsoid_clearance[ellipsoid_best_index])
    boundary_distance = float(
        np.min(
            np.r_[
                ellipsoid_best[1:3] - ellipsoid_lower_domain,
                ellipsoid_upper_domain - ellipsoid_best[1:3],
            ]
        )
    )
    ellipsoid_open_cell_lower = min(
        ellipsoid_sampled_max - ellipsoid_cover,
        boundary_distance - ellipsoid_cover,
    )
    result = {
        "snapshot": str(snapshot_path.resolve()),
        "evidence_role": "paired_causal_planning_snapshot_no_analytic_surface_points",
        "topological_reason": "every continuous start-to-goal path must cross the selected constant-x plane",
        "plane_selection_rule": f"first 10 mm scan plane in [0.30,{goal_x:.2f}] with negative 31x31-plus-Lipschitz sphere upper bound",
        "selected_mandatory_plane_x_m": selected_plane,
        "sphere_plane_scan": scan_records,
        "grid_size": grid_size,
        "safety_margin_m": SAFETY_MARGIN,
        "physical_opening_lower_yz_m": opening_lower_yz.tolist(),
        "physical_opening_upper_yz_m": opening_upper_yz.tolist(),
        "proxy_count": len(centers),
        "sphere_tool_proxy_count": int(np.sum(sphere_mask)),
        "sphere_attachment_domain_yz_m": [sphere_lower.tolist(), sphere_upper_domain.tolist()],
        "sphere_lipschitz_cover_radius_m": sphere_cover,
        "sphere_sampled_max_clearance_m": sphere_sampled_max,
        "sphere_continuous_maximum_upper_bound_m": sphere_continuous_upper,
        "sphere_section_strictly_closed": sphere_continuous_upper < 0.0,
        "sphere_closure_margin_m": max(0.0, -sphere_continuous_upper),
        "ellipsoid_tool_proxy_count": int(np.sum(ellipsoid_mask)),
        "ellipsoid_attachment_domain_yz_m": [ellipsoid_lower_domain.tolist(), ellipsoid_upper_domain.tolist()],
        "ellipsoid_lipschitz_cover_radius_m": ellipsoid_cover,
        "ellipsoid_sampled_best_clearance_m": ellipsoid_sampled_max,
        "ellipsoid_open_grid_cell_lower_bound_m": ellipsoid_open_cell_lower,
        "ellipsoid_section_has_strict_open_ball": ellipsoid_open_cell_lower > 0.0,
        "ellipsoid_open_margin_m": max(0.0, ellipsoid_open_cell_lower),
        "ellipsoid_best_attachment_point_m": ellipsoid_best.tolist(),
    }
    output = snapshot_path.parent / "continuous_cross_section"
    output.mkdir(exist_ok=True)
    (output / "certificate.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output / "clearance_fields.npz",
        sphere_ys=sphere_ys,
        sphere_zs=sphere_zs,
        sphere_clearance=sphere_clearance.reshape(grid_size, grid_size),
        ellipsoid_ys=ellipsoid_ys,
        ellipsoid_zs=ellipsoid_zs,
        ellipsoid_clearance=ellipsoid_clearance.reshape(grid_size, grid_size),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--grid-size", type=int, default=51)
    args = parser.parse_args()
    print(json.dumps(certify(args.snapshot, args.grid_size), indent=2))


if __name__ == "__main__":
    main()
