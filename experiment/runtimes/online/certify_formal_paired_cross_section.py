"""Certify paired sphere closure and exact-ellipsoid opening from causal runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from model import (
    attachment_position,
    build_model,
    build_robot_certificate,
    certificate_world_state,
    set_configuration,
)
from native_ellipsoid_support import NativeEllipsoidSupport
from run_protocol_v3_async_online import _protocol_scene


SAFETY_MARGIN_M = 0.006


def _single_file(run_root: Path, name: str) -> Path:
    matches = list(run_root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {name} below {run_root}, found {len(matches)}"
        )
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _mandatory_attachment_sphere(scene):
    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(scene.q0, dtype=float))
    certificate = build_robot_certificate(model)
    positions, _, radii = certificate_world_state(model, data, certificate)
    attachment = attachment_position(model, data)
    site = model.site("attachment_site")
    attachment_body = int(model.site_bodyid[site.id])
    front_x = min(box.center[0] - box.half_size[0] for box in scene.boxes)
    target_x = float(scene.waypoints[-1][0])
    candidates = []
    for index, sphere in enumerate(certificate):
        if sphere.body_id != attachment_body:
            continue
        offset_bound = float(
            np.linalg.norm(
                np.asarray(sphere.local_center) - model.site_pos[site.id]
            )
        )
        target_lower_x = target_x - offset_bound
        if positions[index, 0] < front_x and target_lower_x > front_x:
            candidates.append(
                (
                    -float(radii[index]),
                    offset_bound,
                    index,
                    target_lower_x,
                )
            )
    if not candidates:
        raise RuntimeError("no attachment-body certificate sphere must cross front")
    _, offset_bound, index, target_lower_x = min(candidates)
    return {
        "index": int(index),
        "radius_m": float(radii[index]),
        "initial_center_m": positions[index].tolist(),
        "initial_attachment_m": attachment.tolist(),
        "attachment_offset_bound_m": float(offset_bound),
        "target_center_x_lower_bound_m": float(target_lower_x),
        "front_x_m": float(front_x),
        "body_name": certificate[index].body_name,
    }


def _opening(scene):
    boxes = {box.name: box for box in scene.boxes}
    left = boxes["drawer_left"]
    right = boxes["drawer_right"]
    bottom = boxes["drawer_bottom"]
    ceiling = boxes["drawer_ceiling"]
    lower = np.array(
        [
            left.center[1] + left.half_size[1],
            bottom.center[2] + bottom.half_size[2],
        ],
        dtype=float,
    )
    upper = np.array(
        [
            right.center[1] - right.half_size[1],
            ceiling.center[2] - ceiling.half_size[2],
        ],
        dtype=float,
    )
    return lower, upper


def _sphere_section(
    snapshot,
    plane_x: float,
    lower: np.ndarray,
    upper: np.ndarray,
    robot_radius: float,
    grid_size: int,
):
    centers = np.asarray(snapshot["centers"], dtype=float)
    radii = np.asarray(snapshot["sphere_radii"], dtype=float)
    offsets = np.asarray(snapshot["uncertainty_offsets"], dtype=float)
    effective = radii + offsets
    ys = np.linspace(lower[0], upper[0], grid_size)
    zs = np.linspace(lower[1], upper[1], grid_size)
    yz = np.array(np.meshgrid(ys, zs, indexing="ij")).reshape(2, -1).T
    points = np.c_[np.full(len(yz), plane_x), yz]
    cover = 0.5 * math.sqrt(
        float(ys[1] - ys[0]) ** 2 + float(zs[1] - zs[0]) ** 2
    )
    tree = cKDTree(centers)
    search_radius = float(
        np.max(effective) + robot_radius + SAFETY_MARGIN_M
    )
    def evaluate(query_points: np.ndarray) -> tuple[np.ndarray, int, int]:
        values = np.empty(len(query_points), dtype=float)
        candidate_total = 0
        candidate_maximum = 0
        for begin in range(0, len(query_points), 4096):
            batch = query_points[begin : begin + 4096]
            candidate_lists = tree.query_ball_point(
                batch, search_radius, workers=-1
            )
            for local_index, (query, indices) in enumerate(
                zip(batch, candidate_lists)
            ):
                candidate_total += len(indices)
                candidate_maximum = max(candidate_maximum, len(indices))
                if not indices:
                    # Every omitted proxy has nonnegative clearance by the
                    # search-radius construction, so zero is a conservative
                    # lower value for detecting an opening.
                    values[begin + local_index] = 0.0
                    continue
                selected = np.asarray(indices, dtype=np.int64)
                values[begin + local_index] = float(
                    np.min(
                        np.linalg.norm(centers[selected] - query, axis=1)
                        - effective[selected]
                        - robot_radius
                        - SAFETY_MARGIN_M
                    )
                )
        return values, candidate_total, candidate_maximum

    clearance, candidate_total, candidate_maximum = evaluate(points)
    best = int(np.argmax(clearance))
    sampled_max = float(clearance[best])
    uniform_grid_upper = sampled_max + cover

    # A uniform grid can be inconclusive when the true closure margin is
    # sub-millimetre.  Certify the entire continuous rectangle with adaptive
    # 1-Lipschitz branch-and-bound.  A cell centred at c with half-diagonal h
    # has max(phi) <= phi(c)+h.  Only cells whose upper bound is nonnegative
    # are split, so no interpolation or obstacle-surface sampling is assumed.
    cell_centers = np.array(
        [[plane_x, 0.5 * (lower[0] + upper[0]), 0.5 * (lower[1] + upper[1])]],
        dtype=float,
    )
    half_y = np.array([0.5 * (upper[0] - lower[0])], dtype=float)
    half_z = np.array([0.5 * (upper[1] - lower[1])], dtype=float)
    certified_upper_max = -np.inf
    evaluated_cells = 0
    adaptive_depth = 0
    open_witness = None
    unresolved_upper = float("inf")
    for depth in range(17):
        adaptive_depth = depth
        cell_clearance, local_total, local_maximum = evaluate(cell_centers)
        candidate_total += local_total
        candidate_maximum = max(candidate_maximum, local_maximum)
        evaluated_cells += len(cell_centers)
        positive = np.flatnonzero(cell_clearance >= 0.0)
        if len(positive):
            witness = int(positive[0])
            open_witness = {
                "point_m": cell_centers[witness].tolist(),
                "clearance_m": float(cell_clearance[witness]),
            }
            unresolved_upper = float(
                np.max(
                    cell_clearance
                    + np.sqrt(half_y * half_y + half_z * half_z)
                )
            )
            break
        upper_bounds = cell_clearance + np.sqrt(
            half_y * half_y + half_z * half_z
        )
        certified = upper_bounds < 0.0
        if np.any(certified):
            certified_upper_max = max(
                certified_upper_max, float(np.max(upper_bounds[certified]))
            )
        unresolved = ~certified
        if not np.any(unresolved):
            unresolved_upper = certified_upper_max
            break
        unresolved_upper = float(np.max(upper_bounds[unresolved]))
        if depth == 16:
            break
        parent_centers = cell_centers[unresolved]
        parent_half_y = half_y[unresolved]
        parent_half_z = half_z[unresolved]
        child_half_y = 0.5 * parent_half_y
        child_half_z = 0.5 * parent_half_z
        child_centers = []
        for sign_y, sign_z in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            child = parent_centers.copy()
            child[:, 1] += sign_y * child_half_y
            child[:, 2] += sign_z * child_half_z
            child_centers.append(child)
        cell_centers = np.concatenate(child_centers, axis=0)
        half_y = np.tile(child_half_y, 4)
        half_z = np.tile(child_half_z, 4)
        if len(cell_centers) > 2_000_000:
            raise RuntimeError("adaptive cross-section certificate exceeded cell cap")
    adaptive_closed = bool(
        open_witness is None
        and len(cell_centers) > 0
        and unresolved_upper < 0.0
    )
    return {
        "grid_size": int(grid_size),
        "lipschitz_constant": 1.0,
        "grid_cover_radius_m": float(cover),
        "sampled_maximum_clearance_m": sampled_max,
        "uniform_grid_continuous_upper_bound_m": float(uniform_grid_upper),
        "adaptive_lipschitz_depth": int(adaptive_depth),
        "adaptive_cells_evaluated": int(evaluated_cells),
        "continuous_maximum_upper_bound_m": float(unresolved_upper),
        "strictly_closed": adaptive_closed,
        "closure_margin_m": float(max(0.0, -unresolved_upper)),
        "open_witness": open_witness,
        "sampled_least_blocked_point_m": points[best].tolist(),
        "candidate_count_mean": float(
            candidate_total / max(1, len(points) + evaluated_cells)
        ),
        "candidate_count_max": int(candidate_maximum),
    }


def _ellipsoid_open_point(
    snapshot,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    robot_radius: float,
):
    centers = np.asarray(snapshot["centers"], dtype=float)
    shapes = np.asarray(snapshot["ellipsoid_shapes"], dtype=float)
    uncertainty = np.asarray(
        snapshot["proxy_uncertainty_shapes"], dtype=float
    )
    offsets = np.asarray(snapshot["uncertainty_offsets"], dtype=float)
    kernel = NativeEllipsoidSupport()
    point = np.asarray(point, dtype=float)
    center_lines = centers - point[None, :]
    center_norms = np.linalg.norm(center_lines, axis=1)
    initial = np.zeros_like(center_lines)
    nonzero = center_norms > 1.0e-15
    initial[nonzero] = center_lines[nonzero] / center_norms[nonzero, None]
    robot_shape = np.eye(3) * float(robot_radius) ** 2
    normals, newton, residuals = kernel.normals_sum_newton_warm(
        point,
        robot_shape,
        centers,
        shapes,
        uncertainty,
        initial,
        max_iterations=16,
    )
    retry = residuals > 1.0e-7
    if np.any(retry):
        retry_normals, retry_iterations, retry_residuals = (
            kernel.normals_sum_newton_warm(
                point,
                robot_shape,
                centers[retry],
                shapes[retry],
                uncertainty[retry],
                np.zeros((int(np.count_nonzero(retry)), 3), dtype=float),
                max_iterations=64,
            )
        )
        normals[retry] = retry_normals
        newton[retry] += retry_iterations
        residuals[retry] = retry_residuals
    if float(np.max(residuals)) > 1.0e-7:
        raise RuntimeError("support-sum opening solve exceeded residual gate")

    q_support = np.sqrt(
        np.maximum(np.einsum("ni,nij,nj->n", normals, shapes, normals), 0.0)
    )
    u_support = np.sqrt(
        np.maximum(
            np.einsum("ni,nij,nj->n", normals, uncertainty, normals), 0.0
        )
    )
    distances = (
        np.einsum("ni,ni->n", normals, center_lines)
        - robot_radius
        - q_support
        - u_support
        - offsets
        - SAFETY_MARGIN_M
    )
    limiting = int(np.argmin(distances))
    clearance = float(distances[limiting])
    qn = shapes[limiting] @ normals[limiting]
    un = uncertainty[limiting] @ normals[limiting]
    limiting_surface = (
        centers[limiting]
        - qn / max(float(q_support[limiting]), 1.0e-15)
        - un / max(float(u_support[limiting]), 1.0e-15)
        - offsets[limiting] * normals[limiting]
    )
    boundary_distance = float(
        np.min(
            np.r_[
                point[1:3] - lower,
                upper - point[1:3],
            ]
        )
    )
    open_radius = min(clearance, boundary_distance)
    return {
        "test_point_m": point.tolist(),
        "exact_minimum_clearance_m": clearance,
        "opening_boundary_distance_m": boundary_distance,
        "strict_open_ball_radius_m": float(max(0.0, open_radius)),
        "has_strict_open_ball": bool(open_radius > 0.0),
        "limiting_proxy_index": limiting,
        "limiting_proxy_id": int(snapshot["proxy_ids"][limiting]),
        "closest_point_solver": (
            "same_native_safeguarded_support_sum_newton_warm_then_cold_retry_"
            "as_online_liuqp"
        ),
        "newton_iterations_total": int(np.sum(newton)),
        "cold_retry_pairs": int(np.count_nonzero(retry)),
        "maximum_kkt_residual": float(np.max(residuals)),
        "limiting_surface_point_m": limiting_surface.tolist(),
        "limiting_normal": normals[limiting].tolist(),
        "geometry": "E(Q) minkowski_sum E(U) minkowski_sum B(offset)",
    }


def certify(
    sphere_run: Path,
    ellipsoid_run: Path,
    output: Path,
    grid_size: int,
) -> dict:
    sphere_summary_path = _single_file(sphere_run, "summary.json")
    ellipsoid_summary_path = _single_file(ellipsoid_run, "summary.json")
    sphere_snapshot_path = _single_file(
        sphere_run, "final_causal_proxies.npz"
    )
    ellipsoid_snapshot_path = _single_file(
        ellipsoid_run, "final_causal_proxies.npz"
    )
    sphere_summary = json.loads(
        sphere_summary_path.read_text(encoding="utf-8")
    )
    ellipsoid_summary = json.loads(
        ellipsoid_summary_path.read_text(encoding="utf-8")
    )
    if sphere_summary["representation"] != "sphere":
        raise RuntimeError("sphere run has the wrong representation")
    if ellipsoid_summary["representation"] != "ellipsoid":
        raise RuntimeError("ellipsoid run has the wrong representation")
    for key in ("scene_version", "centervox_filter_size_m"):
        if sphere_summary[key] != ellipsoid_summary[key]:
            raise RuntimeError(f"paired summaries disagree on {key}")
    scene = _protocol_scene(sphere_summary["scene_version"])
    mandatory = _mandatory_attachment_sphere(scene)
    lower, upper = _opening(scene)
    sphere_snapshot = np.load(sphere_snapshot_path)
    ellipsoid_snapshot = np.load(ellipsoid_snapshot_path)
    sphere_section = _sphere_section(
        sphere_snapshot,
        mandatory["front_x_m"],
        lower,
        upper,
        mandatory["radius_m"],
        grid_size,
    )
    center_point = np.array(
        [
            mandatory["front_x_m"],
            0.5 * (lower[0] + upper[0]),
            0.5 * (lower[1] + upper[1]),
        ],
        dtype=float,
    )
    successful_run_ellipsoid_open = _ellipsoid_open_point(
        ellipsoid_snapshot,
        center_point,
        lower,
        upper,
        mandatory["radius_m"],
    )
    causal_gates = {
        "sphere_observability_passed": bool(
            sphere_summary["observability_passed"]
        ),
        "ellipsoid_observability_passed": bool(
            ellipsoid_summary["observability_passed"]
        ),
        "sphere_global_map_preloaded": bool(
            sphere_summary["global_map_preloaded"]
        ),
        "ellipsoid_global_map_preloaded": bool(
            ellipsoid_summary["global_map_preloaded"]
        ),
        "sphere_future_frames_used": bool(
            sphere_summary["future_frames_used"]
        ),
        "ellipsoid_future_frames_used": bool(
            ellipsoid_summary["future_frames_used"]
        ),
        "sphere_truth_feedback_to_control": bool(
            sphere_summary["truth_observability_feedback_to_control"]
        ),
        "ellipsoid_truth_feedback_to_control": bool(
            ellipsoid_summary["truth_observability_feedback_to_control"]
        ),
    }
    causal_pass = bool(
        causal_gates["sphere_observability_passed"]
        and causal_gates["ellipsoid_observability_passed"]
        and not causal_gates["sphere_global_map_preloaded"]
        and not causal_gates["ellipsoid_global_map_preloaded"]
        and not causal_gates["sphere_future_frames_used"]
        and not causal_gates["ellipsoid_future_frames_used"]
        and not causal_gates["sphere_truth_feedback_to_control"]
        and not causal_gates["ellipsoid_truth_feedback_to_control"]
    )
    result = {
        "certificate": "formal_paired_causal_cross_section_v2_support_sum",
        "topological_argument": (
            "the selected attachment-body robot sphere starts before the "
            "drawer front and, because its fixed body-frame offset from the "
            "end-effector is bounded, must finish beyond the front at the "
            "task target; continuity therefore forces its center through the "
            "entire certified front rectangle"
        ),
        "safety_margin_m": SAFETY_MARGIN_M,
        "physical_opening_lower_yz_m": lower.tolist(),
        "physical_opening_upper_yz_m": upper.tolist(),
        "mandatory_robot_sphere": mandatory,
        "sphere_snapshot": str(sphere_snapshot_path.resolve()),
        "sphere_snapshot_sha256": _sha256(sphere_snapshot_path),
        "ellipsoid_snapshot": str(ellipsoid_snapshot_path.resolve()),
        "ellipsoid_snapshot_sha256": _sha256(ellipsoid_snapshot_path),
        "sphere_section": sphere_section,
        "same_sphere_run_snapshot_matched_ellipsoid_section": None,
        "same_snapshot_note": (
            "the formal cap-sphere snapshot intentionally stores only its "
            "independent bucket-local ball family; same-point-cloud crossed "
            "ellipsoid evidence is certified by the crossed-snapshot audit"
        ),
        "successful_ellipsoid_run_section": successful_run_ellipsoid_open,
        "causal_gates": causal_gates,
        "causal_gates_passed": causal_pass,
        "paired_certificate_passed": bool(
            sphere_section["strictly_closed"]
            and successful_run_ellipsoid_open["has_strict_open_ball"]
            and causal_pass
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "certificate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sphere_run", type=Path)
    parser.add_argument("ellipsoid_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grid-size", type=int, default=301)
    args = parser.parse_args()
    if args.grid_size < 3:
        raise ValueError("grid size must be at least 3")
    result = certify(
        args.sphere_run,
        args.ellipsoid_run,
        args.output,
        args.grid_size,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
