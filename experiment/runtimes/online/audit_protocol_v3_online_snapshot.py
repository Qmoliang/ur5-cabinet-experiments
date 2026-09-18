"""Path-free Q+U clearance audit for a protocol-v3 causal proxy snapshot.

The script evaluates already frozen, independently obtained contact-free IK
configurations at the mandatory x=0.39 m section and at the one final target.
It does not search for or publish a path, waypoint, or controller target.
Clearance uses the same exact support sum as ellipsoid LiuQP:

  n'(o-p) - rho_R(n) - rho_Q(n) - rho_U(n) - delta - d_safe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from ellipsoid_model import optimal_support_normal_with_uncertainty
from model import (
    build_model,
    build_robot_certificate,
    certificate_world_state,
    set_configuration,
)
from protocol_drawer_scene import SAFETY_MARGIN, formal_protocol_scene


ROOT = Path(__file__).resolve().parent
STATIC_GATE = (
    ROOT / "formal_results" / "protocol_v6" / "static_geometry_gate"
    / "static_geometry_gate.json"
)


def _configuration_clearance(model, data, robot, arrays, q):
    set_configuration(model, data, np.asarray(q, dtype=float))
    positions, _, radii = certificate_world_state(model, data, robot)
    centers = np.asarray(arrays["centers"], dtype=float)
    shapes = np.asarray(arrays["ellipsoid_shapes"], dtype=float)
    uncertainty = np.asarray(arrays["proxy_uncertainty_shapes"], dtype=float)
    offsets = np.asarray(arrays["uncertainty_offsets"], dtype=float)
    maximum_support = np.sqrt(
        np.maximum(np.linalg.eigvalsh(shapes)[:, -1], 0.0)
    ) + np.sqrt(np.maximum(np.linalg.eigvalsh(uncertainty)[:, -1], 0.0))
    best = np.inf
    limiting = None
    maximum_residual = 0.0
    iterations = 0
    exact_pairs = 0
    for robot_index, (position, radius) in enumerate(zip(positions, radii)):
        delta = centers - position
        norms = np.linalg.norm(delta, axis=1)
        # This is a rigorous lower bound because each directional support is
        # no larger than its maximum semi-axis.  Once it exceeds the current
        # exact best, the remaining pairs cannot be limiting.
        lower_bounds = (
            norms
            - float(radius)
            - maximum_support
            - offsets
            - SAFETY_MARGIN
        )
        for obstacle_index in np.argsort(lower_bounds):
            if float(lower_bounds[obstacle_index]) >= best:
                break
            result = optimal_support_normal_with_uncertainty(
                position,
                centers[obstacle_index],
                shapes[obstacle_index],
                uncertainty[obstacle_index],
                initial_normal=(
                    delta[obstacle_index]
                    / max(float(norms[obstacle_index]), 1.0e-15)
                ),
            )
            exact_pairs += 1
            clearance = (
                result.clearance_without_robot
                - float(radius)
                - float(offsets[obstacle_index])
                - SAFETY_MARGIN
            )
            maximum_residual = max(maximum_residual, float(result.residual))
            iterations += int(result.iterations)
            if clearance < best:
                best = float(clearance)
                proxy_id = int(arrays["proxy_ids"][obstacle_index])
                base_axes = np.sqrt(
                    np.maximum(np.linalg.eigvalsh(shapes[obstacle_index]), 0.0)
                )
                uncertainty_axes = np.sqrt(
                    np.maximum(
                        np.linalg.eigvalsh(uncertainty[obstacle_index]), 0.0
                    )
                )
                member_mask = (
                    np.asarray(arrays["filtered_cluster_indices"])
                    == obstacle_index
                )
                member_points = np.asarray(arrays["filtered_points"])[member_mask]
                member_uncertainty = np.asarray(
                    arrays["filtered_uncertainty_shapes"]
                )[member_mask]
                member_maximum_axes = (
                    np.zeros(3)
                    if not len(member_uncertainty)
                    else np.max(
                        np.sqrt(
                            np.maximum(
                                np.linalg.eigvalsh(member_uncertainty), 0.0
                            )
                        ),
                        axis=0,
                    )
                )
                maximum_member_trace = (
                    0.0
                    if not len(member_uncertainty)
                    else float(
                        np.max(np.trace(member_uncertainty, axis1=1, axis2=2))
                    )
                )
                limiting = {
                    "robot_index": robot_index,
                    "robot_body": robot[robot_index].body_name,
                    "robot_source_geom": model.geom(
                        robot[robot_index].source_geom_id
                    ).name,
                    "robot_center_m": positions[robot_index].tolist(),
                    "robot_radius_m": float(radius),
                    "obstacle_index": int(obstacle_index),
                    "proxy_id": proxy_id,
                    "proxy_center_m": centers[obstacle_index].tolist(),
                    "proxy_base_semi_axes_m": base_axes.tolist(),
                    "proxy_uncertainty_semi_axes_m": uncertainty_axes.tolist(),
                    "proxy_scalar_offset_m": float(offsets[obstacle_index]),
                    "proxy_member_count": int(np.count_nonzero(member_mask)),
                    "proxy_member_aabb_extent_m": (
                        np.zeros(3)
                        if not len(member_points)
                        else np.ptp(member_points, axis=0)
                    ).tolist(),
                    "maximum_member_uncertainty_semi_axes_m": (
                        member_maximum_axes.tolist()
                    ),
                    "uncertainty_union_trace_inflation": (
                        0.0
                        if maximum_member_trace <= 0.0
                        else float(
                            np.trace(uncertainty[obstacle_index])
                            / maximum_member_trace
                        )
                    ),
                    "normal": result.normal.tolist(),
                    "support_kkt_residual": float(result.residual),
                    "support_iterations": int(result.iterations),
                }
    return {
        "minimum_q_plus_u_clearance_m": best,
        "safe": best >= 0.0,
        "limiting_pair": limiting,
        "maximum_support_kkt_residual": maximum_residual,
        "support_iterations": iterations,
        "exact_pairs_after_spectral_lower_bound": exact_pairs,
    }


def audit(snapshot: Path) -> dict:
    arrays = np.load(snapshot)
    gate = json.loads(STATIC_GATE.read_text(encoding="utf-8"))
    scene = formal_protocol_scene()
    model = build_model(scene)
    data = mujoco.MjData(model)
    robot = build_robot_certificate(model)
    crossing_q = gate["ellipsoid_crossing"]["q"]
    goal_q = [row["q"] for row in gate["goal_ik"]]
    return {
        "snapshot": str(snapshot.resolve()),
        "scene": scene.name,
        "evidence_role": "path_free_actual_causal_snapshot_q_plus_u_audit",
        "path_planner": None,
        "intermediate_targets": None,
        "safety_margin_m": SAFETY_MARGIN,
        "proxy_count": len(arrays["centers"]),
        "mandatory_crossing": _configuration_clearance(
            model, data, robot, arrays, crossing_q
        ),
        "goal_configurations": [
            _configuration_clearance(model, data, robot, arrays, q)
            for q in goal_q
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.snapshot)
    output = args.output or args.snapshot.with_name("online_snapshot_q_plus_u_audit.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
