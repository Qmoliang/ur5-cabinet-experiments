"""Validation run: solve the unchanged sphere LiuQP with SLSQP after OSQP stops."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from scipy.optimize import LinearConstraint, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "core"))

from controller import ProtocolLiuQPController
from known_volume import build_volume_cover, coverage_audit
from robot import DT, JOINT_NAMES, attachment_position, build_model, build_robot_certificate, set_configuration
from run import (
    CONTACT_DISTANCE,
    MAXIMUM_CELL_SPAN,
    MVT_BASE_VOXEL_SIZE,
    MVT_OUTWARD_PADDING,
    NEAR_DISTANCE,
    SAFETY_MARGIN,
    SUCCESS_HOLD_CYCLES,
    SUCCESS_TOLERANCE,
    load_scene,
)
from voxel_index import NativeMultilevelMVT


def main() -> None:
    output = ROOT / "results" / "sphere_same_qp_slsqp_validation"
    output.mkdir(parents=True, exist_ok=True)
    scene = load_scene()
    cover = build_volume_cover(scene.boxes, "sphere", MAXIMUM_CELL_SPAN)
    audit = coverage_audit(cover)
    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(scene.q0))
    robot = build_robot_certificate(model)
    robot_radii = np.asarray([item.radius for item in robot])
    padding = NEAR_DISTANCE + SAFETY_MARGIN + MVT_OUTWARD_PADDING
    index = NativeMultilevelMVT(
        cover.centers,
        cover.aabb_half_extents + MVT_OUTWARD_PADDING,
        MVT_BASE_VOXEL_SIZE,
        float(np.max(robot_radii) + padding),
        query_padding=padding,
        simd=True,
    )
    controller = ProtocolLiuQPController(
        model=model,
        data=data,
        scene=scene,
        robot_spheres=robot,
        obstacle_centers=cover.centers,
        representation="sphere",
        obstacle_radii=cover.sphere_radii,
        obstacle_offsets=np.zeros(len(cover.centers)),
        proxy_ids=cover.proxy_ids,
        safety_margin=SAFETY_MARGIN,
        near_distance=NEAR_DISTANCE,
        contact_distance=CONTACT_DISTANCE,
        obstacle_index=index,
        redundant_plane_pruning=True,
    )
    controller.task_gain = 2.0
    controller.max_task_speed = 0.18
    controller.osqp_absolute_tolerance = 1e-4
    controller.osqp_relative_tolerance = 1e-4
    controller.osqp_max_iterations = 1000
    target = np.asarray(scene.waypoints[-1])
    q_history = [data.qpos[: len(JOINT_NAMES)].copy()]
    ee_history = [attachment_position(model, data)]
    rows = []
    fallback_previous = np.zeros(len(JOINT_NAMES))
    fallback_cycles = 0
    fallback_failures = 0
    penetrating_cycles = 0
    maximum_hold = 0
    hold = 0
    started = time.perf_counter()
    try:
        for cycle in range(3000):
            q_before = data.qpos[: len(JOINT_NAMES)].copy()
            qdot, metrics = controller.solve(target)
            fallback_used = False
            fallback_success = False
            if not metrics.status.startswith("solved"):
                fallback_used = True
                fallback_cycles += 1
                hessian, gradient, matrix, lower, upper = controller.last_qp_arrays
                constraint = LinearConstraint(matrix, lower, upper)
                qp = minimize(
                    lambda x: 0.5 * x @ hessian @ x + gradient @ x,
                    fallback_previous,
                    jac=lambda x: hessian @ x + gradient,
                    constraints=[constraint],
                    method="SLSQP",
                    options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
                )
                violation = max(
                    float(np.max(np.where(np.isfinite(lower), lower - matrix @ qp.x, -np.inf))),
                    float(np.max(np.where(np.isfinite(upper), matrix @ qp.x - upper, -np.inf))),
                    0.0,
                )
                fallback_success = bool(qp.success and violation <= 1e-8)
                if fallback_success:
                    qdot = np.asarray(qp.x)
                    controller.previous_velocity = qdot.copy()
                else:
                    qdot = np.zeros(len(JOINT_NAMES))
                    fallback_failures += 1
            fallback_previous = qdot.copy()
            set_configuration(model, data, q_before + qdot * DT)
            ee = attachment_position(model, data)
            error = float(np.linalg.norm(target - ee))
            penetrating = sum(float(data.contact[i].dist) < -1e-8 for i in range(data.ncon))
            penetrating_cycles += int(penetrating > 0)
            if error < SUCCESS_TOLERANCE and penetrating == 0:
                hold += 1
                maximum_hold = max(maximum_hold, hold)
            else:
                hold = 0
            q_history.append(data.qpos[: len(JOINT_NAMES)].copy())
            ee_history.append(ee.copy())
            rows.append(
                {
                    "cycle": cycle,
                    "time_s": cycle * DT,
                    "error_m": error,
                    "osqp_status": metrics.status,
                    "fallback_used": fallback_used,
                    "fallback_success": fallback_success,
                    "executed_qdot_norm": float(np.linalg.norm(qdot)),
                    "penetrating_contacts": penetrating,
                    "limiting_robot_index": metrics.limiting_robot_index,
                    "limiting_proxy_id": metrics.limiting_proxy_id,
                    "minimum_hard_clearance_m": metrics.min_clearance,
                }
            )
    finally:
        index.close()
    write_path = output / "cycles.csv"
    with write_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.save(output / "q_history.npy", np.asarray(q_history))
    np.save(output / "ee_history.npy", np.asarray(ee_history))
    np.savez_compressed(
        output / "proxies.npz",
        centers=cover.centers,
        sphere_radii=cover.sphere_radii,
        ellipsoid_shapes=cover.ellipsoid_shapes,
        cell_half_extents=cover.cell_half_extents,
        box_indices=cover.box_indices,
    )
    summary = {
        "experiment": "sphere_same_qp_robust_solver_validation",
        "purpose": "remove OSQP iteration-limit confounding without changing the QP",
        "path_planner": None,
        "random_dither": False,
        "complete_solid_cell_volume_covered": audit["complete_solid_cell_volume_covered"],
        "cycles": len(rows),
        "fallback_cycles": fallback_cycles,
        "fallback_failures": fallback_failures,
        "final_error_m": rows[-1]["error_m"],
        "minimum_error_m": min(row["error_m"] for row in rows),
        "maximum_success_hold_cycles": maximum_hold,
        "success": maximum_hold >= SUCCESS_HOLD_CYCLES,
        "exact_penetrating_cycles": penetrating_cycles,
        "wall_duration_s": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
