"""Read-only feasibility and progress audit of the final sphere QP."""
from __future__ import annotations

from pathlib import Path
import sys
import json

import numpy as np
from scipy.optimize import LinearConstraint, linprog, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "core"))

from controller import ProtocolLiuQPController
from known_volume import build_volume_cover
from robot import JOINT_NAMES, build_model, build_robot_certificate, set_configuration
from run import (
    CONTACT_DISTANCE,
    MVT_BASE_VOXEL_SIZE,
    MVT_OUTWARD_PADDING,
    NEAR_DISTANCE,
    SAFETY_MARGIN,
    load_scene,
)
from voxel_index import NativeMultilevelMVT


def main() -> None:
    scene = load_scene()
    cover = build_volume_cover(scene.boxes, "sphere")
    model = build_model(scene)
    data = __import__("mujoco").MjData(model)
    q = np.load(ROOT / "results" / "sphere" / "q_history.npy")[-1]
    set_configuration(model, data, q)
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
    try:
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
        controller.osqp_max_iterations = 6000
        target = np.asarray(scene.waypoints[-1])
        _qdot, metrics = controller.solve(target)
        hessian, gradient, matrix, lower, upper = controller.last_qp_arrays
        upper_mask = np.isfinite(upper)
        lower_mask = np.isfinite(lower)
        aub = np.vstack((matrix[upper_mask], -matrix[lower_mask]))
        bub = np.concatenate((upper[upper_mask], -lower[lower_mask]))
        feasibility = linprog(
            np.zeros(len(JOINT_NAMES)),
            A_ub=aub,
            b_ub=bub,
            bounds=[(None, None)] * len(JOINT_NAMES),
            method="highs",
        )
        ee, jacobian, _desired = controller.task_feedback(target)
        target_direction = target - ee
        target_direction /= np.linalg.norm(target_direction)
        progress_coeff = target_direction @ jacobian
        progress = linprog(
            -progress_coeff,
            A_ub=aub,
            b_ub=bub,
            bounds=[(None, None)] * len(JOINT_NAMES),
            method="highs",
        )
        if feasibility.success:
            constraint = LinearConstraint(matrix, lower, upper)
            qp = minimize(
                lambda x: 0.5 * x @ hessian @ x + gradient @ x,
                feasibility.x,
                jac=lambda x: hessian @ x + gradient,
                constraints=[constraint],
                method="SLSQP",
                options={"maxiter": 2000, "ftol": 1e-12, "disp": False},
            )
        else:
            qp = None
        report = {
                "osqp_status": metrics.status,
                "osqp_primal_residual": metrics.qp_primal_residual,
                "osqp_dual_residual": metrics.qp_dual_residual,
                "linear_constraints_feasible": bool(feasibility.success),
                "feasibility_message": feasibility.message,
                "maximum_instantaneous_target_progress_m_per_s": None
                if not progress.success
                else float(-progress.fun),
                "progress_lp_message": progress.message,
                "slsqp_success": None if qp is None else bool(qp.success),
                "slsqp_message": None if qp is None else qp.message,
                "slsqp_qdot_norm": None if qp is None else float(np.linalg.norm(qp.x)),
                "slsqp_target_progress_m_per_s": None
                if qp is None
                else float(progress_coeff @ qp.x),
                "slsqp_max_constraint_violation": None
                if qp is None
                else float(
                    max(
                        np.max(np.where(np.isfinite(lower), lower - matrix @ qp.x, -np.inf)),
                        np.max(np.where(np.isfinite(upper), matrix @ qp.x - upper, -np.inf)),
                        0.0,
                    )
                ),
            }
        (ROOT / "results" / "sphere_stall_qp_audit.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    finally:
        index.close()


if __name__ == "__main__":
    main()
