"""Rebuild one saved sphere QP and benchmark solver-only settings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import osqp
from scipy import sparse
from scipy.optimize import linprog

from model import build_model, build_robot_certificate, set_configuration
from protocol_drawer_scene import (
    CONTACT_DISTANCE,
    NEAR_DISTANCE,
    SAFETY_MARGIN,
)
from protocol_liuqp_controller import ProtocolLiuQPController
from run_protocol_v3_async_online import _protocol_scene
from run_protocol_v3_online_ablation import _build_mvt_only


def _solve_osqp(P, q, A, lo, hi, **settings):
    solver = osqp.OSQP()
    started = time.perf_counter()
    solver.setup(
        P=sparse.triu(sparse.csc_matrix(P), format="csc"),
        q=q,
        A=sparse.csc_matrix(A),
        l=lo,
        u=hi,
        verbose=False,
        polishing=False,
        warm_starting=True,
        eps_abs=2.0e-5,
        eps_rel=2.0e-5,
        max_iter=6000,
        **settings,
    )
    result = solver.solve(raise_error=False)
    elapsed = 1000.0 * (time.perf_counter() - started)
    return {
        "status": result.info.status.lower(),
        "iterations": int(result.info.iter),
        "run_time_ms": 1000.0 * float(result.info.run_time),
        "wall_ms": elapsed,
        "primal_residual": float(result.info.prim_res),
        "dual_residual": float(result.info.dual_res),
        "rho_estimate": float(result.info.rho_estimate),
        "objective": float(result.info.obj_val),
        "x": None if result.x is None else np.asarray(result.x).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--representation",
        choices=("sphere", "ellipsoid"),
        default="sphere",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with np.load(run_dir / "final_causal_proxies.npz") as saved:
        proxies = SimpleNamespace(
            proxy_ids=np.asarray(saved["proxy_ids"], dtype=np.int64),
            centers=np.asarray(saved["centers"], dtype=float),
            sphere_radii=np.asarray(saved["sphere_radii"], dtype=float),
            proxy_offset_radii=np.asarray(saved["uncertainty_offsets"], dtype=float),
            proxy_uncertainty_shapes=None,
            base_ellipsoid_shapes=np.asarray(saved["ellipsoid_shapes"], dtype=float),
            ellipsoid_outer_shapes=np.asarray(
                saved["ellipsoid_outer_shapes"], dtype=float
            ),
        )
    q_history = np.load(run_dir / "q_history.npy")
    scene = _protocol_scene("camera_quarter")
    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(q_history[-1], dtype=float))
    robot = build_robot_certificate(model)
    robot_radii = np.asarray([item.radius for item in robot], dtype=float)
    table = _build_mvt_only(
        proxies, args.representation, robot_radii, simd=True
    )
    try:
        controller_kwargs = dict(
            model=model,
            data=data,
            scene=scene,
            robot_spheres=robot,
            obstacle_centers=proxies.centers,
            representation=args.representation,
            obstacle_offsets=proxies.proxy_offset_radii,
            proxy_ids=proxies.proxy_ids,
            safety_margin=SAFETY_MARGIN,
            near_distance=NEAR_DISTANCE,
            contact_distance=CONTACT_DISTANCE,
            obstacle_index=table,
            ellipsoid_pair_threads=6,
        )
        if args.representation == "sphere":
            controller_kwargs["obstacle_radii"] = proxies.sphere_radii
        else:
            controller_kwargs["obstacle_shapes"] = (
                proxies.ellipsoid_outer_shapes
            )
        controller = ProtocolLiuQPController(**controller_kwargs)
        _, metrics = controller.solve(np.asarray(scene.waypoints[-1], dtype=float))
        P, q, A, lo, hi = controller.last_qp_arrays
    finally:
        table.close()

    finite_lo = np.isfinite(lo)
    finite_hi = np.isfinite(hi)
    A_ub = np.concatenate((-A[finite_lo], A[finite_hi]), axis=0)
    b_ub = np.concatenate((-lo[finite_lo], hi[finite_hi]), axis=0)
    started = time.perf_counter()
    feasibility = linprog(
        np.zeros(A.shape[1]),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * A.shape[1],
        method="highs",
    )
    linprog_ms = 1000.0 * (time.perf_counter() - started)

    configurations = [
        ("default", {}),
        ("term1", {"check_termination": 1}),
        ("term5", {"check_termination": 5}),
        ("rho_1e-3_fixed", {"rho": 1.0e-3, "adaptive_rho": False}),
        ("rho_1e-2_fixed", {"rho": 1.0e-2, "adaptive_rho": False}),
        ("rho_1e-2_adaptive", {"rho": 1.0e-2, "adaptive_rho": True}),
        (
            "rho_1e-2_adaptive_i25",
            {
                "rho": 1.0e-2,
                "adaptive_rho": True,
                "adaptive_rho_interval": 25,
            },
        ),
        ("rho_1_fixed", {"rho": 1.0, "adaptive_rho": False}),
        ("rho_10_fixed", {"rho": 10.0, "adaptive_rho": False}),
        ("rho_100_fixed", {"rho": 100.0, "adaptive_rho": False}),
        ("adaptive_interval_10", {"adaptive_rho_interval": 10}),
        ("adaptive_interval_25", {"adaptive_rho_interval": 25}),
        ("adaptive_interval_50", {"adaptive_rho_interval": 50}),
        ("scaling_20", {"scaling": 20}),
        ("scaling_50", {"scaling": 50}),
        ("alpha_1", {"alpha": 1.0}),
        ("alpha_18", {"alpha": 1.8}),
    ]
    results = {
        "source_run": str(run_dir),
        "representation": args.representation,
        "source_cycle": int(len(q_history) - 1),
        "qp_row_sha256": metrics.qp_row_sha256,
        "rows": int(len(A)),
        "source_status": metrics.status,
        "source_solve_ms": metrics.solve_ms,
        "linprog_feasibility": {
            "success": bool(feasibility.success),
            "status": int(feasibility.status),
            "message": feasibility.message,
            "wall_ms": linprog_ms,
        },
        "settings": {
            name: _solve_osqp(P, q, A, lo, hi, **settings)
            for name, settings in configurations
        },
    }
    output = run_dir / f"{args.representation}_qp_settings_benchmark.json"
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
