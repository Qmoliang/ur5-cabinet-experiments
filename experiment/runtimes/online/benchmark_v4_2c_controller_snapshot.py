"""Benchmark one frozen v4.2c proxy snapshot without advancing MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import psutil

from model import build_robot_certificate, certificate_world_positions, set_configuration
from protocol_drawer_scene import formal_protocol_scene
from run_protocol_v3_online_ablation import _build_mvt_and_audit, _new_controller


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=40)
    parser.add_argument("--affinity-cpus", type=int, default=None)
    parser.add_argument(
        "--lazy-erase-remove",
        action="store_true",
        help="use the scalar exact closest point only after ordered erase-remove",
    )
    parser.add_argument(
        "--full-pair-batch",
        action="store_true",
        help="disable native lazy pruning and solve every broad-phase pair",
    )
    args = parser.parse_args()
    if args.affinity_cpus is not None:
        psutil.Process().cpu_affinity(list(range(args.affinity_cpus)))
    arrays = np.load(args.run_dir / "final_causal_proxies.npz")
    shapes = np.asarray(arrays["ellipsoid_shapes"], dtype=float)
    proxies = SimpleNamespace(
        centers=np.asarray(arrays["centers"], dtype=float),
        sphere_radii=np.asarray(arrays["sphere_radii"], dtype=float),
        base_ellipsoid_shapes=shapes,
        ellipsoid_outer_shapes=np.asarray(
            arrays["ellipsoid_outer_shapes"], dtype=float
        ),
        proxy_uncertainty_shapes=np.asarray(
            arrays["proxy_uncertainty_shapes"], dtype=float
        ),
        proxy_offset_radii=np.asarray(arrays["uncertainty_offsets"], dtype=float),
        proxy_ids=np.asarray(arrays["proxy_ids"], dtype=np.int64),
    )
    scene = formal_protocol_scene()
    model = mujoco.MjModel.from_xml_path(str(args.run_dir / "scene.xml"))
    data = mujoco.MjData(model)
    q = np.asarray(np.load(args.run_dir / "q_history.npy")[-1], dtype=float)
    set_configuration(model, data, q)
    robot = build_robot_certificate(model)
    radii = np.asarray([item.radius for item in robot], dtype=float)
    positions = certificate_world_positions(data, robot)
    table, missing = _build_mvt_and_audit(
        proxies, "ellipsoid", positions, radii, simd=True
    )
    try:
        controller = _new_controller(
            "ellipsoid", model, data, scene, robot, proxies, table
        )
        if args.lazy_erase_remove:
            controller.native_exact_batch = False
            controller.native_exact_pair_batch = False
        if args.full_pair_batch:
            controller.native_exact_lazy_prune = False
        target = np.asarray(scene.waypoints[-1], dtype=float)
        controller.solve(target)
        rows = []
        for _ in range(args.repetitions):
            qdot, metrics = controller.solve(target)
            rows.append(
                {
                    "total_ms": metrics.total_controller_ms,
                    "geometry_ms": metrics.geometry_ms,
                    "candidate_pairs": metrics.broadphase_candidate_pairs,
                    "newton_iterations": metrics.closest_point_newton_iterations,
                    "qp_row_sha256": metrics.qp_row_sha256,
                    "qdot": qdot.tolist(),
                }
            )
    finally:
        table.close()
    total = np.asarray([item["total_ms"] for item in rows])
    geometry = np.asarray([item["geometry_ms"] for item in rows])
    qdots = np.asarray([item["qdot"] for item in rows], dtype=float)
    result = {
        "run_dir": str(args.run_dir.resolve()),
        "repetitions": args.repetitions,
        "lazy_erase_remove": bool(args.lazy_erase_remove),
        "full_pair_batch": bool(args.full_pair_batch),
        "mvt_missing_candidates": missing,
        "proxy_count": len(proxies.centers),
        "candidate_pairs": rows[-1]["candidate_pairs"],
        "newton_iterations_last": rows[-1]["newton_iterations"],
        "total_ms_p50": float(np.percentile(total, 50)),
        "total_ms_p99": float(np.percentile(total, 99)),
        "geometry_ms_p50": float(np.percentile(geometry, 50)),
        "geometry_ms_p99": float(np.percentile(geometry, 99)),
        "qp_rows_identical": len({item["qp_row_sha256"] for item in rows}) == 1,
        "qdot_bitwise_identical": bool(np.all(qdots == qdots[0])),
        "qdot_max_abs_delta": float(np.max(np.abs(qdots - qdots[0]))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
