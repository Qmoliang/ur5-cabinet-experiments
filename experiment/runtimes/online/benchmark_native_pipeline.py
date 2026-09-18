"""Reproducible MVT/AVX2 and ellipsoid narrow-stage ablation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from ellipsoid_model import build_robot_ellipsoid_certificate, ellipsoid_world_state
from ellipsoid_qp_controller import EllipsoidLiuQPController
from incremental_drawer_scene import incremental_drawer_v2_candidate_scene
from model import build_model, set_configuration
from multilevel_voxel_table import MultilevelVoxelTable
from native_mvt import NativeMVT


ROOT = Path(__file__).resolve().parent


def _timed(function, repetitions: int) -> tuple[float, float, float]:
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    values = np.asarray(samples)
    return float(np.median(values)), float(np.percentile(values, 95)), float(np.percentile(values, 99))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "results_incremental_v2" / "pursuit_smoke" / "incremental_drawer_candidate_v2_ellipsoid_causal_online",
    )
    parser.add_argument("--query-repetitions", type=int, default=250)
    parser.add_argument("--controller-repetitions", type=int, default=25)
    args = parser.parse_args()

    snapshot = np.load(args.run_dir / "final_causal_map_and_proxies.npz")
    trajectory = np.load(args.run_dir / "trajectory.npz")
    centers = np.asarray(snapshot["centers"], dtype=float)
    shapes = np.asarray(snapshot["ellipsoid_shapes"], dtype=float)
    offsets = np.asarray(snapshot["proxy_offsets"], dtype=float)
    obstacle_half = np.sqrt(np.maximum(np.diagonal(shapes, axis1=1, axis2=2), 0.0)) + offsets[:, None]

    scene = incremental_drawer_v2_candidate_scene()
    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, trajectory["q"][-1])
    robot = build_robot_ellipsoid_certificate(model)
    robot_centers, _, _, robot_shapes = ellipsoid_world_state(model, data, robot)
    query_half = np.sqrt(np.maximum(np.diagonal(robot_shapes, axis1=1, axis2=2), 0.0)) + 0.16

    python_table = MultilevelVoxelTable.from_ellipsoids(
        centers, shapes, voxel_size=0.065, query_padding=0.16, offset_radii=offsets
    )
    scalar = NativeMVT(centers, obstacle_half, 0.065, query_padding=0.16, simd=False)
    avx2 = NativeMVT(centers, obstacle_half, 0.065, query_padding=0.16, simd=True)

    def naive_queries():
        return [
            np.flatnonzero(
                np.all(centers - obstacle_half <= center + half, axis=1)
                & np.all(centers + obstacle_half >= center - half, axis=1)
            )
            for center, half in zip(robot_centers, query_half)
        ]

    def python_queries():
        return [python_table.query_ellipsoid(center, shape) for center, shape in zip(robot_centers, robot_shapes)]

    def scalar_queries():
        return [scalar.query_aabb(center, half) for center, half in zip(robot_centers, query_half)]

    def avx2_queries():
        return [avx2.query_aabb(center, half) for center, half in zip(robot_centers, query_half)]

    expected = naive_queries()
    for name, values in (("cpp_scalar", scalar_queries()), ("cpp_avx2", avx2_queries())):
        for reference, actual in zip(expected, values):
            np.testing.assert_array_equal(actual, reference, err_msg=name)
    # Python MVT is checked as a conservative candidate superset because its
    # cell-level query intentionally does not perform the final proxy AABB pass.
    for reference, actual in zip(expected, python_queries()):
        if not set(reference).issubset(set(actual)):
            raise AssertionError("Python MVT dropped an oracle candidate")

    query_benchmarks = {}
    for name, function in (
        ("naive_full_aabb", naive_queries),
        ("python_mvt", python_queries),
        ("cpp_scalar_mvt", scalar_queries),
        ("cpp_avx2_mvt", avx2_queries),
    ):
        median, p95, p99 = _timed(function, args.query_repetitions)
        query_benchmarks[name] = {"median_ms": median, "p95_ms": p95, "p99_ms": p99}

    def make_controller(native: bool):
        local_model = build_model(scene)
        local_data = mujoco.MjData(local_model)
        set_configuration(local_model, local_data, trajectory["q"][-1])
        local_robot = build_robot_ellipsoid_certificate(local_model)
        controller = EllipsoidLiuQPController(
            local_model,
            local_data,
            scene,
            local_robot,
            centers,
            shapes,
            obstacle_offsets=offsets,
            safety_margin=0.006,
            obstacle_index=avx2,
            native_support_batch=native,
        )
        return controller

    target = np.asarray(trajectory["target"], dtype=float)
    native_controller = make_controller(True)
    python_controller = make_controller(False)
    native_velocity, _ = native_controller.solve(target)
    python_velocity, _ = python_controller.solve(target)
    np.testing.assert_allclose(native_velocity, python_velocity, rtol=2.0e-5, atol=2.0e-6)

    controller_benchmarks = {}
    for name, controller in (("python_support", python_controller), ("cpp_batch_support", native_controller)):
        def solve():
            controller.previous_velocity[:] = 0.0
            controller.solve(target)
        median, p95, p99 = _timed(solve, args.controller_repetitions)
        _, metrics = controller.solve(target)
        controller_benchmarks[name] = {
            "median_ms": median,
            "p95_ms": p95,
            "p99_ms": p99,
            "metrics": metrics.as_dict(),
        }

    result = {
        "run_dir": str(args.run_dir.resolve()),
        "proxy_count": len(centers),
        "robot_ellipsoid_count": len(robot),
        "query_batch_count": len(robot_centers),
        "native_mvt": asdict(avx2.stats),
        "cpp_scalar_and_avx2_exactly_match_naive_aabb": True,
        "python_mvt_is_conservative_superset": True,
        "native_and_python_qdot_allclose": True,
        "query_benchmarks": query_benchmarks,
        "controller_benchmarks": controller_benchmarks,
    }
    output = args.run_dir / "native_pipeline_benchmark.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    scalar.close()
    avx2.close()


if __name__ == "__main__":
    main()
