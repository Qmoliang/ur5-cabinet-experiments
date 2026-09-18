"""Benchmark the final multilevel MVT scalar and AVX2 query kernels.

The primary benchmark replays every causal proxy generation from the final
ellipsoid run at its recorded publication configuration.  A separately named
CenterVox stress benchmark applies the same kernel to the final pre-aggregation
observed representatives; it is not reported as controller timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from model import build_robot_certificate, certificate_world_state, set_configuration
from native_mvt import NativeMultilevelMVT
from protocol_drawer_scene import NEAR_DISTANCE, SAFETY_MARGIN
from run_protocol_v3_online_ablation import MVT_BASE_VOXEL_SIZE


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = (
    ROOT
    / "formal_results"
    / "final_two_camera"
    / "formal_evidence_final_v5"
    / (
        "A-CV-formal-mvt_simd-ellipsoid-cv7.5mm-ui1.25-"
        "cam320x180s1-r70mm-wrist_forearm-observed_only-paced-"
        "postsweepaudit"
    )
)
DEFAULT_OUTPUT = (
    ROOT
    / "formal_results"
    / "final_two_camera"
    / "final_mvt_simd_benchmark_v5.json"
)
QUERY_PADDING = NEAR_DISTANCE + SAFETY_MARGIN


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stats(samples_ns: list[int]) -> dict[str, float]:
    values = np.asarray(samples_ns, dtype=float) / 1.0e6
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def _time_batch(table, positions, radii, repetitions: int) -> list[int]:
    for _ in range(12):
        table.query_spheres(positions, radii)
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        table.query_spheres(positions, radii)
        samples.append(time.perf_counter_ns() - started)
    return samples


def _verify_queries(table, centers, half, positions, radii) -> tuple[int, ...]:
    rows = table.query_spheres(positions, radii)
    counts = []
    for position, radius, actual in zip(positions, radii, rows):
        query_radius = float(radius) + QUERY_PADDING
        lo = centers - half
        hi = centers + half
        delta = np.maximum(np.maximum(lo - position, position - hi), 0.0)
        expected = np.flatnonzero(
            np.einsum("ij,ij->i", delta, delta)
            <= query_radius * query_radius
        )
        np.testing.assert_array_equal(actual, expected)
        counts.append(len(actual))
    return tuple(counts)


def _tables(centers, half, maximum_query_half):
    scalar = NativeMultilevelMVT(
        centers,
        half,
        base_voxel_size=MVT_BASE_VOXEL_SIZE,
        maximum_query_half_extent=maximum_query_half,
        query_padding=QUERY_PADDING,
        simd=False,
    )
    avx2 = NativeMultilevelMVT(
        centers,
        half,
        base_voxel_size=MVT_BASE_VOXEL_SIZE,
        maximum_query_half_extent=maximum_query_half,
        query_padding=QUERY_PADDING,
        simd=True,
    )
    return scalar, avx2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=300)
    parser.add_argument("--centervox-repetitions", type=int, default=160)
    args = parser.parse_args()
    if args.repetitions <= 0 or args.centervox_repetitions <= 0:
        parser.error("repetition counts must be positive")

    q_history = np.asarray(np.load(args.run_dir / "q_history.npy"), dtype=float)
    proxy_archive = np.load(args.run_dir / "causal_proxy_snapshots.npz")
    point_archive = np.load(args.run_dir / "causal_observability_snapshots.npz")
    model = mujoco.MjModel.from_xml_path(str(args.run_dir / "scene.xml"))
    data = mujoco.MjData(model)
    robot = build_robot_certificate(model)
    robot_radii = np.asarray([item.radius for item in robot], dtype=float)
    maximum_query_half = float(np.max(robot_radii) + QUERY_PADDING)

    scalar_samples: list[int] = []
    avx2_samples: list[int] = []
    candidate_counts: list[int] = []
    generation_rows = []
    offsets = np.asarray(proxy_archive["offsets"], dtype=np.int64)
    publish_cycles = np.asarray(proxy_archive["publish_cycles"], dtype=np.int64)
    for generation, publish_cycle in enumerate(publish_cycles):
        start, stop = map(int, offsets[generation : generation + 2])
        centers = np.asarray(proxy_archive["centers"][start:stop], dtype=float)
        outer = np.asarray(
            proxy_archive["ellipsoid_outer_shapes"][start:stop], dtype=float
        )
        proxy_offsets = np.asarray(
            proxy_archive["uncertainty_offsets"][start:stop], dtype=float
        )
        half = (
            np.sqrt(
                np.maximum(np.diagonal(outer, axis1=1, axis2=2), 0.0)
            )
            + proxy_offsets[:, None]
        )
        cycle = min(int(publish_cycle), len(q_history) - 1)
        set_configuration(model, data, q_history[cycle])
        positions, _, _ = certificate_world_state(model, data, robot)
        scalar, avx2 = _tables(centers, half, maximum_query_half)
        try:
            scalar_counts = _verify_queries(
                scalar, centers, half, positions, robot_radii
            )
            avx_counts = _verify_queries(
                avx2, centers, half, positions, robot_radii
            )
            if scalar_counts != avx_counts:
                raise AssertionError("scalar and AVX2 candidate counts differ")
            local_scalar = _time_batch(
                scalar, positions, robot_radii, args.repetitions
            )
            local_avx2 = _time_batch(
                avx2, positions, robot_radii, args.repetitions
            )
            scalar_samples.extend(local_scalar)
            avx2_samples.extend(local_avx2)
            candidate_counts.extend(scalar_counts)
            generation_rows.append(
                {
                    "generation": generation,
                    "publish_cycle": int(publish_cycle),
                    "proxy_count": len(centers),
                    "mvt_levels": scalar.stats.level_count,
                    "cell_lookups_per_robot_query": (
                        scalar.stats.cell_lookups_per_query
                    ),
                    "candidate_pairs": int(sum(scalar_counts)),
                    "scalar_median_ms_per_65_queries": _stats(local_scalar)[
                        "median_ms"
                    ],
                    "avx2_median_ms_per_65_queries": _stats(local_avx2)[
                        "median_ms"
                    ],
                }
            )
        finally:
            scalar.close()
            avx2.close()

    scalar_stats = _stats(scalar_samples)
    avx2_stats = _stats(avx2_samples)
    primary_speedup = scalar_stats["median_ms"] / avx2_stats["median_ms"]

    point_offsets = np.asarray(point_archive["offsets"], dtype=np.int64)
    point_start, point_stop = map(int, point_offsets[-2:])
    centervox = np.asarray(
        point_archive["points"][point_start:point_stop], dtype=float
    )
    cover = np.asarray(
        point_archive["cover_radii"][point_start:point_stop], dtype=float
    )
    centervox_half = np.repeat(cover[:, None], 3, axis=1)
    set_configuration(model, data, q_history[-1])
    final_positions, _, _ = certificate_world_state(model, data, robot)
    scalar, avx2 = _tables(centervox, centervox_half, maximum_query_half)
    try:
        scalar_counts = _verify_queries(
            scalar, centervox, centervox_half, final_positions, robot_radii
        )
        avx_counts = _verify_queries(
            avx2, centervox, centervox_half, final_positions, robot_radii
        )
        if scalar_counts != avx_counts:
            raise AssertionError("CenterVox scalar and AVX2 results differ")
        dense_scalar_samples = _time_batch(
            scalar,
            final_positions,
            robot_radii,
            args.centervox_repetitions,
        )
        dense_avx_samples = _time_batch(
            avx2,
            final_positions,
            robot_radii,
            args.centervox_repetitions,
        )
        dense_scalar = _stats(dense_scalar_samples)
        dense_avx = _stats(dense_avx_samples)
        dense_result = {
            "role": (
                "kernel stress on pre-aggregation causal CenterVox; "
                "not controller timing"
            ),
            "center_voxel_count": len(centervox),
            "query_batch_size": len(robot_radii),
            "candidate_pairs": int(sum(scalar_counts)),
            "scalar": dense_scalar,
            "avx2": dense_avx,
            "median_speedup": (
                dense_scalar["median_ms"] / dense_avx["median_ms"]
            ),
            "scalar_and_avx2_equal_full_scan_oracle": True,
            "mvt_levels": scalar.stats.level_count,
            "cell_lookups_per_robot_query": scalar.stats.cell_lookups_per_query,
        }
    finally:
        scalar.close()
        avx2.close()

    result = {
        "benchmark": "formal_causal_multilevel_mvt_aabb_scalar_vs_avx2",
        "run_dir": str(args.run_dir.resolve()),
        "representation": "ellipsoid",
        "causal_proxy_generations": len(publish_cycles),
        "query_batch_size": len(robot_radii),
        "query_repetitions_per_generation": args.repetitions,
        "primary_controller_proxy_workload": {
            "scalar": scalar_stats,
            "avx2": avx2_stats,
            "median_speedup": primary_speedup,
            "scalar_and_avx2_equal_full_scan_oracle": True,
            "candidate_pairs_per_robot_query_p50": float(
                np.percentile(candidate_counts, 50)
            ),
            "candidate_pairs_per_robot_query_p99": float(
                np.percentile(candidate_counts, 99)
            ),
            "generation_rows": generation_rows,
        },
        "centervox_stress_kernel_only": dense_result,
        "implementation": {
            "simd_width_float_lanes": 8,
            "instruction_set": "AVX2",
            "mvt_levels": 5,
            "cell_lookups_per_query": 135,
            "source_hashes": {
                "benchmark_formal_mvt_simd.py": _sha256(Path(__file__)),
                "native_mvt.py": _sha256(ROOT / "native_mvt.py"),
                "native_mvt/native_mvt.cpp": _sha256(
                    ROOT / "native_mvt" / "native_mvt.cpp"
                ),
                "native_mvt/native_mvt.dll": _sha256(
                    ROOT / "native_mvt" / "native_mvt.dll"
                ),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
