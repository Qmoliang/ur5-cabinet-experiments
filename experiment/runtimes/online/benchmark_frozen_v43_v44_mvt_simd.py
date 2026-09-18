"""Offline MVT/AABB/SIMD ablation on the frozen v4.3/v4.4 workloads.

This script never changes the frozen runs.  It replays their immutable proxy
snapshots and joint configurations through the *current* native broad-phase
kernel, checks every result against a float32 O(N) oracle, and stores timing
samples in a separate supplement directory.  Because the historical native
source/binary was not archived, the report explicitly records hash matches and
must not be described as a timing rerun of the historical controller binary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import time

import mujoco
import numpy as np

from model import build_robot_certificate, certificate_world_state, set_configuration
from native_mvt import BruteForceAABBIndex, NativeMultilevelMVT


ROOT = Path(__file__).resolve().parent
QUERY_PADDING_M = 0.040 + 0.006
BASE_VOXEL_SIZE_M = 0.015


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def timing_stats(samples_ns: list[int]) -> dict[str, float]:
    values = np.asarray(samples_ns, dtype=float) / 1.0e6
    return {
        "sample_count": int(len(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def time_query(index, positions: np.ndarray, radii: np.ndarray, repetitions: int) -> list[int]:
    for _ in range(12):
        index.query_spheres(positions, radii)
    samples: list[int] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        index.query_spheres(positions, radii)
        samples.append(time.perf_counter_ns() - started)
    return samples


def rows_equal(left, right) -> bool:
    return len(left) == len(right) and all(
        np.array_equal(a, b) for a, b in zip(left, right)
    )


def analyse_run(run_dir: Path, repetitions: int) -> dict:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    archive = np.load(run_dir / "causal_proxy_snapshots.npz")
    q_history = np.asarray(np.load(run_dir / "q_history.npy"), dtype=float)
    model = mujoco.MjModel.from_xml_path(str(run_dir / "scene.xml"))
    data = mujoco.MjData(model)
    certificate = build_robot_certificate(model)
    robot_radii = np.asarray([item.radius for item in certificate], dtype=float)
    if len(robot_radii) != int(summary["robot_certificate_spheres"]):
        raise AssertionError("current robot certificate count differs from frozen run")
    maximum_query_half = float(np.max(robot_radii) + QUERY_PADDING_M)

    offsets = np.asarray(archive["offsets"], dtype=np.int64)
    publish_cycles = np.asarray(archive["publish_cycles"], dtype=np.int64)
    all_samples = {"full_scan": [], "mvt_scalar": [], "mvt_avx2": []}
    candidate_counts: list[int] = []
    generation_rows: list[dict] = []
    oracle_checks = 0

    for generation, publish_cycle in enumerate(publish_cycles):
        start, stop = map(int, offsets[generation : generation + 2])
        centers = np.asarray(archive["centers"][start:stop], dtype=float)
        outer = np.asarray(
            archive["ellipsoid_outer_shapes"][start:stop], dtype=float
        )
        proxy_offsets = np.asarray(
            archive["uncertainty_offsets"][start:stop], dtype=float
        )
        half = np.sqrt(
            np.maximum(np.diagonal(outer, axis1=1, axis2=2), 0.0)
        ) + proxy_offsets[:, None]
        cycle = min(int(publish_cycle), len(q_history) - 1)
        set_configuration(model, data, q_history[cycle])
        positions, _, _ = certificate_world_state(model, data, certificate)

        full = BruteForceAABBIndex(
            centers, half, query_padding=QUERY_PADDING_M
        )
        scalar = NativeMultilevelMVT(
            centers,
            half,
            base_voxel_size=BASE_VOXEL_SIZE_M,
            maximum_query_half_extent=maximum_query_half,
            query_padding=QUERY_PADDING_M,
            simd=False,
        )
        avx2 = NativeMultilevelMVT(
            centers,
            half,
            base_voxel_size=BASE_VOXEL_SIZE_M,
            maximum_query_half_extent=maximum_query_half,
            query_padding=QUERY_PADDING_M,
            simd=True,
        )
        try:
            expected = full.query_spheres(positions, robot_radii)
            scalar_rows = scalar.query_spheres(positions, robot_radii)
            avx2_rows = avx2.query_spheres(positions, robot_radii)
            if not rows_equal(expected, scalar_rows):
                raise AssertionError(
                    f"generation {generation}: scalar MVT differs from full scan"
                )
            if not rows_equal(expected, avx2_rows):
                raise AssertionError(
                    f"generation {generation}: AVX2 MVT differs from full scan"
                )
            oracle_checks += len(expected) * 2
            candidate_counts.extend(len(row) for row in expected)

            local = {
                "full_scan": time_query(full, positions, robot_radii, repetitions),
                "mvt_scalar": time_query(
                    scalar, positions, robot_radii, repetitions
                ),
                "mvt_avx2": time_query(avx2, positions, robot_radii, repetitions),
            }
            for name, samples in local.items():
                all_samples[name].extend(samples)
            local_stats = {name: timing_stats(values) for name, values in local.items()}
            generation_rows.append(
                {
                    "generation": int(generation),
                    "publish_cycle": int(publish_cycle),
                    "proxy_count": int(len(centers)),
                    "candidate_pairs": int(sum(len(row) for row in expected)),
                    "candidate_fraction_of_all_pairs": float(
                        sum(len(row) for row in expected)
                        / max(1, len(centers) * len(robot_radii))
                    ),
                    "mvt_levels": int(scalar.stats.level_count),
                    "cell_lookups_per_robot_query": int(
                        scalar.stats.cell_lookups_per_query
                    ),
                    "timing": local_stats,
                }
            )
        finally:
            scalar.close()
            avx2.close()

    aggregate = {name: timing_stats(values) for name, values in all_samples.items()}
    aggregate["speedup_full_scan_over_mvt_scalar_median"] = float(
        aggregate["full_scan"]["median_ms"]
        / aggregate["mvt_scalar"]["median_ms"]
    )
    aggregate["speedup_full_scan_over_mvt_avx2_median"] = float(
        aggregate["full_scan"]["median_ms"]
        / aggregate["mvt_avx2"]["median_ms"]
    )
    aggregate["speedup_mvt_scalar_over_avx2_median"] = float(
        aggregate["mvt_scalar"]["median_ms"]
        / aggregate["mvt_avx2"]["median_ms"]
    )

    current_files = {
        "model.py": ROOT / "model.py",
        "native_mvt.py": ROOT / "native_mvt.py",
        "native_mvt/native_mvt.cpp": ROOT / "native_mvt" / "native_mvt.cpp",
        "native_mvt/native_mvt.dll": ROOT / "native_mvt" / "native_mvt.dll",
    }
    current_hashes = {name: sha256(path) for name, path in current_files.items()}
    recorded_hashes = summary.get("source_hashes", {})
    hash_matches = {
        name: bool(recorded_hashes.get(name) == value)
        for name, value in current_hashes.items()
    }
    return {
        "analysis": "offline_current_kernel_on_frozen_workload",
        "historical_controller_timing_rerun": False,
        "run_dir": str(run_dir.resolve()),
        "summary_sha256": sha256(summary_path),
        "representation": summary["representation"],
        "proxy_generations": int(len(publish_cycles)),
        "robot_query_count_per_batch": int(len(robot_radii)),
        "repetitions_per_generation": int(repetitions),
        "oracle_comparisons": int(oracle_checks),
        "all_scalar_and_avx2_rows_equal_full_scan": True,
        "candidate_count_per_robot_query": {
            "median": float(np.median(candidate_counts)),
            "p95": float(np.percentile(candidate_counts, 95)),
            "p99": float(np.percentile(candidate_counts, 99)),
            "maximum": int(np.max(candidate_counts)),
        },
        "aggregate_timing": aggregate,
        "generation_rows": generation_rows,
        "implementation": {
            "full_scan": "float32 NumPy O(N) sphere--AABB oracle",
            "mvt": "five-level size-selected table; 27 cells per level",
            "simd": "AVX2, eight float lanes; broad phase only",
            "base_voxel_size_m": BASE_VOXEL_SIZE_M,
            "query_padding_m": QUERY_PADDING_M,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "current_hashes": current_hashes,
            "frozen_recorded_hashes": {
                name: recorded_hashes.get(name) for name in current_hashes
            },
            "current_hash_matches_frozen": hash_matches,
            "all_relevant_hashes_match_frozen": bool(all(hash_matches.values())),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    result = analyse_run(args.run_dir.resolve(), args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
