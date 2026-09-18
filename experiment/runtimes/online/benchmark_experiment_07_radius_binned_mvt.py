"""Benchmark the Experiment 07 radius-binned MVT prototype on frozen snapshots."""

from __future__ import annotations

import json
from pathlib import Path
import time

import mujoco
import numpy as np

from experiment_07_scenes import experiment_07_birdcage_scene, experiment_07_drawer_scene
from model import build_model, build_robot_certificate, certificate_world_positions, set_configuration
from native_mvt import BruteForceAABBIndex, NativeMultilevelMVT
from radius_binned_mvt import RadiusBinnedMVT


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "formal_results" / "experiment_07"
BASE_VOXEL = 0.015
INDEX_PADDING = 5.0e-6
QUERY_PADDING = 0.046005
REPEATS = 100


def latest_batch(scene):
    items = sorted((RESULT_ROOT / scene).glob("smoke_*_s_e0_et"))
    return [item for item in items if (item / "batch_manifest.json").exists()][-1]


def load_snapshot(batch, group):
    summary_path = next((batch / group / "run_00").glob("*/summary.json"))
    snapshot_path = next((batch / group / "run_00").glob("*/final_causal_proxies.npz"))
    return json.loads(summary_path.read_text(encoding="utf-8")), np.load(snapshot_path)


def half_extents(snapshot, representation):
    offsets = np.asarray(snapshot["uncertainty_offsets"], dtype=float)
    if representation == "sphere":
        radii = np.asarray(snapshot["sphere_radii"], dtype=float) + offsets
        return np.repeat(radii[:, None], 3, axis=1) + INDEX_PADDING
    shape = np.asarray(snapshot["ellipsoid_outer_shapes"], dtype=float)
    return np.sqrt(np.maximum(np.diagonal(shape, axis1=1, axis2=2), 0.0)) + offsets[:, None] + INDEX_PADDING


def timed(index, positions, radii):
    started = time.perf_counter()
    result = None
    for _ in range(REPEATS):
        result = index.query_spheres(positions, radii)
    elapsed = (time.perf_counter() - started) * 1000.0 / REPEATS
    return elapsed, result


def main():
    scenes = {"drawer": experiment_07_drawer_scene(), "birdcage": experiment_07_birdcage_scene()}
    rows = []
    for scene_name, scene in scenes.items():
        model = build_model(scene)
        data = mujoco.MjData(model)
        set_configuration(model, data, np.asarray(scene.q0, dtype=float))
        robot = build_robot_certificate(model)
        positions = certificate_world_positions(data, robot)
        radii = np.asarray([item.radius for item in robot], dtype=float)
        for group in ("S", "E0", "ET"):
            summary, snapshot = load_snapshot(latest_batch(scene_name), group)
            centers = np.asarray(snapshot["centers"], dtype=float)
            half = half_extents(snapshot, summary["representation"])
            global_index = NativeMultilevelMVT(
                centers, half, BASE_VOXEL, float(np.max(radii) + QUERY_PADDING),
                query_padding=QUERY_PADDING, simd=True,
            )
            binned = RadiusBinnedMVT(
                centers, half, BASE_VOXEL, QUERY_PADDING
            ).build(radii)
            oracle = BruteForceAABBIndex(centers, half, query_padding=QUERY_PADDING)
            try:
                global_ms, global_candidates = timed(global_index, positions, radii)
                binned_ms, binned_candidates = timed(binned, positions, radii)
                oracle_ms, oracle_candidates = timed(oracle, positions, radii)
                global_equal = all(
                    np.array_equal(np.sort(a), np.sort(b))
                    for a, b in zip(global_candidates, oracle_candidates)
                )
                binned_equal = all(
                    np.array_equal(np.sort(a), np.sort(b))
                    for a, b in zip(binned_candidates, oracle_candidates)
                )
                rows.append({
                    "scene": scene_name,
                    "group": group,
                    "proxy_count": len(centers),
                    "query_count": len(radii),
                    "global_level_counts": list(global_index.stats.level_proxy_counts),
                    "binned_class_level_counts": [list(v) for v in binned.stats.class_level_proxy_counts],
                    "global_index_references": global_index.stats.index_references,
                    "binned_index_references": binned.stats.index_references,
                    "global_query_ms_mean": global_ms,
                    "binned_query_ms_mean": binned_ms,
                    "full_aabb_ms_mean": oracle_ms,
                    "global_candidate_total": sum(len(v) for v in global_candidates),
                    "binned_candidate_total": sum(len(v) for v in binned_candidates),
                    "oracle_candidate_total": sum(len(v) for v in oracle_candidates),
                    "global_oracle_equal": global_equal,
                    "binned_oracle_equal": binned_equal,
                    "binned_speedup_over_global": global_ms / binned_ms,
                    "binned_reference_multiplier": binned.stats.index_references / global_index.stats.index_references,
                })
            finally:
                global_index.close()
                binned.close()
    report = {
        "experiment": "07",
        "prototype": "query-radius-binned replicated MVT",
        "mock_data": False,
        "timing_repeats": REPEATS,
        "passed_candidate_equivalence": all(row["binned_oracle_equal"] for row in rows),
        "rows": rows,
    }
    (RESULT_ROOT / "radius_binned_mvt_prototype.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed_candidate_equivalence"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

