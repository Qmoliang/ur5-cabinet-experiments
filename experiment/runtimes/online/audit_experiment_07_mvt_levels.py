"""Audit effective MVT level use in Experiment 07 smoke snapshots."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiment_07_scenes import experiment_07_birdcage_scene, experiment_07_drawer_scene
from model import build_model, build_robot_certificate
from native_mvt import NativeMultilevelMVT


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "formal_results" / "experiment_07"
TABLE_ROOT = ROOT / "tables" / "experiment_07"
BASE_VOXEL = 0.015
INDEX_PADDING = 5.0e-6
QUERY_PADDING = 0.040 + 0.006 + 5.0e-6


def latest_complete_batch(scene: str) -> Path:
    candidates = sorted((RESULT_ROOT / scene).glob("smoke_*_s_e0_et"))
    candidates = [path for path in candidates if (path / "batch_manifest.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"no complete S/E0/ET smoke batch for {scene}")
    return candidates[-1]


def load_run(batch: Path, group: str):
    summaries = list((batch / group / "run_00").glob("*/summary.json"))
    snapshots = list((batch / group / "run_00").glob("*/final_causal_proxies.npz"))
    if len(summaries) != 1 or len(snapshots) != 1:
        raise RuntimeError(f"unexpected run layout: {batch} {group}")
    return json.loads(summaries[0].read_text(encoding="utf-8")), np.load(snapshots[0]), summaries[0]


def half_extents(snapshot, representation: str) -> np.ndarray:
    offsets = np.asarray(snapshot["uncertainty_offsets"], dtype=float)
    if representation == "sphere":
        radii = np.asarray(snapshot["sphere_radii"], dtype=float) + offsets
        return np.repeat(radii[:, None], 3, axis=1) + INDEX_PADDING
    shape = np.asarray(snapshot["ellipsoid_outer_shapes"], dtype=float)
    return (
        np.sqrt(np.maximum(np.diagonal(shape, axis1=1, axis2=2), 0.0))
        + offsets[:, None]
        + INDEX_PADDING
    )


def table_stats(centers, half, maximum_robot_radius):
    table = NativeMultilevelMVT(
        centers,
        half,
        base_voxel_size=BASE_VOXEL,
        maximum_query_half_extent=float(maximum_robot_radius + QUERY_PADDING),
        query_padding=QUERY_PADDING,
        simd=True,
    )
    try:
        stats = table.stats
        counts = list(stats.level_proxy_counts)
        total = max(sum(counts), 1)
        shares = [count / total for count in counts]
        return {
            "voxel_sizes_m": list(stats.level_voxel_sizes),
            "proxy_counts": counts,
            "proxy_shares": shares,
            "nonempty_levels": sum(count > 0 for count in counts),
            "significant_levels_5pct": sum(share >= 0.05 for share in shares),
            "dominant_level_share": max(shares),
            "occupied_cells": stats.occupied_cells,
            "index_references": stats.index_references,
            "cell_lookups_per_query": stats.cell_lookups_per_query,
        }
    finally:
        table.close()


def main() -> None:
    scenes = {
        "drawer": experiment_07_drawer_scene(),
        "birdcage": experiment_07_birdcage_scene(),
    }
    records = []
    details = []
    for scene_name, scene in scenes.items():
        batch = latest_complete_batch(scene_name)
        robot_radii = np.asarray(
            [item.radius for item in build_robot_certificate(build_model(scene))],
            dtype=float,
        )
        classes = (
            ("small", robot_radii <= 0.025),
            ("medium", (robot_radii > 0.025) & (robot_radii <= 0.045)),
            ("large", robot_radii > 0.045),
        )
        for group in ("S", "E0", "ET"):
            summary, snapshot, summary_path = load_run(batch, group)
            centers = np.asarray(snapshot["centers"], dtype=float)
            half = half_extents(snapshot, summary["representation"])
            global_stats = table_stats(centers, half, np.max(robot_radii))
            class_stats = {}
            for class_name, mask in classes:
                radii = robot_radii[mask]
                if len(radii):
                    class_stats[class_name] = {
                        "robot_sphere_count": int(len(radii)),
                        "maximum_robot_radius_m": float(np.max(radii)),
                        **table_stats(centers, half, np.max(radii)),
                    }
            effective = bool(
                global_stats["significant_levels_5pct"] >= 2
                and global_stats["dominant_level_share"] <= 0.90
            )
            record = {
                "scene": scene_name,
                "group": group,
                "representation": summary["representation"],
                "proxy_count": len(centers),
                "level_proxy_counts": json.dumps(global_stats["proxy_counts"]),
                "level_proxy_shares": json.dumps(global_stats["proxy_shares"]),
                "significant_levels_5pct": global_stats["significant_levels_5pct"],
                "dominant_level_share": global_stats["dominant_level_share"],
                "index_references": global_stats["index_references"],
                "full_oracle_passed": summary["all_mvt_oracle_checks_passed"],
                "effective_multilevel_passed": effective,
            }
            records.append(record)
            details.append(
                {
                    **record,
                    "summary_path": str(summary_path.relative_to(ROOT)),
                    "global_max_query_rule": global_stats,
                    "radius_class_diagnostic": class_stats,
                }
            )

    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    with (TABLE_ROOT / "T26_mvt_level_smoke.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "experiment": "07",
        "gate": "E7-D5 diagnostic",
        "mock_data": False,
        "criterion": "at least two levels each hold >=5% of proxies and dominant share <=90%",
        "query_padding_m": QUERY_PADDING,
        "passed": all(row["effective_multilevel_passed"] for row in records),
        "records": details,
    }
    (RESULT_ROOT / "mvt_level_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
