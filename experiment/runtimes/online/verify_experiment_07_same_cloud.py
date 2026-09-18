"""Verify the Experiment 07 S/SA/E0/ET30 immutable initial snapshots."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "formal_results" / "experiment_07"
TABLE_ROOT = ROOT / "tables" / "experiment_07"
INPUT_KEYS = ("filtered_points", "filtered_uncertainty_shapes", "filtered_point_offsets")


def latest_complete_batch(scene):
    required = ["S", "SA", "E0", "ET30"]
    paths = sorted((RESULT_ROOT / scene).glob("smoke_*"), reverse=True)
    for path in paths:
        manifest_path = path / "batch_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("groups") == required:
            return path
    raise FileNotFoundError(f"no complete frozen-group batch for {scene}")


def load(batch, group):
    root = batch / group / "run_00"
    summary_path = next(root.glob("*/summary.json"))
    proxy_path = next(root.glob("*/final_causal_proxies.npz"))
    return json.loads(summary_path.read_text(encoding="utf-8")), np.load(proxy_path), proxy_path


def input_hash(snapshot):
    digest = hashlib.sha256()
    for key in INPUT_KEYS:
        value = np.ascontiguousarray(snapshot[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def direct_coverage(snapshot, representation):
    points = np.asarray(snapshot["filtered_points"], dtype=float)
    centers = np.asarray(snapshot["centers"], dtype=float)
    inverse = np.asarray(snapshot["filtered_cluster_indices"], dtype=int)
    if representation == "sphere":
        radii = np.asarray(snapshot["base_sphere_radii"], dtype=float)
        residual = np.linalg.norm(points - centers[inverse], axis=1) - radii[inverse]
        return float(np.max(residual)), bool(np.max(residual) <= 1.0e-9)
    shapes = np.asarray(snapshot["ellipsoid_shapes"], dtype=float)
    values = np.empty(len(points))
    for proxy_index in np.unique(inverse):
        indices = np.flatnonzero(inverse == proxy_index)
        local = points[indices] - centers[proxy_index]
        solved = np.linalg.solve(shapes[proxy_index], local.T).T
        values[indices] = np.sum(local * solved, axis=1)
    residual = float(np.max(values) - 1.0)
    return residual, bool(residual <= 1.0e-8)


def main():
    records = []
    scene_reports = []
    for scene in ("drawer", "birdcage"):
        batch = latest_complete_batch(scene)
        loaded = {group: load(batch, group) for group in ("S", "SA", "E0", "ET30")}
        hashes = {group: input_hash(item[1]) for group, item in loaded.items()}
        reference = loaded["S"][1]
        exact_equal = all(
            all(np.array_equal(reference[key], item[1][key]) for key in INPUT_KEYS)
            for item in loaded.values()
        )
        for group, (summary, snapshot, path) in loaded.items():
            residual, covered = direct_coverage(snapshot, summary["representation"])
            axes = np.sqrt(np.maximum(np.linalg.eigvalsh(snapshot["ellipsoid_shapes"]), 0.0))
            ratios = axes[:, -1] / np.maximum(axes[:, 0], 1.0e-15)
            record = {
                "scene": scene,
                "group": group,
                "input_sha256": hashes[group],
                "same_input_exact": exact_equal,
                "center_voxels": len(snapshot["filtered_points"]),
                "proxy_count": len(snapshot["centers"]),
                "minimum_core_semi_axis_m": (
                    None if group == "S" else float(np.min(axes[:, 0]))
                ),
                "core_axis_ratio_p95": (
                    None if group == "S" else float(np.percentile(ratios, 95))
                ),
                "direct_core_coverage_residual": residual,
                "direct_core_coverage_passed": covered,
                "online_coverage_passed": summary["all_centervox_coverage_checks_passed"],
                "snapshot_path": str(path.relative_to(ROOT)),
            }
            records.append(record)
        scene_reports.append(
            {
                "scene": scene,
                "same_input_exact": exact_equal,
                "input_hashes": hashes,
                "all_direct_core_coverage_passed": all(
                    row["direct_core_coverage_passed"] for row in records if row["scene"] == scene
                ),
            }
        )
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    csv_rows = [{**row, "snapshot_path": row["snapshot_path"]} for row in records]
    with (TABLE_ROOT / "T25_same_cloud_initial_smoke.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    report = {
        "experiment": "07",
        "gate": "E7-D3 immutable initial snapshot",
        "mock_data": False,
        "scope": "generation-0 smoke snapshot; full crossed trajectory replay remains pending",
        "passed": all(
            row["same_input_exact"] and row["all_direct_core_coverage_passed"]
            for row in scene_reports
        ),
        "scenes": scene_reports,
        "records": records,
    }
    (RESULT_ROOT / "same_cloud_initial_snapshot_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

