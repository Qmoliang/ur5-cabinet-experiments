"""Audit MVT versus independent full AABB over Experiment 07 drawer trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from model import build_model, build_robot_certificate, certificate_world_positions, set_configuration
from native_mvt import NativeMultilevelMVT
from protocol_drawer_scene import NEAR_DISTANCE, SAFETY_MARGIN
from run_experiment_07 import GROUPS
from run_protocol_v3_async_online import _protocol_scene
from run_protocol_v3_online_ablation import (
    MOTION_BROADPHASE_PADDING,
    MVT_BASE_VOXEL_SIZE,
    MVT_NUMERICAL_OUTWARD_PADDING,
)

ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "formal_results" / "experiment_07"
GROUP_NAMES = ("S", "SA", "E0", "ET30")


def latest_300_run(group: str) -> Path:
    candidates = []
    for batch in (RESULT_ROOT / "drawer").glob("smoke_*"):
        manifest_path = batch / "batch_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest.get("results", []):
            if row.get("group") == group and row.get("cycles") == 300:
                candidates.append((batch.stat().st_mtime_ns, batch / group / "run_00"))
    if not candidates:
        raise FileNotFoundError(group)
    run_root = max(candidates)[1]
    return next(path for path in run_root.iterdir() if path.is_dir())


def half_extents(snapshot, representation: str) -> np.ndarray:
    offsets = np.asarray(snapshot["uncertainty_offsets"], dtype=float)
    if representation == "sphere":
        radii = np.asarray(snapshot["sphere_radii"], dtype=float) + offsets
        return np.repeat(radii[:, None], 3, axis=1)
    outer = np.asarray(snapshot["ellipsoid_outer_shapes"], dtype=float)
    return np.sqrt(np.maximum(np.diagonal(outer, axis1=1, axis2=2), 0.0)) + offsets[:, None]


def digest_rows(rows) -> str:
    digest = hashlib.sha256()
    for values in rows:
        array = np.ascontiguousarray(values, dtype=np.int64)
        digest.update(np.asarray([len(array)], dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> None:
    scene = _protocol_scene("exp07_drawer")
    model = build_model(scene)
    data = mujoco.MjData(model)
    robot = build_robot_certificate(model)
    robot_radii = np.asarray([item.radius for item in robot], dtype=float)
    query_padding = (
        NEAR_DISTANCE
        + SAFETY_MARGIN
        + MOTION_BROADPHASE_PADDING
        + MVT_NUMERICAL_OUTWARD_PADDING
    )
    maximum_query_half = float(np.max(robot_radii) + query_padding)
    reports = []
    for group in GROUP_NAMES:
        run = latest_300_run(group)
        q_values = np.vstack((np.asarray(scene.q0, dtype=float), np.load(run / "q_history.npy")))
        snapshot = np.load(run / "final_causal_proxies.npz")
        representation = GROUPS[group]["representation"]
        centers = np.asarray(snapshot["centers"], dtype=float)
        half = half_extents(snapshot, representation)
        lo = centers - half
        hi = centers + half
        table = NativeMultilevelMVT(
            centers,
            half + MVT_NUMERICAL_OUTWARD_PADDING,
            base_voxel_size=MVT_BASE_VOXEL_SIZE,
            maximum_query_half_extent=maximum_query_half,
            query_padding=query_padding,
            simd=True,
        )
        expected_rows = []
        filtered_actual_rows = []
        missing = 0
        extras = 0
        order_equal = True
        try:
            for q in q_values:
                set_configuration(model, data, q)
                positions = certificate_world_positions(data, robot)
                actual_rows = table.query_spheres(positions, robot_radii)
                for center, radius, actual in zip(positions, robot_radii, actual_rows):
                    query_radius = float(radius) + query_padding
                    delta = np.maximum(np.maximum(lo - center, center - hi), 0.0)
                    expected = np.flatnonzero(
                        np.einsum("ij,ij->i", delta, delta)
                        <= query_radius * query_radius
                    )
                    actual = np.asarray(actual, dtype=np.int64)
                    filtered = actual[np.isin(actual, expected, assume_unique=True)]
                    missing += len(np.setdiff1d(expected, actual, assume_unique=True))
                    extras += len(np.setdiff1d(actual, expected, assume_unique=True))
                    order_equal = order_equal and np.array_equal(expected, filtered)
                    expected_rows.append(expected)
                    filtered_actual_rows.append(filtered)
        finally:
            table.close()
            snapshot.close()
        expected_hash = digest_rows(expected_rows)
        actual_hash = digest_rows(filtered_actual_rows)
        reports.append(
            {
                "group": group,
                "run": str(run.relative_to(ROOT)),
                "robot_queries": len(expected_rows),
                "missing_candidates": missing,
                "extra_broadphase_candidates": extras,
                "active_eligible_id_order_equal": order_equal,
                "full_aabb_logical_row_sha256": expected_hash,
                "mvt_filtered_logical_row_sha256": actual_hash,
                "logical_rows_equal": expected_hash == actual_hash,
                "passed": missing == 0 and order_equal and expected_hash == actual_hash,
            }
        )
    report = {
        "experiment": "07.2",
        "gate": "E7-D5b drawer development-trajectory MVT/AABB logical control input equivalence",
        "mock_data": False,
        "argument": (
            "A pair outside the independent sphere-AABB action-distance predicate "
            "cannot pass the exact NEAR predicate because AABB distance is a lower "
            "bound on exact proxy distance. MVT extras therefore create no QP row; "
            "the filtered eligible proxy ID order is compared exactly."
        ),
        "scope": "all 301 saved configurations x 65 robot balls for each 300-cycle drawer development run",
        "groups": reports,
        "passed": all(row["passed"] for row in reports),
    }
    output = RESULT_ROOT / "drawer_trajectory_mvt_equivalence_07_2.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()



