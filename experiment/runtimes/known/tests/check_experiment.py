"""Independent delivery checks for the known-environment full-volume experiment."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "core"))

from known_volume import build_volume_cover, coverage_audit
from run import SUCCESS_HOLD_CYCLES, SUCCESS_TOLERANCE, load_scene


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maximum_hold(errors: np.ndarray, penetrating: np.ndarray) -> int:
    current = 0
    maximum = 0
    for error, collision in zip(errors, penetrating):
        current = current + 1 if error < SUCCESS_TOLERANCE and collision == 0 else 0
        maximum = max(maximum, current)
    return maximum


def main() -> None:
    scene = load_scene()
    sphere = build_volume_cover(scene.boxes, "sphere")
    ellipsoid = build_volume_cover(scene.boxes, "ellipsoid")
    assert len(sphere.centers) == len(ellipsoid.centers) == 368
    assert np.array_equal(sphere.centers, ellipsoid.centers)
    assert np.array_equal(sphere.cell_half_extents, ellipsoid.cell_half_extents)
    assert np.array_equal(sphere.box_indices, ellipsoid.box_indices)
    assert np.allclose(sphere.sphere_radii, np.linalg.norm(sphere.cell_half_extents, axis=1))
    expected_shapes = np.asarray(
        [np.diag(3.0 * half * half) for half in ellipsoid.cell_half_extents]
    )
    assert np.array_equal(ellipsoid.ellipsoid_shapes, expected_shapes)
    for cover in (sphere, ellipsoid):
        assert coverage_audit(cover)["complete_solid_cell_volume_covered"]

    for box_index, box in enumerate(scene.boxes):
        mask = sphere.box_indices == box_index
        reconstructed = float(np.sum(8.0 * np.prod(sphere.cell_half_extents[mask], axis=1)))
        expected = float(8.0 * np.prod(np.asarray(box.half_size)))
        assert abs(reconstructed - expected) <= 1e-14

    report = {}
    for case, regenerated in (("sphere", sphere), ("ellipsoid", ellipsoid)):
        directory = ROOT / "results" / case
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        logged = np.load(directory / "proxies.npz")
        assert np.array_equal(logged["centers"], regenerated.centers)
        assert np.array_equal(logged["cell_half_extents"], regenerated.cell_half_extents)
        assert np.array_equal(logged["box_indices"], regenerated.box_indices)
        assert np.array_equal(logged["sphere_radii"], regenerated.sphere_radii)
        assert np.array_equal(logged["ellipsoid_shapes"], regenerated.ellipsoid_shapes)
        q = np.load(directory / "q_history.npy")
        ee = np.load(directory / "ee_history.npy")
        assert q.shape == (3001, 6)
        assert ee.shape == (3001, 3)
        with (directory / "cycles.csv").open(encoding="utf-8") as handle:
            cycles = list(csv.DictReader(handle))
        assert len(cycles) == 3000
        errors = np.asarray([float(item["error_m"]) for item in cycles])
        penetrating = np.asarray([int(item["exact_penetrating_contact_count"]) for item in cycles])
        recomputed_final = float(np.linalg.norm(np.asarray(scene.waypoints[-1]) - ee[-1]))
        assert abs(recomputed_final - summary["final_error_m"]) <= 1e-14
        assert abs(errors[-1] - summary["final_error_m"]) <= 1e-14
        assert maximum_hold(errors, penetrating) == summary["maximum_success_hold_cycles"]
        assert summary["success"] == (maximum_hold(errors, penetrating) >= SUCCESS_HOLD_CYCLES)
        assert summary["exact_penetrating_cycles"] == int(np.count_nonzero(penetrating)) == 0
        assert summary["environment_known"] is True
        assert summary["depth_camera_used"] is False
        assert summary["point_cloud_used"] is False
        assert summary["online_voxel_map_used"] is False
        assert summary["path_planner"] is None
        assert summary["source_hashes"] == {
            name: hash_file(ROOT / name) for name in summary["source_hashes"]
        }
        report[case] = {
            "success": summary["success"],
            "final_error_mm": 1000.0 * summary["final_error_m"],
            "maximum_success_hold_cycles": summary["maximum_success_hold_cycles"],
            "penetrating_cycles": summary["exact_penetrating_cycles"],
        }

    assert report["ellipsoid"]["success"] is True
    assert report["sphere"]["success"] is False
    robust = json.loads(
        (ROOT / "results" / "sphere_same_qp_slsqp_validation" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert robust["fallback_cycles"] > 0
    assert robust["fallback_failures"] == 0
    assert robust["success"] is False
    assert robust["final_error_m"] > 0.3
    assert robust["exact_penetrating_cycles"] == 0
    print(
        json.dumps(
            {
                "all_checks_passed": True,
                "shared_complete_solid_cells": 368,
                "main_runs": report,
                "robust_sphere_same_qp": robust,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
