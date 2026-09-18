"""Read-only count/coverage audit for the frozen v4.4c sphere baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "formal_results" / "final_two_camera" / (
    "formal_strict_goal_v4_3_surface_core_sphere_camera_quarter_"
    "cv7.5mm_cr70mm_u1p05_stride1_vmax180mmps_"
    "affinity_perception_heavy_resolution_bounded_formal3_"
    "adaptive_irredundant_sphere_cover"
)
FILTER_SIZE = 0.0075


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = list(RESULT_ROOT.rglob("final_causal_proxies.npz"))
    if len(paths) != 1:
        raise AssertionError(f"expected one frozen proxy file, found {len(paths)}")
    archive = np.load(paths[0])
    centers = np.asarray(archive["centers"], dtype=float)
    radii = np.asarray(archive["sphere_radii"], dtype=float)
    points = np.asarray(archive["filtered_points"], dtype=float)
    owner = np.asarray(archive["filtered_cluster_indices"], dtype=np.int64)
    cell_u = np.asarray(archive["filtered_uncertainty_shapes"], dtype=float)
    # The frozen v4.4 artifact predates persistence of filtered_point_offsets.
    # In this directional path relocation/residual was already included in U;
    # report the missing field explicitly instead of silently inventing values.
    if "filtered_point_offsets" in archive.files:
        cell_offsets = np.asarray(archive["filtered_point_offsets"], dtype=float)
        offset_source = "persisted"
    else:
        cell_offsets = np.zeros(len(points), dtype=float)
        offset_source = "absent_in_v4.4_artifact_directional_U_used_zero"
    support = np.sqrt(
        np.maximum(np.linalg.eigvalsh(cell_u)[:, -1], 0.0)
    ) + cell_offsets

    if len(owner) != len(points) or np.any((owner < 0) | (owner >= len(centers))):
        raise AssertionError("invalid frozen owner array")
    assigned = np.bincount(owner, minlength=len(centers))
    if np.any(assigned == 0):
        raise AssertionError("frozen sphere with no assigned CenterVox")
    assigned_max_distance = np.zeros(len(centers), dtype=float)
    assigned_max_required = np.zeros(len(centers), dtype=float)
    for proxy_index in range(len(centers)):
        indices = np.flatnonzero(owner == proxy_index)
        distance = np.linalg.norm(points[indices] - centers[proxy_index], axis=1)
        assigned_max_distance[proxy_index] = float(np.max(distance))
        assigned_max_required[proxy_index] = float(
            np.max(distance + support[indices])
        )
    assigned_radius_slack = radii - assigned_max_required

    cover_count = np.zeros(len(points), dtype=np.int32)
    chunk = 128
    for start in range(0, len(centers), chunk):
        stop = min(start + chunk, len(centers))
        distance = np.linalg.norm(
            centers[start:stop, None, :] - points[None, :, :], axis=2
        )
        cover_count += np.count_nonzero(
            distance + support[None, :] <= radii[start:stop, None] + 1.0e-12,
            axis=0,
        ).astype(np.int32)
    if np.any(cover_count < 1):
        raise AssertionError("frozen final spheres do not cover every CenterVox")

    unique_witnesses = np.zeros(len(centers), dtype=np.int64)
    covered_cells = np.zeros(len(centers), dtype=np.int64)
    unique_mask = cover_count == 1
    for start in range(0, len(centers), chunk):
        stop = min(start + chunk, len(centers))
        distance = np.linalg.norm(
            centers[start:stop, None, :] - points[None, :, :], axis=2
        )
        covered = (
            distance + support[None, :]
            <= radii[start:stop, None] + 1.0e-12
        )
        covered_cells[start:stop] = np.count_nonzero(covered, axis=1)
        unique_witnesses[start:stop] = np.count_nonzero(
            covered & unique_mask[None, :], axis=1
        )

    geometrically_contained = np.zeros(len(centers), dtype=bool)
    for start in range(0, len(centers), chunk):
        stop = min(start + chunk, len(centers))
        distance = np.linalg.norm(
            centers[start:stop, None, :] - centers[None, :, :], axis=2
        )
        contained = (
            distance + radii[start:stop, None]
            <= radii[None, :] + 1.0e-12
        )
        row = np.arange(stop - start)
        contained[row, np.arange(start, stop)] = False
        geometrically_contained[start:stop] = np.any(contained, axis=1)

    geometric_budget = np.sqrt(3.0) * FILTER_SIZE
    observed_area_estimate = len(points) * FILTER_SIZE**2
    disk_area_lower_bound = observed_area_estimate / (
        np.pi * geometric_budget**2
    )
    square_grid_reference = observed_area_estimate / (
        2.0 * geometric_budget**2
    )
    result = {
        "artifact": str(paths[0]),
        "proxy_count": int(len(centers)),
        "center_voxel_count": int(len(points)),
        "filter_size_mm": FILTER_SIZE * 1000.0,
        "geometric_budget_mm": geometric_budget * 1000.0,
        "cell_offset_source": offset_source,
        "sphere_radius_mm": {
            key: value * 1000.0 for key, value in quantiles(radii).items()
        },
        "cell_support_radius_mm": {
            key: value * 1000.0 for key, value in quantiles(support).items()
        },
        "assigned_centervox_per_sphere": quantiles(assigned),
        "all_covered_centervox_per_sphere": quantiles(covered_cells),
        "coverage_multiplicity_per_centervox": quantiles(cover_count),
        "unique_witnesses_per_sphere": quantiles(unique_witnesses),
        "spheres_with_zero_unique_witness": int(
            np.count_nonzero(unique_witnesses == 0)
        ),
        "geometrically_contained_spheres": int(
            np.count_nonzero(geometrically_contained)
        ),
        "assigned_max_center_distance_mm": {
            key: value * 1000.0
            for key, value in quantiles(assigned_max_distance).items()
        },
        "assigned_radius_slack_micrometre": {
            key: value * 1.0e6
            for key, value in quantiles(assigned_radius_slack).items()
        },
        "naive_observed_surface_area_m2": float(observed_area_estimate),
        "disk_area_ideal_lower_count": float(disk_area_lower_bound),
        "square_grid_surface_reference_count": float(square_grid_reference),
        "actual_over_square_grid_reference": float(
            len(centers) / square_grid_reference
        ),
        "actual_over_twice_square_grid_reference": bool(
            len(centers) > 2.0 * square_grid_reference
        ),
        "interpretation_limit": (
            "area references approximate one locally planar surface sample per "
            "occupied CenterVox; edges, oblique surfaces and occlusion change area"
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
