"""Aggregate the certified drawer and FastIRIS-style sphere/ellipsoid runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_geometry_ablation"
PAIRS = {
    "shelf_drawer_certified": (
        "shelf_drawer_certified_sphere_direct_pca_graph_mahal_cluster100mm_over025mm_pc006mm_vox006mm_surfacecover_dur060s",
        "shelf_drawer_certified_ellipsoid_direct_pca_graph_mahal_cluster100mm_over025mm_pc006mm_vox006mm_surfacecover_dur030s",
    ),
    "fastiris_iiwa_shelf_style": (
        "fastiris_iiwa_shelf_style_sphere_direct_pca_graph_mahal_cluster100mm_over025mm_pc006mm_vox006mm_surfacecover",
        "fastiris_iiwa_shelf_style_ellipsoid_direct_pca_graph_mahal_cluster100mm_over025mm_pc006mm_vox006mm_surfacecover",
    ),
    "fastiris_4_shelves_style": (
        "fastiris_4_shelves_style_sphere_direct_pca_graph_mahal_cluster100mm_over025mm_pc006mm_vox006mm_surfacecover",
        "fastiris_4_shelves_style_ellipsoid_direct_pca_graph_mahal_cluster100mm_over025mm_pc006mm_vox006mm_surfacecover",
    ),
    "fastiris_iiwa_bins_style": (
        "fastiris_iiwa_bins_style_sphere_direct_pca_graph_mahal_cluster100mm_over025mm_pc006mm_vox006mm_surfacecover",
        "fastiris_iiwa_bins_style_ellipsoid_direct_pca_graph_mahal_cluster100mm_over025mm_pc006mm_vox006mm_surfacecover",
    ),
}
FIELDS = (
    "scene",
    "certificate",
    "success",
    "stalled",
    "simulated_time_s",
    "final_error_m",
    "minimum_certificate_clearance_m",
    "qdot_sign_flip_rate",
    "solved_qp_fraction",
    "obstacle_proxies",
    "surface_cover_radius_m",
    "surface_validation_spacing_m",
    "surface_validation_sampled_max_distance_m",
    "surface_validation_lipschitz_correction_m",
    "mean_broadphase_reduction_fraction",
)


def main() -> None:
    rows = []
    for scene, pair in PAIRS.items():
        summaries = []
        for certificate, directory in zip(("sphere", "ellipsoid"), pair):
            summary = json.loads(
                (RESULTS / directory / "summary.json").read_text(encoding="utf-8")
            )
            summary["certificate"] = certificate
            summaries.append(summary)
            rows.append({field: summary.get(field) for field in FIELDS})
        left, right = summaries
        for field in (
            "obstacle_proxies",
            "surface_cover_radius_m",
            "filtered_point_count",
            "pointcloud_cluster_size_m",
        ):
            if left[field] != right[field]:
                raise AssertionError(f"{scene}: unmatched {field}")
        if not left["surface_cover_applied"] or not right["surface_cover_applied"]:
            raise AssertionError(f"{scene}: continuous cover was disabled")

    json_path = RESULTS / "certified_surface_ablation_summary.json"
    csv_path = RESULTS / "certified_surface_ablation_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json_path)
    print(csv_path)
    for row in rows:
        print(
            f"{row['scene']:30s} {row['certificate']:9s} "
            f"success={str(row['success']):5s} "
            f"error={1000*row['final_error_m']:7.2f} mm "
            f"delta={1000*row['surface_cover_radius_m']:6.2f} mm"
        )


if __name__ == "__main__":
    main()
