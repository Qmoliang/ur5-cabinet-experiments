"""Verify frozen ellipsoid and v4.4c adaptive sphere evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "formal_results" / "final_two_camera"
SPHERE_DIR = RESULTS / (
    "formal_strict_goal_v4_3_surface_core_sphere_camera_quarter_"
    "cv7.5mm_cr70mm_u1p05_stride1_vmax180mmps_affinity_"
    "perception_heavy_resolution_bounded_formal3_"
    "adaptive_irredundant_sphere_cover"
)
ELLIPSOID_DIR = RESULTS / (
    "formal_strict_goal_v4_3_surface_core_ellipsoid_camera_quarter_"
    "cv7.5mm_cr70mm_u1p05_stride1_vmax180mmps_affinity_"
    "perception_heavy_formal_redesign_no_coreinflate_r70_paired"
)


def load_summary(directory: Path) -> tuple[Path, dict]:
    matches = list(directory.rglob("summary.json"))
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one summary below {directory}, found {len(matches)}"
        )
    return matches[0], json.loads(matches[0].read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-full-evidence", action="store_true")
    args = parser.parse_args()

    sphere_path, sphere = load_summary(SPHERE_DIR)
    ellipsoid_path, ellipsoid = load_summary(ELLIPSOID_DIR)
    sphere_geometry_control = bool(
        sphere["representation"] == "sphere"
        and sphere["proxy_radius_policy"]
        == "adaptive_irredundant_fixed_candidate_cap"
        and sphere["all_centervox_coverage_checks_passed"]
        and sphere["all_sphere_cover_inclusion_minimal_checks_passed"]
        and sphere["all_mvt_oracle_checks_passed"]
        and sphere["all_proxy_radius_limit_checks_passed"]
        and sphere["sweep_audit_failures"] == 0
        and sphere["sweep_guard_modified_cycles"] == 0
        and sphere["exact_penetrating_cycles"] == 0
        and sphere["realtime_controller_passed"]
        and sphere["final_proxy_count"] == 3697
        and sphere["maximum_observed_sphere_effective_radius_m"] < 0.030
    )
    ellipsoid_frozen = bool(
        ellipsoid["representation"] == "ellipsoid"
        and ellipsoid["success"]
        and ellipsoid["final_error_m"] < 0.001
        and ellipsoid["final_proxy_count"] == 2187
    )
    full_evidence = bool(
        sphere_geometry_control and sphere["observability_passed"]
    )
    result = {
        "sphere_summary": str(sphere_path),
        "ellipsoid_summary": str(ellipsoid_path),
        "sphere_geometry_control_passed": sphere_geometry_control,
        "sphere_task_success": sphere["success"],
        "sphere_final_error_m": sphere["final_error_m"],
        "sphere_final_proxy_count": sphere["final_proxy_count"],
        "sphere_controller_p99_ms": sphere["controller_ms_p99"],
        "sphere_observability_passed": sphere["observability_passed"],
        "sphere_observability_late": sphere["observability_late_samples"],
        "sphere_observability_never": sphere[
            "observability_never_observed_samples"
        ],
        "ellipsoid_frozen_passed": ellipsoid_frozen,
        "ellipsoid_final_error_m": ellipsoid["final_error_m"],
        "ellipsoid_final_proxy_count": ellipsoid["final_proxy_count"],
        "full_evidence_passed": full_evidence,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not sphere_geometry_control or not ellipsoid_frozen:
        raise SystemExit(1)
    if args.require_full_evidence and not full_evidence:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

