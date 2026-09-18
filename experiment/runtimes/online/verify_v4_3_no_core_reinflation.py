"""Verify the frozen v4.3 no-core-reinflation formal pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE / "formal_results" / "final_two_camera"
SUFFIX = (
    "camera_quarter_cv7.5mm_cr70mm_u1p05_stride1_vmax180mmps_"
    "affinity_perception_heavy_formal_redesign_no_coreinflate_r70_paired"
)


def _root(representation: str) -> Path:
    name = f"formal_strict_goal_v4_3_surface_core_{representation}_{SUFFIX}"
    return BASE / name


def _load_run(root: Path) -> tuple[dict, dict, Path]:
    summaries = list(root.rglob("summary.json"))
    proxy_files = list(root.rglob("final_causal_proxies.npz"))
    if len(summaries) != 1 or len(proxy_files) != 1:
        raise FileNotFoundError(f"incomplete or ambiguous formal root: {root}")
    completion_path = root / "v4_3_run_complete.json"
    completion = json.loads(completion_path.read_text("utf-8"))
    summary = json.loads(summaries[0].read_text("utf-8"))
    return summary, completion, proxy_files[0]


def _safety_gates(summary: dict) -> bool:
    return bool(
        summary["all_centervox_coverage_checks_passed"]
        and summary["all_mvt_oracle_checks_passed"]
        and summary["all_proxy_radius_limit_checks_passed"]
        and summary["sweep_audit_failures"] == 0
        and summary["sweep_guard_modified_cycles"] == 0
        and summary["exact_penetrating_cycles"] == 0
        and summary["liu_qp_command_executed_without_guard_modification"]
        and summary["realtime_controller_passed"]
        and summary["path_planner"] in ("", None)
        and not summary["random_dither"]
        and not summary["truth_observability_feedback_to_control"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-full-evidence", action="store_true")
    args = parser.parse_args()

    sphere, sphere_completion, _ = _load_run(_root("sphere"))
    ellipsoid, ellipsoid_completion, proxy_path = _load_run(
        _root("ellipsoid")
    )
    with np.load(proxy_path) as archive:
        core_axes = np.sqrt(
            np.maximum(
                np.linalg.eigvalsh(
                    np.asarray(archive["ellipsoid_shapes"], dtype=float)
                ),
                0.0,
            )
        )
        outer_axes = np.sqrt(
            np.maximum(
                np.linalg.eigvalsh(
                    np.asarray(archive["ellipsoid_outer_shapes"], dtype=float)
                ),
                0.0,
            )
        )
        uncertainty_axes = np.sqrt(
            np.maximum(
                np.linalg.eigvalsh(
                    np.asarray(
                        archive["proxy_uncertainty_shapes"], dtype=float
                    )
                ),
                0.0,
            )
        )

    shared_fields = (
        "centervox_filter_size_m",
        "certificate_radius_limit_m",
        "uncertainty_fusion_mode",
        "maximum_uncertainty_union_inflation_limit",
        "camera_width_px",
        "camera_height_px",
        "camera_pixel_stride",
        "success_tolerance_m",
        "success_hold_cycles",
        "task_gain",
        "max_task_speed_m_per_s",
        "osqp_absolute_tolerance",
        "osqp_relative_tolerance",
        "osqp_max_iterations",
    )
    shared_configuration = all(
        sphere[field] == ellipsoid[field] for field in shared_fields
    )
    no_core_reinflation = bool(
        ellipsoid["uncertainty_fusion_mode"] == "separate_uncertainty"
        and ellipsoid["direct_centervox_thin_axis_inflation"] == 1.0
        and "no direct thin-axis inflation"
        in ellipsoid_completion["core_construction"]
    )
    sphere_negative = bool(
        not sphere["success"]
        and not sphere["ever_sustained_success"]
        and sphere_completion["negative_control_eligible"]
    )
    ellipsoid_target = bool(
        ellipsoid["success"]
        and ellipsoid["ever_sustained_success"]
        and ellipsoid["final_error_m"] < ellipsoid["success_tolerance_m"]
    )
    core_task_acceptance = bool(
        shared_configuration
        and no_core_reinflation
        and sphere_negative
        and ellipsoid_target
        and _safety_gates(sphere)
        and _safety_gates(ellipsoid)
    )
    full_evidence = bool(
        core_task_acceptance
        and sphere["observability_passed"]
        and ellipsoid["observability_passed"]
    )
    report = {
        "protocol": "strict_goal_v4_3_surface_core",
        "mock_data": False,
        "shared_configuration": shared_configuration,
        "no_core_reinflation": no_core_reinflation,
        "core_short_axis_m": {
            "minimum": float(np.min(core_axes[:, 0])),
            "median": float(np.median(core_axes[:, 0])),
            "p99": float(np.quantile(core_axes[:, 0], 0.99)),
        },
        "core_long_axis_p99_m": float(
            np.quantile(core_axes[:, -1], 0.99)
        ),
        "separate_uncertainty_long_axis_p99_m": float(
            np.quantile(uncertainty_axes[:, -1], 0.99)
        ),
        "outer_long_axis_p99_m": float(
            np.quantile(outer_axes[:, -1], 0.99)
        ),
        "sphere": {
            "success": sphere["success"],
            "minimum_error_m": sphere["minimum_error_m"],
            "final_error_m": sphere["final_error_m"],
            "controller_ms_p99": sphere["controller_ms_p99"],
            "negative_control_eligible": sphere_negative,
            "observability_passed": sphere["observability_passed"],
        },
        "ellipsoid": {
            "success": ellipsoid["success"],
            "first_success_completion_time_s": ellipsoid[
                "first_success_completion_time_s"
            ],
            "minimum_error_m": ellipsoid["minimum_error_m"],
            "final_error_m": ellipsoid["final_error_m"],
            "controller_ms_p99": ellipsoid["controller_ms_p99"],
            "observability_passed": ellipsoid["observability_passed"],
            "observability_late_samples": ellipsoid[
                "observability_late_samples"
            ],
            "observability_never_observed_samples": ellipsoid[
                "observability_never_observed_samples"
            ],
        },
        "core_task_acceptance_passed": core_task_acceptance,
        "full_protocol_evidence_passed": full_evidence,
        "remaining_blocker": (
            None
            if full_evidence
            else "ellipsoid causal observed-before-risk gate"
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    required = full_evidence if args.require_full_evidence else core_task_acceptance
    raise SystemExit(0 if required else 1)


if __name__ == "__main__":
    main()
