"""Run protocol-v5 online LiuQP with independent irredundant certificates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psutil

from run_protocol_v3_async_online import run_one
from run_strict_goal_v4_3_surface_core import _affinities


ROOT = Path(__file__).resolve().parent
FORMAL_CYCLES = 3000
MAX_SNAPSHOT_AGE_P99_MS = 5000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("representation", choices=("sphere", "ellipsoid"))
    parser.add_argument(
        "--map-mode",
        choices=("v52_single_union", "v53_directional_leaves"),
        default="v52_single_union",
        help="Freeze the historical v5.2 map or use the v5.3 directional-leaf union.",
    )
    parser.add_argument("--smoke-cycles", type=int)
    parser.add_argument("--radius-limit-mm", type=float, default=70.0)
    parser.add_argument(
        "--sphere-policy",
        choices=("cap", "resolution_bounded"),
        default="cap",
    )
    parser.add_argument(
        "--scene-version",
        choices=("camera_quarter", "camera_compensated", "camera_v5_balanced"),
        default="camera_v5_balanced",
    )
    parser.add_argument("--camera-width", type=int, default=160)
    parser.add_argument("--camera-height", type=int, default=90)
    parser.add_argument("--camera-pixel-stride", type=int, default=1)
    parser.add_argument("--centervox-size-mm", type=float, default=7.5)
    parser.add_argument("--uncertainty-union-limit", type=float, default=1.05)
    parser.add_argument("--maximum-cycles", type=int)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--affinity-layout",
        choices=(
            "halves",
            "interleaved",
            "perception_heavy",
            "balanced_14_18",
            "isolated_12_2_18",
        ),
        default="isolated_12_2_18",
    )
    parser.add_argument(
        "--ellipsoid-pair-threads", type=int, choices=range(4, 9), default=6
    )
    parser.add_argument(
        "--control-priority",
        choices=("normal", "above_normal"),
        default="above_normal",
    )
    parser.add_argument("--task-gain", type=float, default=2.0)
    parser.add_argument("--max-task-speed", type=float, default=0.18)
    parser.add_argument("--osqp-adaptive-row-threshold", type=int, default=512)
    parser.add_argument("--osqp-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--osqp-max-iterations", type=int, default=1000)
    args = parser.parse_args()

    if args.radius_limit_mm <= 0.0:
        parser.error("--radius-limit-mm must be positive")
    if args.centervox_size_mm <= 0.0:
        parser.error("--centervox-size-mm must be positive")
    if args.camera_width <= 0 or args.camera_height <= 0:
        parser.error("camera dimensions must be positive")
    if args.camera_pixel_stride <= 0:
        parser.error("--camera-pixel-stride must be positive")
    if args.maximum_cycles is not None and args.smoke_cycles is not None:
        parser.error("use either --maximum-cycles or --smoke-cycles")

    radius_limit = args.radius_limit_mm / 1000.0
    centervox_size = args.centervox_size_mm / 1000.0
    cycle_limit = (
        args.maximum_cycles
        if args.maximum_cycles is not None
        else (
            args.smoke_cycles
            if args.smoke_cycles is not None
            else FORMAL_CYCLES
        )
    )
    formal = args.smoke_cycles is None and args.maximum_cycles is None
    sphere_cover_mode = (
        "cap_irredundant"
        if args.sphere_policy == "cap"
        else "adaptive_irredundant"
    )
    ellipsoid_cover_mode = (
        "adaptive_irredundant"
        if args.representation == "ellipsoid"
        else "matched"
    )
    radius_limit_representation = args.representation

    control, control_compute, publication, perception = _affinities(
        args.affinity_layout
    )
    pair_threads = (
        args.ellipsoid_pair_threads
        if args.ellipsoid_pair_threads is not None
        else (
            max(4, min(8, len(control_compute) // 2))
            if control_compute is not None
            else (7 if control is not None and len(control) == 14 else 8)
        )
    )

    phase = "formal" if formal else "dev"
    protocol_tag = (
        "protocol_v5_3"
        if args.map_mode == "v53_directional_leaves"
        else "protocol_v5_0"
    )
    leaf = (
        f"{protocol_tag}_{phase}_{args.representation}_"
        f"{args.sphere_policy}_R{args.radius_limit_mm:g}mm_"
        f"{args.scene_version}_cam{args.camera_width}x{args.camera_height}"
        f"s{args.camera_pixel_stride}_cv{args.centervox_size_mm:g}mm"
    )
    if args.label:
        leaf += f"_{args.label}"
    output = (
        ROOT / "formal_results" / "final_two_camera" / protocol_tag / leaf
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite v5 result: {output}")
    output.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()
    original_affinity = process.cpu_affinity()
    original_priority = process.nice()
    if control is not None:
        process.cpu_affinity(list(control))
    if args.control_priority == "above_normal":
        process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    try:
        summary = run_one(
            args.representation,
            "mvt_simd",
            output,
            maximum_cycles=int(cycle_limit),
            unknown_policy="observed_only",
            realtime_pacing=True,
            perception_executor="process",
            centervox_size=centervox_size,
            maximum_uncertainty_union_inflation=args.uncertainty_union_limit,
            certificate_radius_limit=radius_limit,
            uncertainty_fusion_mode=(
                "partitioned_separate_uncertainty"
                if args.map_mode == "v53_directional_leaves"
                else "separate_uncertainty"
            ),
            direct_thin_axis_inflation=1.0,
            direct_tangent_subdivisions=1,
            direct_partition_mode="grid",
            sphere_cover_mode=(
                sphere_cover_mode
                if args.representation == "sphere"
                else "matched"
            ),
            ellipsoid_cover_mode=ellipsoid_cover_mode,
            radius_limit_representation=radius_limit_representation,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_pixel_stride=args.camera_pixel_stride,
            camera_names=("ur5_depth_wrist", "ur5_depth_forearm"),
            scene_version=args.scene_version,
            record_dense_map_snapshots=True,
            sweep_guard_mode="audit_only",
            success_tolerance=0.001,
            success_hold_cycles=50,
            stop_on_success=False,
            perception_cpu_affinity=perception,
            control_cpu_affinity=control,
            ellipsoid_pair_threads=pair_threads,
            control_compute_cpu_affinity=control_compute,
            publication_cpu_affinity=publication,
            control_process_priority=args.control_priority,
            perception_process_priority="normal",
            task_gain=args.task_gain,
            max_task_speed=args.max_task_speed,
            osqp_adaptive_row_threshold=args.osqp_adaptive_row_threshold,
            osqp_tolerance=args.osqp_tolerance,
            osqp_max_iterations=args.osqp_max_iterations,
        )
    finally:
        process.nice(original_priority)
        process.cpu_affinity(original_affinity)

    coverage_minimal = bool(
        summary["all_sphere_cover_inclusion_minimal_checks_passed"]
        if args.representation == "sphere"
        else summary["all_certificate_cover_inclusion_minimal_checks_passed"]
    )
    snapshot_age_passed = bool(
        summary["snapshot_age_ms_p99"] <= MAX_SNAPSHOT_AGE_P99_MS
    )
    non_task_gates = bool(
        summary["observability_passed"]
        and summary["all_centervox_coverage_checks_passed"]
        and coverage_minimal
        and summary["all_mvt_oracle_checks_passed"]
        and summary["all_proxy_radius_limit_checks_passed"]
        and summary["exact_penetrating_cycles"] == 0
        and summary["sweep_audit_failures"] == 0
        and summary["sweep_guard_modified_cycles"] == 0
        and summary["liu_qp_command_executed_without_guard_modification"]
        and summary["realtime_controller_passed"]
        and summary["deadline_misses"] == 0
        and snapshot_age_passed
    )
    completion = {
        "protocol": "v5.3" if args.map_mode == "v53_directional_leaves" else "v5.0",
        "map_mode": args.map_mode,
        "formal": formal,
        "mock_data": False,
        "run_name": summary["run_name"],
        "representation": args.representation,
        "sphere_policy": args.sphere_policy,
        "ellipsoid_cover_mode": ellipsoid_cover_mode,
        "radius_limit_representation": radius_limit_representation,
        "radius_limit_m": radius_limit,
        "camera_resolution": [args.camera_width, args.camera_height],
        "camera_pixel_stride": args.camera_pixel_stride,
        "scene_version": args.scene_version,
        "affinity_layout": args.affinity_layout,
        "native_pair_threads": pair_threads,
        "control_priority": args.control_priority,
        "cycles": summary["cycles"],
        "success": summary["success"],
        "ever_sustained_success": summary["ever_sustained_success"],
        "minimum_error_m": summary["minimum_error_m"],
        "final_error_m": summary["final_error_m"],
        "final_proxy_count": summary["final_proxy_count"],
        "coverage_minimal": coverage_minimal,
        "observability_passed": summary["observability_passed"],
        "published_perception_frames": summary["published_perception_frames"],
        "snapshot_age_ms_p99": summary.get("snapshot_age_ms_p99"),
        "snapshot_age_p99_limit_ms": MAX_SNAPSHOT_AGE_P99_MS,
        "snapshot_age_passed": snapshot_age_passed,
        "controller_ms_p99": summary["controller_ms_p99"],
        "deadline_misses": summary["deadline_misses"],
        "realtime_controller_passed": summary["realtime_controller_passed"],
        "non_task_gates_passed": non_task_gates,
        "evidence_eligible": bool(formal and non_task_gates),
    }
    (output / "v5_run_complete.json").write_text(
        json.dumps(completion, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(completion, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
