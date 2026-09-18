"""Run the no-core-reinflation v4.3 paired LiuQP experiment.

The obstacle geometry is built from the same causal CenterVox representatives
for both representations.  The ellipsoid core Q encloses only representative
surface points.  Camera pixel/depth uncertainty remains a separate U term and
the CenterVox residual remains a scalar offset, so neither is fused back into
Q.  The ellipsoid controller evaluates the exact support sum

    sqrt(n' Q_robot n) + sqrt(n' Q_core n) + sqrt(n' U n) + offset

with its safeguarded Newton/bisection kernel.  The sphere control replaces the
same certified proxy by its matched isotropic support radius.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psutil

from run_protocol_v3_async_online import run_one


ROOT = Path(__file__).resolve().parent
FORMAL_CYCLES = 3000


def _affinities(layout: str):
    logical_cpus = os.cpu_count() or 1
    control_compute = None
    publication = None
    if logical_cpus >= 16:
        split = logical_cpus // 2
        if layout == "interleaved":
            control = tuple(range(0, logical_cpus, 2))
            perception = tuple(range(1, logical_cpus, 2))
        elif layout == "perception_heavy":
            split = max(8, 3 * logical_cpus // 8)
            control = tuple(range(0, split))
            perception = tuple(range(split, logical_cpus))
        elif layout == "balanced_14_18":
            split = 14 if logical_cpus == 32 else logical_cpus // 2
            control = tuple(range(0, split))
            perception = tuple(range(split, logical_cpus))
        elif layout == "isolated_12_2_18":
            if logical_cpus != 32:
                raise RuntimeError(
                    "isolated_12_2_18 requires the formal 32-logical-CPU host"
                )
            control = tuple(range(0, 14))
            control_compute = tuple(range(0, 12))
            publication = tuple(range(12, 14))
            perception = tuple(range(14, 32))
        else:
            control = tuple(range(0, split))
            perception = tuple(range(split, logical_cpus))
    elif logical_cpus >= 8:
        split = logical_cpus // 2
        control = tuple(range(0, split))
        perception = tuple(range(split, logical_cpus))
    else:
        control = None
        perception = None
    return control, control_compute, publication, perception


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("representation", choices=("sphere", "ellipsoid"))
    parser.add_argument("--smoke-cycles", type=int, default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--centervox-size", type=float, default=0.0075)
    parser.add_argument(
        "--certificate-radius-limit",
        type=float,
        default=0.070,
        help=(
            "maximum final proxy support radius; oversized occupied patches "
            "are recursively subdivided, never repaired by thickening Q_core"
        ),
    )
    parser.add_argument(
        "--scene-version",
        choices=("camera_quarter", "camera_blindspot", "camera_compensated"),
        default="camera_quarter",
    )
    parser.add_argument(
        "--camera-pixel-stride",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help=(
            "ray subsampling stride; the camera model expands each retained "
            "sample U to cover the skipped-pixel footprint"
        ),
    )
    parser.add_argument(
        "--uncertainty-union-limit",
        type=float,
        default=1.05,
        help=(
            "maximum trace inflation allowed when joining measurement U; "
            "this never changes the surface core Q"
        ),
    )
    parser.add_argument(
        "--sphere-cover-mode",
        choices=("auto", "matched", "adaptive_irredundant"),
        default="auto",
        help=(
            "auto enables adaptive irredundant map certificates only for the "
            "sphere run; ellipsoid runs remain on the frozen matched path"
        ),
    )
    parser.add_argument(
        "--affinity-layout",
        choices=(
            "halves",
            "interleaved",
            "perception_heavy",
            "balanced_14_18",
            "isolated_12_2_18",
        ),
        default="perception_heavy",
    )
    parser.add_argument(
        "--ellipsoid-pair-threads", type=int, choices=range(4, 9), default=None
    )
    parser.add_argument(
        "--control-priority", choices=("normal", "above_normal"), default="normal"
    )
    parser.add_argument("--task-gain", type=float, default=2.0)
    parser.add_argument("--max-task-speed", type=float, default=0.18)
    parser.add_argument("--osqp-adaptive-row-threshold", type=int, default=512)
    parser.add_argument("--osqp-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--osqp-max-iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.centervox_size <= 0.0:
        raise ValueError("centervox size must be positive")
    if args.certificate_radius_limit <= 0.0:
        raise ValueError("certificate radius limit must be positive")
    if args.uncertainty_union_limit < 1.0:
        raise ValueError("uncertainty union limit must be at least one")
    sphere_cover_mode = (
        ("adaptive_irredundant" if args.representation == "sphere" else "matched")
        if args.sphere_cover_mode == "auto"
        else args.sphere_cover_mode
    )
    if args.representation == "ellipsoid" and sphere_cover_mode != "matched":
        raise ValueError("ellipsoid v4.3 is frozen and cannot use sphere cover modes")

    cycle_limit = (
        FORMAL_CYCLES if args.smoke_cycles is None else int(args.smoke_cycles)
    )
    control, control_compute, publication, perception = _affinities(
        args.affinity_layout
    )
    pair_threads = (
        args.ellipsoid_pair_threads
        if args.ellipsoid_pair_threads is not None
        else (
            len(control_compute) // 2
            if control_compute is not None
            else (7 if control is not None and len(control) == 14 else 8)
        )
    )

    prefix = "formal" if args.smoke_cycles is None else "smoke"
    union_tag = f"{args.uncertainty_union_limit:.3f}".rstrip("0").rstrip(".")
    union_tag = union_tag.replace(".", "p")
    leaf = (
        f"{prefix}_strict_goal_v4_3_surface_core_{args.representation}"
        f"_{args.scene_version}_cv{args.centervox_size * 1000.0:g}mm"
        f"_cr{args.certificate_radius_limit * 1000.0:g}mm"
        f"_u{union_tag}_stride{args.camera_pixel_stride}"
        f"_vmax{args.max_task_speed * 1000.0:g}mmps"
        f"_affinity_{args.affinity_layout}"
    )
    if args.label:
        leaf += f"_{args.label}"
    if sphere_cover_mode == "adaptive_irredundant":
        leaf += "_adaptive_irredundant_sphere_cover"
    output = ROOT / "formal_results" / "final_two_camera" / leaf
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite preserved evidence: {output}")
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
            maximum_cycles=cycle_limit,
            unknown_policy="observed_only",
            realtime_pacing=True,
            perception_executor="process",
            centervox_size=args.centervox_size,
            maximum_uncertainty_union_inflation=args.uncertainty_union_limit,
            certificate_radius_limit=args.certificate_radius_limit,
            uncertainty_fusion_mode="separate_uncertainty",
            # This parameter is inactive in separate_uncertainty mode.  Keep
            # it at the identity value as an auditable guarantee that v4.3
            # never asks any direct/fused fitter to re-inflate Q_core.
            direct_thin_axis_inflation=1.0,
            direct_tangent_subdivisions=1,
            direct_partition_mode="grid",
            sphere_cover_mode=sphere_cover_mode,
            camera_width=320,
            camera_height=180,
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

    non_task_gates = bool(
        summary["observability_passed"]
        and summary["all_centervox_coverage_checks_passed"]
        and summary["all_mvt_oracle_checks_passed"]
        and summary["all_proxy_radius_limit_checks_passed"]
        and summary["exact_penetrating_cycles"] == 0
        and summary["sweep_audit_failures"] == 0
        and summary["sweep_guard_modified_cycles"] == 0
        and summary["liu_qp_command_executed_without_guard_modification"]
        and summary["realtime_controller_passed"]
    )
    completion = {
        "protocol": "strict_goal_v4_3_surface_core",
        "formal": args.smoke_cycles is None,
        "mock_data": False,
        "representation": args.representation,
        "scene_version": args.scene_version,
        "cycle_limit": cycle_limit,
        "run_name": summary["run_name"],
        "core_construction": (
            "PCA_surface_point_enclosure_only; no camera-U fusion; "
            "no direct thin-axis inflation"
        ),
        "constraint_support": (
            "robot_Q + surface_core_Q + camera_U + centervox_scalar_offset "
            "+ fixed_6mm_safety"
        ),
        "uncertainty_fusion_mode": summary["uncertainty_fusion_mode"],
        "uncertainty_union_limit": args.uncertainty_union_limit,
        "centervox_size_m": args.centervox_size,
        "certificate_radius_limit_m": args.certificate_radius_limit,
        "sphere_cover_mode": sphere_cover_mode,
        "sphere_cover_candidate_count_final": summary[
            "sphere_cover_candidate_count_final"
        ],
        "sphere_cover_selected_count_final": summary[
            "sphere_cover_selected_count_final"
        ],
        "sphere_cover_reverse_deleted_final": summary[
            "sphere_cover_reverse_deleted_final"
        ],
        "sphere_cover_inclusion_minimal": summary[
            "all_sphere_cover_inclusion_minimal_checks_passed"
        ],
        "oversized_proxy_policy": (
            "recursive spatial/uncertainty subdivision with per-child "
            "coverage audit; never thicken Q_core"
        ),
        "camera_pixel_stride": args.camera_pixel_stride,
        "task_gain": summary["task_gain"],
        "max_task_speed_m_per_s": summary["max_task_speed_m_per_s"],
        "configured_osqp_adaptive_row_threshold": summary[
            "configured_osqp_adaptive_row_threshold"
        ],
        "osqp_absolute_tolerance": summary["osqp_absolute_tolerance"],
        "osqp_relative_tolerance": summary["osqp_relative_tolerance"],
        "osqp_max_iterations": summary["osqp_max_iterations"],
        "success": summary["success"],
        "ever_sustained_success": summary["ever_sustained_success"],
        "first_success_completion_time_s": summary[
            "first_success_completion_time_s"
        ],
        "minimum_error_m": summary["minimum_error_m"],
        "final_error_m": summary["final_error_m"],
        "success_tolerance_m": summary["success_tolerance_m"],
        "success_comparison": summary["success_comparison"],
        "success_hold_cycles": summary["success_hold_cycles"],
        "controller_ms_p99": summary["controller_ms_p99"],
        "non_task_gates_passed": non_task_gates,
        "evidence_eligible": summary["evidence_eligible"],
        "negative_control_eligible": bool(
            args.smoke_cycles is None
            and args.representation == "sphere"
            and not summary["success"]
            and not summary["ever_sustained_success"]
            and non_task_gates
        ),
    }
    (output / "v4_3_run_complete.json").write_text(
        json.dumps(completion, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(completion, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
