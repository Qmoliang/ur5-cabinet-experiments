"""Run the frozen v4.2e reachable, partitioned thin-ellipsoid protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psutil

from run_protocol_v3_async_online import run_one


ROOT = Path(__file__).resolve().parent
THIN_AXIS_INFLATION = 1.10
TANGENT_SUBDIVISIONS = 2
FORMAL_CYCLES = 3000  # 60 s at the frozen 50 Hz control rate.


def main(
    *,
    scene_version: str = "camera_quarter",
    protocol_version: str = "v4_2e",
    centervox_size: float = 0.0075,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-cycles", type=int, default=None)
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
        default="halves",
        help="process-only scheduling ablation; never changes geometry or QP",
    )
    parser.add_argument(
        "--ellipsoid-pair-threads",
        type=int,
        choices=range(4, 9),
        default=None,
        help=(
            "explicit C++ closest-point OpenMP ablation; defaults to the "
            "physical-core count implied by the affinity layout"
        ),
    )
    parser.add_argument(
        "--control-priority",
        choices=("normal", "above_normal"),
        default="normal",
        help=(
            "Windows process scheduling ablation; the perception worker "
            "always resets itself to normal priority"
        ),
    )
    args = parser.parse_args()
    cycle_limit = FORMAL_CYCLES if args.smoke_cycles is None else args.smoke_cycles

    logical_cpus = os.cpu_count() or 1
    control_compute_affinity = None
    publication_affinity = None
    if logical_cpus >= 16:
        # The exact closest-point kernel uses eight OpenMP workers.  On the
        # 16-core/32-thread evaluation host, an even process split gives those
        # workers eight physical cores while keeping perception isolated on
        # the other half.  This changes scheduling only, never geometry/QP.
        split = logical_cpus // 2
        if args.affinity_layout == "interleaved":
            control_affinity = tuple(range(0, logical_cpus, 2))
            perception_affinity = tuple(range(1, logical_cpus, 2))
        elif args.affinity_layout == "perception_heavy":
            split = max(8, 3 * logical_cpus // 8)
            control_affinity = tuple(range(0, split))
            perception_affinity = tuple(range(split, logical_cpus))
        elif args.affinity_layout == "balanced_14_18":
            split = 14 if logical_cpus == 32 else logical_cpus // 2
            control_affinity = tuple(range(0, split))
            perception_affinity = tuple(range(split, logical_cpus))
        elif args.affinity_layout == "isolated_12_2_18":
            if logical_cpus != 32:
                raise RuntimeError(
                    "isolated_12_2_18 requires the formal 32-logical-CPU host"
                )
            control_affinity = tuple(range(0, 14))
            control_compute_affinity = tuple(range(0, 12))
            publication_affinity = tuple(range(12, 14))
            perception_affinity = tuple(range(14, 32))
        else:
            control_affinity = tuple(range(0, split))
            perception_affinity = tuple(range(split, logical_cpus))
    elif logical_cpus >= 8:
        split = logical_cpus // 2
        control_affinity = tuple(range(0, split))
        perception_affinity = tuple(range(split, logical_cpus))
    else:
        control_affinity = None
        perception_affinity = None
    # The fused closest-point kernel parallelizes over robot certificate
    # spheres.  Match its width to the physical-core budget of the isolated
    # control partition; 14 logical CPUs correspond to seven physical cores
    # on the formal 16-core/32-thread host.
    ellipsoid_pair_threads = (
        args.ellipsoid_pair_threads
        if args.ellipsoid_pair_threads is not None
        else (
            len(control_compute_affinity) // 2
            if control_compute_affinity is not None
            else (
                7
                if control_affinity is not None and len(control_affinity) == 14
                else 8
            )
        )
    )

    prefix = "formal" if args.smoke_cycles is None else "smoke"
    leaf = (
        f"{prefix}_strict_goal_{protocol_version}_reachable_partitioned_thin"
        f"_{scene_version}_affinity_{args.affinity_layout}"
    )
    if args.label:
        leaf += f"_{args.label}"
    output = ROOT / "formal_results" / "final_two_camera" / leaf
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite preserved evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()
    original_affinity = process.cpu_affinity()
    original_priority = process.nice()
    if control_affinity is not None:
        process.cpu_affinity(list(control_affinity))
    if args.control_priority == "above_normal":
        process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    try:
        summary = run_one(
            "ellipsoid",
            "mvt_simd",
            output,
            maximum_cycles=cycle_limit,
            unknown_policy="observed_only",
            realtime_pacing=True,
            perception_executor="process",
            centervox_size=centervox_size,
            maximum_uncertainty_union_inflation=None,
            certificate_radius_limit=None,
            uncertainty_fusion_mode="fused_certified_centervox",
            direct_thin_axis_inflation=THIN_AXIS_INFLATION,
            direct_tangent_subdivisions=TANGENT_SUBDIVISIONS,
            direct_partition_mode="grid",
            camera_width=320,
            camera_height=180,
            camera_pixel_stride=1,
            camera_names=("ur5_depth_wrist", "ur5_depth_forearm"),
            scene_version=scene_version,
            record_dense_map_snapshots=True,
            sweep_guard_mode="audit_only",
            success_tolerance=0.001,
            success_hold_cycles=50,
            stop_on_success=False,
            perception_cpu_affinity=perception_affinity,
            control_cpu_affinity=control_affinity,
            ellipsoid_pair_threads=ellipsoid_pair_threads,
            control_compute_cpu_affinity=control_compute_affinity,
            publication_cpu_affinity=publication_affinity,
            control_process_priority=args.control_priority,
            perception_process_priority="normal",
        )
    finally:
        process.nice(original_priority)
        process.cpu_affinity(original_affinity)

    completion = {
        "protocol": (
            f"strict_goal_{protocol_version}_reachable_partitioned_thin"
        ),
        "scene_version": scene_version,
        "centervox_size_m": centervox_size,
        "formal": args.smoke_cycles is None,
        "mock_data": False,
        "cycle_limit": cycle_limit,
        "time_limit_s": 0.02 * cycle_limit,
        "run_name": summary["run_name"],
        "success": summary["success"],
        "ever_sustained_success": summary["ever_sustained_success"],
        "first_success_completion_time_s": summary[
            "first_success_completion_time_s"
        ],
        "minimum_error_m": summary["minimum_error_m"],
        "final_error_m": summary["final_error_m"],
        "thin_axis_inflation": THIN_AXIS_INFLATION,
        "tangent_subdivisions": TANGENT_SUBDIVISIONS,
        "affinity_layout": args.affinity_layout,
        "ellipsoid_pair_threads": ellipsoid_pair_threads,
        "control_compute_cpu_affinity": control_compute_affinity,
        "publication_cpu_affinity": publication_affinity,
        "control_process_priority": args.control_priority,
        "perception_process_priority": "normal",
        "safety_margin_m": summary["safety_margin_m"],
        "all_centervox_coverage_checks_passed": summary[
            "all_centervox_coverage_checks_passed"
        ],
        "all_mvt_oracle_checks_passed": summary[
            "all_mvt_oracle_checks_passed"
        ],
        "observability_passed": summary["observability_passed"],
        "maximum_penetration_m": summary["maximum_penetration_m"],
        "sweep_audit_failures": summary["sweep_audit_failures"],
        "controller_ms_p99": summary["controller_ms_p99"],
        "realtime_controller_passed": summary[
            "realtime_controller_passed"
        ],
        "evidence_eligible": summary["evidence_eligible"],
    }
    (output / f"{protocol_version}_run_complete.json").write_text(
        json.dumps(completion, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(completion, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
