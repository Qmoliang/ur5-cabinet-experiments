"""Run strict-goal v4.2c with one certified ellipsoid per CenterVox."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psutil

from run_protocol_v3_async_online import run_one


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-cycles", type=int, default=None)
    parser.add_argument("--smoke-label", default="")
    parser.add_argument("--formal-label", default="")
    parser.add_argument(
        "--scene-version",
        choices=(
            "formal",
            "uniform_v3_1",
            "camera_mid",
            "camera_quarter",
            "camera_three_sixteenths",
        ),
        default="formal",
        help="same frozen task geometry; uniform_v3_1 uses the visibility-only forearm aim",
    )
    args = parser.parse_args()
    logical_cpus = os.cpu_count() or 1
    if logical_cpus >= 16:
        split = min(14, logical_cpus // 2)
        control_affinity = tuple(range(0, split))
        perception_affinity = tuple(range(split, logical_cpus))
    elif logical_cpus >= 8:
        split = logical_cpus // 2
        control_affinity = tuple(range(0, split))
        perception_affinity = tuple(range(split, logical_cpus))
    else:
        control_affinity = None
        perception_affinity = None
    scene_suffix = "" if args.scene_version == "formal" else f"_{args.scene_version}"
    leaf = (
        (
            "formal_strict_goal_v4_2c_direct_centervox_certificate"
            + scene_suffix
            + (f"_{args.formal_label}" if args.formal_label else "")
        )
        if args.smoke_cycles is None
        else (
            "smoke_strict_goal_v4_2c_direct_centervox_certificate"
            + scene_suffix
            + (f"_{args.smoke_label}" if args.smoke_label else "")
        )
    )
    output = ROOT / "formal_results" / "final_two_camera" / leaf
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite preserved evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)
    def execute() -> dict:
        return run_one(
            "ellipsoid",
            "mvt_simd",
            output,
        maximum_cycles=args.smoke_cycles,
        unknown_policy="observed_only",
        realtime_pacing=True,
        perception_executor="process",
        centervox_size=0.0075,
        maximum_uncertainty_union_inflation=None,
        certificate_radius_limit=None,
        uncertainty_fusion_mode="fused_certified_centervox",
        camera_width=320,
        camera_height=180,
        camera_pixel_stride=1,
        camera_names=("ur5_depth_wrist", "ur5_depth_forearm"),
        scene_version=args.scene_version,
        record_dense_map_snapshots=True,
        sweep_guard_mode="audit_only",
        success_tolerance=0.001,
        success_hold_cycles=50,
        stop_on_success=False,
            perception_cpu_affinity=perception_affinity,
            control_cpu_affinity=control_affinity,
        )

    process = psutil.Process()
    original_affinity = process.cpu_affinity()
    if control_affinity is not None:
        process.cpu_affinity(list(control_affinity))
    try:
        summary = execute()
    finally:
        process.cpu_affinity(original_affinity)
    (output / "v4_2c_run_complete.json").write_text(
        json.dumps(
            {
                "protocol": "strict_goal_v4_2c_direct_centervox_certificate",
                "formal": args.smoke_cycles is None,
                "mock_data": False,
                "run_name": summary["run_name"],
                "success": summary["success"],
                "ever_sustained_success": summary["ever_sustained_success"],
                "minimum_error_m": summary["minimum_error_m"],
                "final_error_m": summary["final_error_m"],
                "uncertainty_fusion_mode": summary["uncertainty_fusion_mode"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
