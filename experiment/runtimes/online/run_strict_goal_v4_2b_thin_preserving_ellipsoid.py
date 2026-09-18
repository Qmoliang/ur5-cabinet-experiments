"""Run strict-goal v4.2b with certified thin-axis-preserving fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_protocol_v3_async_online import run_one


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-cycles", type=int, default=None)
    parser.add_argument("--smoke-label", default="")
    args = parser.parse_args()
    leaf = (
        "formal_strict_goal_v4_2b_thin_preserving_certificate"
        if args.smoke_cycles is None
        else (
            "smoke_strict_goal_v4_2b_thin_preserving_certificate"
            + (f"_{args.smoke_label}" if args.smoke_label else "")
        )
    )
    output = ROOT / "formal_results" / "final_two_camera" / leaf
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite preserved evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = run_one(
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
        uncertainty_fusion_mode="fused_certified_ellipsoid",
        camera_width=320,
        camera_height=180,
        camera_pixel_stride=1,
        camera_names=("ur5_depth_wrist", "ur5_depth_forearm"),
        scene_version="formal",
        record_dense_map_snapshots=True,
        sweep_guard_mode="post_control_audit",
        success_tolerance=0.001,
        success_hold_cycles=50,
        stop_on_success=False,
    )
    (output / "v4_2b_run_complete.json").write_text(
        json.dumps(
            {
                "protocol": "strict_goal_v4_2b_thin_preserving_certificate",
                "formal": args.smoke_cycles is None,
                "mock_data": False,
                "run_name": summary["run_name"],
                "success": summary["success"],
                "ever_sustained_success": summary["ever_sustained_success"],
                "minimum_error_m": summary["minimum_error_m"],
                "final_error_m": summary["final_error_m"],
                "uncertainty_fusion_mode": summary["uncertainty_fusion_mode"],
                "maximum_thin_axis_inflation_per_fusion": 1.10,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
