"""Rerun only strict adaptive ellipsoid after the LiuQP contact fix.

The frozen v4 directory is never touched.  Every other experimental parameter
matches ``run_strict_goal_v4_batch.py`` exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

from run_protocol_v3_async_online import run_one


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT
    / "formal_results"
    / "final_two_camera"
    / "formal_strict_goal_v4_1_original_contact_semantics"
)


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite preserved v4.1 evidence: {OUTPUT}"
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = run_one(
        "ellipsoid",
        "mvt_simd",
        OUTPUT,
        unknown_policy="observed_only",
        realtime_pacing=True,
        perception_executor="process",
        centervox_size=0.0075,
        maximum_uncertainty_union_inflation=1.25,
        certificate_radius_limit=None,
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
    (OUTPUT / "v4_1_run_complete.json").write_text(
        json.dumps(
            {
                "protocol": "strict_goal_v4_1_original_contact_semantics",
                "mock_data": False,
                "preserved_v4": str(
                    ROOT
                    / "formal_results"
                    / "final_two_camera"
                    / "formal_strict_goal_v4"
                ),
                "run_name": summary["run_name"],
                "success": summary["success"],
                "ever_sustained_success": summary["ever_sustained_success"],
                "minimum_error_m": summary["minimum_error_m"],
                "final_error_m": summary["final_error_m"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
