"""Run the matched protocol-v4 strict Cartesian-goal comparison.

Every group uses the same current source tree, formal scene, stride-1 dual
moving-camera perception, CenterVox/MVT/SIMD stack, and 45 s fixed duration.
The only experimental factors are representation and certificate scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from run_protocol_v3_async_online import run_one


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT
    / "formal_results"
    / "final_two_camera"
    / "formal_strict_goal_v4"
)
SCALES = (None, 0.030, 0.050, 0.070)
REPRESENTATIONS = ("sphere", "ellipsoid")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    batch_started = perf_counter()
    completed: list[dict] = []
    for radius in SCALES:
        for representation in REPRESENTATIONS:
            label = "adaptive" if radius is None else f"{radius * 1000.0:g}mm"
            print(f"START {representation} {label}", flush=True)
            summary = run_one(
                representation,
                "mvt_simd",
                OUTPUT,
                unknown_policy="observed_only",
                realtime_pacing=True,
                perception_executor="process",
                centervox_size=0.0075,
                maximum_uncertainty_union_inflation=1.25,
                certificate_radius_limit=radius,
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
            completed.append(
                {
                    "representation": representation,
                    "certificate_radius_limit_m": radius,
                    "run_name": summary["run_name"],
                    "success": summary["success"],
                    "ever_sustained_success": summary["ever_sustained_success"],
                    "minimum_error_m": summary["minimum_error_m"],
                    "final_error_m": summary["final_error_m"],
                }
            )
            (OUTPUT / "batch_progress.json").write_text(
                json.dumps(
                    {
                        "protocol": "strict_goal_v4",
                        "mock_data": False,
                        "completed": completed,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(
                f"DONE {representation} {label}: "
                f"min={summary['minimum_error_m'] * 1000.0:.6f} mm, "
                f"final={summary['final_error_m'] * 1000.0:.6f} mm, "
                f"success={summary['success']}",
                flush=True,
            )
    (OUTPUT / "batch_complete.json").write_text(
        json.dumps(
            {
                "protocol": "strict_goal_v4",
                "mock_data": False,
                "wall_duration_s": perf_counter() - batch_started,
                "completed": completed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
