"""Run Experiment 07 drawer or birdcage groups under one frozen contract."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys

import numpy as np
from pathlib import Path

from run_protocol_v3_async_online import _protocol_scene, run_one


ROOT = Path(__file__).resolve().parent
GROUPS = {
    "S": {"representation": "sphere", "minimum_core_semi_axis": 0.0, "sphere_cover_mode": "cap_irredundant"},
    "SA": {"representation": "sphere", "minimum_core_semi_axis": 0.0, "sphere_cover_mode": "adaptive_irredundant"},
    "E0": {"representation": "ellipsoid", "minimum_core_semi_axis": 0.0},
    "ET3": {"representation": "ellipsoid", "minimum_core_semi_axis": 0.00375},
    "ET30": {"representation": "ellipsoid", "minimum_core_semi_axis": 0.00300},
    "ET": {"representation": "ellipsoid", "minimum_core_semi_axis": 0.0075},
}
MAIN_GROUPS = ("S", "SA", "E0", "ET30")
FORMAL_INITIAL_OFFSETS_RAD = np.asarray(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.010, -0.010, 0.008, 0.0, 0.0, 0.0],
        [-0.010, 0.008, -0.006, 0.0, 0.0, 0.0],
        [0.0, 0.012, -0.010, 0.006, 0.0, 0.0],
        [0.0, -0.012, 0.010, -0.006, 0.0, 0.0],
    ],
    dtype=float,
)
SCENES = {"drawer": "exp07_drawer", "birdcage": "exp07_birdcage"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=tuple(SCENES), default="drawer")
    parser.add_argument("--group", choices=("all", *GROUPS), default="all")
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    if args.cycles <= 0:
        parser.error("--cycles must be positive")
    repeats = args.repeats if args.repeats is not None else (1 if args.mode == "smoke" else 5)
    if args.mode == "formal" and repeats != len(FORMAL_INITIAL_OFFSETS_RAD):
        parser.error("formal experiment 07 requires exactly five frozen initial seeds")
    if repeats <= 0 or repeats > len(FORMAL_INITIAL_OFFSETS_RAD):
        parser.error("--repeats must be between one and five")

    groups = MAIN_GROUPS if args.group == "all" else (args.group,)
    scene_q0 = np.asarray(_protocol_scene(SCENES[args.scene]).q0, dtype=float)
    initial_configurations = scene_q0[None, :] + FORMAL_INITIAL_OFFSETS_RAD

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = f"{args.mode}_{stamp}" + (f"_{args.label}" if args.label else "")
    batch_root = ROOT / "formal_results" / "experiment_07" / args.scene / batch_name
    if batch_root.exists():
        raise FileExistsError(f"refusing to overwrite experiment 07 batch: {batch_root}")
    batch_root.mkdir(parents=True)

    results: list[dict] = []
    for group in groups:
        settings = GROUPS[group]
        for repeat in range(repeats):
            run_root = batch_root / group / f"run_{repeat:02d}"
            run_root.mkdir(parents=True)
            summary = run_one(
                settings["representation"],
                "mvt_simd",
                run_root,
                maximum_cycles=(args.cycles if args.mode == "smoke" else None),
                unknown_policy="observed_only",
                realtime_pacing=args.mode == "formal",
                perception_executor=("thread" if args.mode == "smoke" else "process"),
                centervox_size=0.0075,
                maximum_uncertainty_union_inflation=1.05,
                certificate_radius_limit=0.070,
                uncertainty_fusion_mode="separate_uncertainty",
                direct_thin_axis_inflation=1.0,
                minimum_core_semi_axis=settings["minimum_core_semi_axis"],
                direct_tangent_subdivisions=1,
                direct_partition_mode="grid",
                sphere_cover_mode=settings.get("sphere_cover_mode", "matched"),
                ellipsoid_cover_mode=("adaptive_irredundant" if settings["representation"] == "ellipsoid" else "matched"),
                radius_limit_representation=settings["representation"],
                camera_width=160,
                camera_height=90,
                camera_pixel_stride=1,
                camera_names=("ur5_depth_wrist", "ur5_depth_forearm"),
                scene_version=SCENES[args.scene],
                record_dense_map_snapshots=True,
                sweep_guard_mode="audit_only",
                success_tolerance=0.001,
                success_hold_cycles=50,
                stop_on_success=False,
                ellipsoid_pair_threads=6,
                control_process_priority="normal",
                perception_process_priority="normal",
                task_gain=2.0,
                max_task_speed=0.18,
                osqp_adaptive_row_threshold=512,
                osqp_tolerance=1.0e-4,
                osqp_max_iterations=1000,
                initial_configuration=initial_configurations[repeat],
            )
            results.append(
                {
                    "group": group,
                    "repeat": repeat,
                    "initial_seed": repeat,
                    "initial_configuration_rad": summary["initial_configuration_rad"],
                    "initial_configuration_sha256": summary[
                        "initial_configuration_sha256"
                    ],
                    "run_name": summary["run_name"],
                    "cycles": summary["cycles"],
                    "success": summary["success"],
                    "final_error_m": summary["final_error_m"],
                    "final_proxy_count": summary["final_proxy_count"],
                    "minimum_core_semi_axis_m": summary["minimum_core_semi_axis_m"],
                    "final_ellipsoid_core_minimum_semi_axis_m": summary[
                        "final_ellipsoid_core_minimum_semi_axis_m"
                    ],
                    "final_ellipsoid_core_axis_ratio_p95": summary[
                        "final_ellipsoid_core_axis_ratio_p95"
                    ],
                    "all_centervox_coverage_checks_passed": summary[
                        "all_centervox_coverage_checks_passed"
                    ],
                    "all_mvt_oracle_checks_passed": summary[
                        "all_mvt_oracle_checks_passed"
                    ],
                    "final_mvt_level_proxy_counts": summary[
                        "final_mvt_level_proxy_counts"
                    ],
                    "final_mvt_nonempty_level_count": summary[
                        "final_mvt_nonempty_level_count"
                    ],
                    "deadline_misses": summary["deadline_misses"],
                }
            )

    manifest = {
        "experiment": "07",
        "mock_data": False,
        "scene": args.scene,
        "mode": args.mode,
        "groups": list(groups),
        "repeats": repeats,
        "protocol": "plan/experiment-07-protocol.md",
        "command_line": [sys.executable, *sys.argv],
        "formal_initial_offsets_rad": FORMAL_INITIAL_OFFSETS_RAD.tolist(),
        "formal_seed_count": len(FORMAL_INITIAL_OFFSETS_RAD),
        "results": results,
    }
    (batch_root / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()




