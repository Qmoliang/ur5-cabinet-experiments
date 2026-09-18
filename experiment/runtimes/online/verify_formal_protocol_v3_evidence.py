"""Assemble and verify the final protocol-v3 evidence bundle.

This verifier does not rerun or alter a trajectory.  It joins independent
static, online, post-control replay, acceleration-equivalence, and native-kernel
reports and makes the exact claim boundary machine readable.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
FINAL_ROOT = ROOT / "formal_results" / "final_two_camera"
ONLINE_ROOT = FINAL_ROOT / "formal_evidence_final_v5"
SPHERE_DIR = ONLINE_ROOT / (
    "A-CV-formal-mvt_simd-sphere-cv7.5mm-ui1.25-"
    "cam320x180s1-r70mm-wrist_forearm-observed_only-paced-postsweepaudit"
)
ELLIPSOID_DIR = ONLINE_ROOT / (
    "A-CV-formal-mvt_simd-ellipsoid-cv7.5mm-ui1.25-"
    "cam320x180s1-r70mm-wrist_forearm-observed_only-paced-postsweepaudit"
)
STATIC_REPORT = (
    FINAL_ROOT / "static_geometry_gate_final" / "static_geometry_gate.json"
)
ACCELERATION_REPORT = (
    FINAL_ROOT
    / "known_packed_equivalence_final_v3"
    / "acceleration_equivalence_report.json"
)
SIMD_REPORT = FINAL_ROOT / "final_mvt_simd_benchmark_v5.json"
OUTPUT = FINAL_ROOT / "formal_protocol_v3_evidence_manifest_v5.json"
CONTROL_PERIOD_MS = 20.0


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cycle_statistics(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    compute = np.asarray([float(row["control_compute_ms"]) for row in rows])
    lateness = np.asarray([float(row["deadline_lateness_ms"]) for row in rows])
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    return {
        "cycles": len(rows),
        "cycles_with_normal": sum(int(row["normal_pairs"]) > 0 for row in rows),
        "cycles_with_near": sum(
            int(row["near_penalty_terms"]) > 0 for row in rows
        ),
        "cycles_with_contact_recovery": sum(
            int(row["contact_repulsion_rows"]) > 0 for row in rows
        ),
        "zero_command_fallback_cycles": sum(
            not row["status"].lower().startswith("solved") for row in rows
        ),
        "solver_status_counts": statuses,
        "compute_max_ms": float(np.max(compute)),
        "compute_over_20ms_cycles": int(np.sum(compute > CONTROL_PERIOD_MS)),
        "positive_deadline_lateness_cycles": int(np.sum(lateness > 0.0)),
        "maximum_deadline_lateness_ms": float(np.max(lateness)),
    }


def _online_run(directory: Path) -> dict:
    summary = _read_json(directory / "summary.json")
    candidate = _read_json(directory / "candidate_replay_report.json")
    sweep = _read_json(directory / "post_control_sweep_audit_report.json")
    occupancy = _read_json(directory / "occupancy_replay_report.json")
    cycle_stats = _cycle_statistics(directory / "cycles.csv")
    return {
        "directory": str(directory.resolve()),
        "summary": summary,
        "candidate_replay": candidate,
        "continuous_sweep_replay": sweep,
        "occupancy_replay": occupancy,
        "cycle_statistics": cycle_stats,
        "artifact_hashes": {
            name: _sha256(directory / name)
            for name in (
                "scene.xml",
                "summary.json",
                "cycles.csv",
                "q_history.npy",
                "causal_observability_snapshots.npz",
                "causal_proxy_snapshots.npz",
                "causal_source_configurations.npz",
                "candidate_pairs.csv",
                "pair_states.csv",
                "causal_occupancy_replay_deltas.npz",
                "post_control_sweep_audit.csv",
            )
        },
    }


def main() -> None:
    static = _read_json(STATIC_REPORT)
    acceleration = _read_json(ACCELERATION_REPORT)
    simd = _read_json(SIMD_REPORT)
    runs = {
        "sphere": _online_run(SPHERE_DIR),
        "ellipsoid": _online_run(ELLIPSOID_DIR),
    }
    checks: dict[str, bool] = {}

    def check(name: str, condition: bool) -> None:
        checks[name] = bool(condition)

    sphere = runs["sphere"]["summary"]
    ellipsoid = runs["ellipsoid"]["summary"]

    check("static_geometry_gate", static["passed"])
    check("no_planner_in_static_gate", static["no_path_planner"])
    check("physical_goal_has_collision_free_ik", static["collision_free_goal_ik_count"] >= 1)
    check("same_liuqp_reaches_without_obstacles", static["no_obstacle_liuqp"]["success"])
    check(
        "sphere_continuous_section_strictly_closed",
        static["sphere_cross_section"]["strictly_closed"]
        and static["sphere_cross_section"]["selected"]["continuous_upper_bound_m"]
        <= -static["sphere_cross_section"]["required_closure_margin_m"],
    )
    check(
        "ellipsoid_section_open_with_margin",
        static["ellipsoid_crossing"]["exact_contact_free"]
        and static["ellipsoid_crossing"]["full_pair_min_clearance_m"]
        >= static["required_ellipsoid_open_margin_m"],
    )
    check(
        "matched_static_proxy_coverage",
        static["coverage"]["matched_centers_and_count"]
        and static["coverage"]["sphere_covers_all_cells"]
        and static["coverage"]["ellipsoid_covers_all_cells"],
    )

    check("sphere_fails_frozen_duration", not sphere["success"] and sphere["cycles"] == 2250)
    check("ellipsoid_reaches_single_goal", ellipsoid["success"] and ellipsoid["target_count"] == 1)
    check(
        "same_online_source_hashes",
        sphere["source_hashes"] == ellipsoid["source_hashes"],
    )
    check(
        "same_online_scene",
        runs["sphere"]["artifact_hashes"]["scene.xml"]
        == runs["ellipsoid"]["artifact_hashes"]["scene.xml"]
        == static["hashes"]["scene_xml_sha256"],
    )
    check(
        "no_online_cheat_inputs",
        all(
            item["path_planner"] is None
            and item["intermediate_targets"] is None
            and item["random_dither"] is False
            and item["truth_collision_backtracking"] is False
            and item["target_count"] == 1
            for item in (sphere, ellipsoid)
        ),
    )
    check(
        "two_local_moving_cameras_and_no_global_map",
        all(
            item["camera_count"] == 2
            and item["local_moving_cameras"]
            and item["incremental_free_occupied_unknown_map"]
            and not item["global_map_preloaded"]
            and not item["future_frames_used"]
            for item in (sphere, ellipsoid)
        ),
    )
    check(
        "online_observability_deadlines",
        all(
            item["observability_passed"]
            and item["observability_late_samples"] == 0
            and item["observability_never_observed_samples"] == 0
            for item in (sphere, ellipsoid)
        ),
    )
    check(
        "centervox_coverage",
        sphere["all_centervox_coverage_checks_passed"]
        and ellipsoid["all_centervox_coverage_checks_passed"],
    )
    check(
        "online_mvt_oracle",
        sphere["all_mvt_oracle_checks_passed"]
        and ellipsoid["all_mvt_oracle_checks_passed"],
    )
    check(
        "zero_exact_mujoco_penetrations",
        all(
            item["exact_penetrating_cycles"] == 0
            and item["maximum_penetration_m"] == 0.0
            for item in (sphere, ellipsoid)
        ),
    )
    check(
        "liuqp_command_not_modified_online",
        all(
            item["sweep_guard_mode"] == "post_control_audit"
            and item["sweep_guard_modified_cycles"] == 0
            and item["liu_qp_command_executed_without_guard_modification"]
            for item in (sphere, ellipsoid)
        ),
    )

    for representation, run in runs.items():
        candidate = run["candidate_replay"]
        sweep = run["continuous_sweep_replay"]
        occupancy = run["occupancy_replay"]
        check(
            f"{representation}_candidate_replay",
            candidate["passed"]
            and candidate["all_per_cycle_candidate_counts_matched"]
            and not candidate["online_control_or_publication_timing_affected"],
        )
        check(
            f"{representation}_continuous_sweep_replay",
            sweep["passed"]
            and sweep["unsafe_cycles"] == 0
            and not sweep["liu_qp_commands_modified"]
            and not sweep["online_control_or_publication_timing_affected"],
        )
        check(
            f"{representation}_occupancy_replay",
            occupancy["passed"]
            and occupancy["all_online_frame_state_counts_matched"]
            and not occupancy["online_control_or_publication_timing_affected"],
        )
        check(
            f"{representation}_three_states_exercised",
            all(
                run["cycle_statistics"][key] > 0
                for key in (
                    "cycles_with_normal",
                    "cycles_with_near",
                    "cycles_with_contact_recovery",
                )
            ),
        )

    check("acceleration_exact_equivalence", acceleration["passed"])
    check(
        "true_multilevel_mvt_avx2",
        simd["implementation"]["instruction_set"] == "AVX2"
        and simd["implementation"]["simd_width_float_lanes"] == 8
        and simd["implementation"]["mvt_levels"] == 5
        and simd["implementation"]["cell_lookups_per_query"] == 135
        and simd["primary_controller_proxy_workload"][
            "scalar_and_avx2_equal_full_scan_oracle"
        ]
        and simd["centervox_stress_kernel_only"][
            "scalar_and_avx2_equal_full_scan_oracle"
        ],
    )
    check(
        "frozen_p99_realtime_target",
        sphere["control_compute_ms_p99"] <= CONTROL_PERIOD_MS
        and ellipsoid["control_compute_ms_p99"] <= CONTROL_PERIOD_MS,
    )

    rare_deadline_outliers = {
        representation: {
            key: run["cycle_statistics"][key]
            for key in (
                "compute_max_ms",
                "compute_over_20ms_cycles",
                "positive_deadline_lateness_cycles",
                "maximum_deadline_lateness_ms",
            )
        }
        for representation, run in runs.items()
    }
    hard_realtime_all_cycles = all(
        values["compute_over_20ms_cycles"] == 0
        for values in rare_deadline_outliers.values()
    )
    required_checks_passed = all(checks.values())
    result = {
        "protocol": "formal_protocol_v3_final_two_camera",
        "passed": required_checks_passed,
        "combined_post_audit_evidence_eligible": required_checks_passed,
        "claim_boundary": {
            "representation_claim": (
                "Matched conservative spheres close a required continuous section; "
                "matched ellipsoids preserve it and the unmodified final-goal-only "
                "ellipsoid LiuQP reaches the target."
            ),
            "generality": (
                "This supports mitigation of sphere-induced false closure in this "
                "registered scene, not global completeness or elimination of all "
                "nonconvex local minima."
            ),
            "realtime": (
                "The frozen p99 <= 20 ms soft-real-time target passes. Rare Windows "
                "scheduling/geometry outliers remain, so hard per-cycle real time is "
                "not claimed."
            ),
            "unobserved_space": (
                "Unknown voxels are maintained and displayed but are monitor-only; "
                "the result does not prove safety in arbitrary unseen space."
            ),
        },
        "checks": checks,
        "hard_realtime_all_cycles_passed": hard_realtime_all_cycles,
        "rare_deadline_outliers": rare_deadline_outliers,
        "static_geometry": static,
        "online_runs": runs,
        "acceleration_equivalence": acceleration,
        "native_mvt_simd_benchmark": simd,
        "evidence_report_hashes": {
            "static_geometry_gate": _sha256(STATIC_REPORT),
            "acceleration_equivalence": _sha256(ACCELERATION_REPORT),
            "native_mvt_simd_benchmark": _sha256(SIMD_REPORT),
            "verifier_source": _sha256(Path(__file__)),
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "combined_post_audit_evidence_eligible": result[
                    "combined_post_audit_evidence_eligible"
                ],
                "checks": checks,
                "hard_realtime_all_cycles_passed": hard_realtime_all_cycles,
                "rare_deadline_outliers": rare_deadline_outliers,
                "output": str(OUTPUT.resolve()),
            },
            indent=2,
        )
    )
    if not required_checks_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
