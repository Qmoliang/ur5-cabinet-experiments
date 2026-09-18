"""Final integrity gate for the v2 freeze and v3.3 uniform-radius delivery."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
FINAL_ROOT = ROOT / "formal_results" / "final_two_camera"
FORMAL = FINAL_ROOT / "formal_uniform_radius_v3_3"
INVALID = FINAL_ROOT / "formal_uniform_radius_v3_3_no_ui_invalid"
V2_MANIFEST = FINAL_ROOT / "protocol_v2_adaptive_radius_frozen" / "frozen_manifest.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def verify() -> dict:
    run_dirs = sorted(path for path in FORMAL.iterdir() if path.is_dir())
    if len(run_dirs) != 6:
        raise AssertionError(f"expected six formal run directories, found {len(run_dirs)}")
    invalid_dirs = sorted(path for path in INVALID.iterdir() if path.is_dir())
    if len(invalid_dirs) != 6:
        raise AssertionError("invalid no-union-limit pilot is not preserved as six runs")

    groups = {}
    source_hashes = None
    scene_hash = None
    for run_dir in run_dirs:
        summary = _json(run_dir / "summary.json")
        representation = summary["representation"]
        scale = int(round(summary["certificate_radius_limit_m"] * 1000.0))
        group = ("S" if representation == "sphere" else "E") + str(scale)
        if group in groups:
            raise AssertionError(f"duplicate group {group}")
        if summary["cycles"] != 2250 or summary["simulated_duration_s"] != 45.0:
            raise AssertionError(f"{group} did not complete the frozen duration")
        if summary["maximum_uncertainty_union_inflation_limit"] != 1.25:
            raise AssertionError(f"{group} omitted the frozen 1.25 union limit")
        if not (
            summary["all_proxy_radius_limit_checks_passed"]
            and summary["all_centervox_coverage_checks_passed"]
            and summary["all_mvt_oracle_checks_passed"]
            and summary["exact_penetrating_cycles"] == 0
            and summary["path_planner"] is None
            and summary["intermediate_targets"] is None
            and summary["random_dither"] is False
            and summary["truth_collision_backtracking"] is False
            and summary["robot_certificate_spheres"] == 65
        ):
            raise AssertionError(f"{group} violated a frozen safety/no-cheat invariant")
        if source_hashes is None:
            source_hashes = summary["source_hashes"]
            scene_hash = _sha256(run_dir / "scene.xml")
        elif summary["source_hashes"] != source_hashes or _sha256(
            run_dir / "scene.xml"
        ) != scene_hash:
            raise AssertionError(f"{group} source or scene hash differs")
        with (run_dir / "perception_frames.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            frames = list(csv.DictReader(stream))
        if not frames or not all(
            _truth(row["certificate_radius_limit_ok"])
            and _truth(row["coverage_ok"])
            and int(row["mvt_missing_candidates"]) == 0
            for row in frames
        ):
            raise AssertionError(f"{group} has a failed per-frame certificate gate")
        candidate = _json(run_dir / "candidate_replay_report.json")
        sweep = _json(run_dir / "post_control_sweep_audit_report.json")
        occupancy = _json(run_dir / "recorded_occupancy_replay_report.json")
        section = _json(run_dir / "online_mandatory_section_audit.json")
        if not (
            candidate["passed"]
            and candidate["all_per_cycle_candidate_counts_matched"]
            and sweep["passed"]
            and sweep["unsafe_cycles"] == 0
            and occupancy["passed"]
            and occupancy["cycles_match_source_proxy_and_frame_logs"]
        ):
            raise AssertionError(f"{group} failed a post-control evidence replay")
        groups[group] = {
            "success": bool(summary["success"]),
            "minimum_error_mm": summary["minimum_error_m"] * 1000.0,
            "control_p99_ms": summary["control_compute_ms_p99"],
            "observability_passed": bool(summary["observability_passed"]),
            "sphere_closed": bool(
                section["sphere_certificate_section"]["continuously_closed"]
            ),
            "ellipsoid_open": bool(
                section["ellipsoid_q_plus_u_section"]["certified_open"]
            ),
        }

    expected = {"S30", "E30", "S50", "E50", "S70", "E70"}
    if set(groups) != expected:
        raise AssertionError(f"formal group set differs: {set(groups)}")
    if any(row["success"] for row in groups.values()):
        raise AssertionError("aggregate expectation changed: a run now reports success")
    if groups["S30"]["sphere_closed"]:
        raise AssertionError("S30 must not be labelled continuously closed")
    if not groups["S50"]["sphere_closed"] or not groups["S70"]["sphere_closed"]:
        raise AssertionError("S50/S70 continuous sphere closure evidence is missing")
    if not all(groups[group]["ellipsoid_open"] for group in ("E30", "E50", "E70")):
        raise AssertionError("an ellipsoid group lost its certified section opening")
    if groups["E70"]["control_p99_ms"] > 20.0:
        raise AssertionError("E70 no longer passes the 20 ms control p99 gate")

    aggregate = _json(FORMAL / "uniform_radius_results.json")
    if len(aggregate["rows"]) != 6 or aggregate["mock_data"] is not False:
        raise AssertionError("aggregate result is incomplete or marked mock")
    v2 = _json(V2_MANIFEST)
    v2_files = v2.get("files", [])
    if len(v2_files) != 100:
        raise AssertionError(f"v2 frozen manifest expected 100 files, found {len(v2_files)}")
    for run_dir in invalid_dirs:
        summary = _json(run_dir / "summary.json")
        if summary["maximum_uncertainty_union_inflation_limit"] is not None:
            raise AssertionError("invalid pilot archive does not contain the documented omission")

    excluded = {"frozen_manifest.json", "delivery_verification.json"}
    evidence_files = sorted(
        path for path in FORMAL.rglob("*") if path.is_file() and path.name not in excluded
    )
    frozen = {
        "protocol": "uniform_radius_v3_3",
        "files": [
            {
                "path": str(path.relative_to(FORMAL)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in evidence_files
        ],
    }
    (FORMAL / "frozen_manifest.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )
    result = {
        "delivery_integrity_passed": True,
        "scientific_target_ellipsoid_success_achieved": False,
        "allowed_positive_claim": (
            "S50/S70 actual online sphere certificates continuously close the "
            "mandatory section while the corresponding Q+U ellipsoid snapshots "
            "retain certified openings"
        ),
        "forbidden_claim": "the v3.3 uniform-radius ellipsoid online controller succeeded",
        "groups": groups,
        "formal_evidence_files_hashed": len(evidence_files),
        "formal_evidence_bytes_hashed": int(sum(path.stat().st_size for path in evidence_files)),
        "v2_frozen_files": len(v2_files),
        "invalid_pilot_runs_preserved": len(invalid_dirs),
        "scene_sha256": scene_hash,
    }
    (FORMAL / "delivery_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
