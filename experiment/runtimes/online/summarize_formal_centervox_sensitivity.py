"""Aggregate the final 5/7.5/10 mm CenterVox sensitivity runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "formal_results" / "final_two_camera"
MAIN = FINAL / "formal_evidence_final_v5"
SENSITIVITY = FINAL / "sensitivity_centervox_final"
OUTPUT_JSON = SENSITIVITY / "centervox_sensitivity_summary.json"
OUTPUT_CSV = SENSITIVITY / "centervox_sensitivity_summary.csv"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _directory(size_mm: float, representation: str) -> Path:
    root = MAIN if size_mm == 7.5 else SENSITIVITY
    token = f"cv{size_mm:g}mm"
    matches = [
        item
        for item in root.iterdir()
        if item.is_dir()
        and f"-{representation}-" in item.name
        and token in item.name
        and item.name.endswith("postsweepaudit")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {size_mm:g} mm {representation} run, found {matches}"
        )
    return matches[0]


def _row(size_mm: float, representation: str) -> dict:
    directory = _directory(size_mm, representation)
    summary = _json(directory / "summary.json")
    candidate = _json(directory / "candidate_replay_report.json")
    sweep = _json(directory / "post_control_sweep_audit_report.json")
    occupancy = _json(directory / "occupancy_replay_report.json")
    causal_safety = bool(
        summary["observability_passed"]
        and summary["all_centervox_coverage_checks_passed"]
        and summary["all_mvt_oracle_checks_passed"]
        and summary["exact_penetrating_cycles"] == 0
        and candidate["passed"]
        and sweep["passed"]
        and occupancy["passed"]
    )
    return {
        "centervox_size_mm": size_mm,
        "representation": representation,
        "success": summary["success"],
        "first_success_s": summary["first_success_time_s"],
        "final_error_mm": 1000.0 * summary["final_error_m"],
        "cycles": summary["cycles"],
        "final_center_voxels": summary["final_center_voxels"],
        "final_proxy_count": summary["final_proxy_count"],
        "published_frames": summary["published_perception_frames"],
        "effective_published_perception_hz_wall": summary[
            "effective_published_perception_hz_wall"
        ],
        "perception_pipeline_p99_ms": summary["perception_pipeline_ms_p99"],
        "snapshot_age_p95_ms": summary["snapshot_age_ms_p95"],
        "control_compute_p99_ms": summary["control_compute_ms_p99"],
        "deadline_misses": summary["deadline_misses"],
        "observability_passed": summary["observability_passed"],
        "observability_late_samples": summary["observability_late_samples"],
        "coverage_passed": summary["all_centervox_coverage_checks_passed"],
        "mvt_oracle_passed": summary["all_mvt_oracle_checks_passed"],
        "exact_penetrating_cycles": summary["exact_penetrating_cycles"],
        "candidate_replay_passed": candidate["passed"],
        "candidate_pair_rows": candidate["candidate_pair_rows"],
        "continuous_sweep_passed": sweep["passed"],
        "continuous_sweep_unsafe_cycles": sweep["unsafe_cycles"],
        "continuous_sweep_minimum_clearance_mm": (
            None
            if sweep["minimum_clearance_m"] is None
            else 1000.0 * sweep["minimum_clearance_m"]
        ),
        "occupancy_replay_passed": occupancy["passed"],
        "causal_safety_evidence_eligible": causal_safety,
        "frozen_p99_realtime_passed": summary["control_compute_ms_p99"] <= 20.0,
        "directory": str(directory.resolve()),
    }


def main() -> None:
    rows = [
        _row(size_mm, representation)
        for size_mm in (5.0, 7.5, 10.0)
        for representation in ("sphere", "ellipsoid")
    ]
    eligible_ellipsoid_successes = [
        row
        for row in rows
        if row["representation"] == "ellipsoid"
        and row["success"]
        and row["causal_safety_evidence_eligible"]
        and row["frozen_p99_realtime_passed"]
    ]
    result = {
        "experiment": "formal_centervox_resolution_and_proxy_count_sensitivity",
        "fixed_variables": (
            "scene, q0, target, QP, robot certificate, two moving cameras, "
            "320x180 rays, MVT-AVX2, uncertainty-union limit 1.25, no planner, "
            "no dither, no truth backtracking, post-control sweep audit"
        ),
        "rows": rows,
        "only_eligible_ellipsoid_success_size_mm": [
            row["centervox_size_mm"] for row in eligible_ellipsoid_successes
        ],
        "interpretation": {
            "5mm": (
                "Denser representatives increase proxy construction latency. The "
                "ellipsoid reaches but misses observability deadlines; the sphere "
                "has persistent worsening conservative overlaps in sweep replay."
            ),
            "7.5mm": (
                "Only tested setting where the ellipsoid reaches, observability and "
                "all post audits pass, and control p99 is within 20 ms."
            ),
            "10mm": (
                "Coverage still passes, but larger conservative residuals expand "
                "candidate/active pairs; the ellipsoid fails and its p99 exceeds 20 ms."
            ),
        },
    }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
