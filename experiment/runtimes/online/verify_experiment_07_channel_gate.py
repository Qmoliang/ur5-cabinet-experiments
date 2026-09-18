"""Interpret the Experiment 07.2 drawer cross-section under its own hypothesis."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / "formal_results"
    / "experiment_07"
    / "drawer"
    / "paired_cross_section_sa_et30_initial_smoke"
    / "certificate.json"
)
OUTPUT = ROOT / "formal_results" / "experiment_07" / "drawer_channel_gate_07_2.json"


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    sphere = source["sphere_section"]
    ellipsoid = source["successful_ellipsoid_run_section"]
    checks = {
        "causal_gates_passed": source["causal_gates_passed"],
        "et30_has_strict_open_ball": ellipsoid["has_strict_open_ball"],
        "et30_open_ball_at_least_2mm": ellipsoid["strict_open_ball_radius_m"] >= 0.002,
        "nearest_point_kkt_residual_below_1e-7": ellipsoid["maximum_kkt_residual"] <= 1.0e-7,
        "sphere_open_witness_reported": sphere["open_witness"]["clearance_m"] > 0.0,
        "physical_boundary_farther_than_et30_limit": (
            ellipsoid["opening_boundary_distance_m"]
            > ellipsoid["strict_open_ball_radius_m"]
        ),
    }
    report = {
        "experiment": "07.2",
        "gate": "E7-D4 local continuous cross-section",
        "mock_data": False,
        "source_certificate": str(SOURCE.relative_to(ROOT)),
        "hypothesis": (
            "Experiment 07 does not require sphere closure. It requires a numerically "
            "certified open neighborhood for ET30 and reports SA clearance separately."
        ),
        "scope": (
            "This certifies a continuous open ball in the mandatory drawer-front "
            "cross-section plus causal exact-kernel evidence. It is not a complete "
            "configuration-space path or task-success certificate."
        ),
        "sa_open_witness_clearance_m": sphere["open_witness"]["clearance_m"],
        "et30_strict_open_ball_radius_m": ellipsoid["strict_open_ball_radius_m"],
        "maximum_kkt_residual": ellipsoid["maximum_kkt_residual"],
        "checks": checks,
        "passed": all(checks.values()),
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

