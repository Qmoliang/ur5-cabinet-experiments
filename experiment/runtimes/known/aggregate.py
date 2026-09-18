"""Aggregate only real logs from the two known-volume main runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    rows = []
    details = {}
    for case in ("sphere", "ellipsoid"):
        run_dir = ROOT / "results" / case
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        coverage = json.loads((run_dir / "coverage.json").read_text(encoding="utf-8"))
        with (run_dir / "cycles.csv").open(encoding="utf-8") as handle:
            cycles = list(csv.DictReader(handle))
        solved = sum(item["status"].startswith("solved") for item in cycles)
        non_solved = len(cycles) - solved
        row = {
            "representation": case,
            "proxy_count": summary["proxy_count"],
            "complete_solid_volume_covered": coverage["complete_solid_cell_volume_covered"],
            "maximum_corner_measure": coverage["maximum_corner_measure"],
            "primitive_to_box_volume_sum_ratio": summary["primitive_to_box_volume_sum_ratio"],
            "success": summary["success"],
            "final_error_mm": 1000.0 * summary["final_error_m"],
            "minimum_error_mm": 1000.0 * summary["minimum_error_m"],
            "maximum_ee_x_m": summary["maximum_ee_x_m"],
            "solved_cycles": solved,
            "non_solved_cycles": non_solved,
            "exact_penetrating_cycles": summary["exact_penetrating_cycles"],
            "controller_ms_p50": summary["controller_ms_p50"],
            "controller_ms_p95": summary["controller_ms_p95"],
            "controller_ms_p99": summary["controller_ms_p99"],
            "candidate_pairs_p50": summary["broadphase_candidate_pairs_p50"],
            "candidate_pairs_p99": summary["broadphase_candidate_pairs_p99"],
            "qp_iterations_p50": summary["qp_iterations_p50"],
            "qp_iterations_p99": summary["qp_iterations_p99"],
        }
        rows.append(row)
        details[case] = row
    output = ROOT / "tables" / "results.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    robust = json.loads(
        (ROOT / "results" / "sphere_same_qp_slsqp_validation" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    comparison = {
        "main_runs": details,
        "sphere_same_qp_robust_solver_validation": robust,
        "interpretation_gate": {
            "same_solid_partition": True,
            "both_full_volume_coverage_audits_passed": all(
                row["complete_solid_volume_covered"] for row in rows
            ),
            "both_no_physical_penetration": all(
                row["exact_penetrating_cycles"] == 0 for row in rows
            ),
            "sphere_osqp_status_is_a_confounder_in_main_run": details["sphere"]["non_solved_cycles"] > 0,
            "robust_same_qp_validation_remained_blocked": not robust["success"],
        },
    }
    (ROOT / "results" / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
