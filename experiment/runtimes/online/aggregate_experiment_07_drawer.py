"""Aggregate the completed Experiment 07A formal drawer matrix."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "formal_results" / "experiment_07"
TABLE_ROOT = ROOT / "tables" / "experiment_07"
GROUPS = ("S", "SA", "E0", "ET30")


def latest_formal_batch(group: str) -> Path:
    found = []
    for batch in (RESULT_ROOT / "drawer").glob("formal_*"):
        manifest_path = batch / "batch_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("mode") == "formal"
            and manifest.get("groups") == [group]
            and manifest.get("repeats") == 5
        ):
            found.append(batch)
    if not found:
        raise FileNotFoundError(group)
    return max(found, key=lambda path: path.stat().st_mtime_ns)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - half, center + half


def main() -> None:
    rows = []
    source_hashes = {}
    for group in GROUPS:
        batch = latest_formal_batch(group)
        source_hashes[group] = []
        for run_root in sorted((batch / group).glob("run_*")):
            leaf = next(path for path in run_root.iterdir() if path.is_dir())
            summary = json.loads((leaf / "summary.json").read_text(encoding="utf-8"))
            counts = summary["final_mvt_level_proxy_counts"]
            dominant = max(counts) / sum(counts)
            source_hashes[group].append(summary["source_hashes"])
            rows.append(
                {
                    "scene": "drawer",
                    "group": group,
                    "seed": int(run_root.name.split("_")[-1]),
                    "success": summary["success"],
                    "evidence_eligible": summary["evidence_eligible"],
                    "final_error_m": summary["final_error_m"],
                    "minimum_error_m": summary["minimum_error_m"],
                    "steady_error_p95_m": summary["steady_error_p95_m"],
                    "maximum_success_hold_cycles": summary["maximum_success_hold_cycles"],
                    "final_proxy_count": summary["final_proxy_count"],
                    "core_axis_ratio_p95": summary["final_ellipsoid_core_axis_ratio_p95"],
                    "mvt_level_counts": json.dumps(counts),
                    "mvt_dominant_share": dominant,
                    "mvt_structure_passed": (
                        None
                        if group == "S"
                        else sum(count / sum(counts) >= 0.05 for count in counts) >= 2
                        and dominant <= 0.90
                    ),
                    "mvt_oracle_passed": summary["all_mvt_oracle_checks_passed"],
                    "coverage_passed": summary["all_centervox_coverage_checks_passed"],
                    "observability_passed": summary["observability_passed"],
                    "observability_late_samples": summary["observability_late_samples"],
                    "observability_never_observed_samples": summary[
                        "observability_never_observed_samples"
                    ],
                    "exact_penetrating_cycles": summary["exact_penetrating_cycles"],
                    "deadline_misses": summary["deadline_misses"],
                    "controller_ms_p99": summary["controller_ms_p99"],
                    "realtime_controller_passed": summary["realtime_controller_passed"],
                    "initial_configuration_sha256": summary[
                        "initial_configuration_sha256"
                    ],
                    "run": str(leaf.relative_to(ROOT)),
                }
            )
    reference_hashes = source_hashes["S"][0]
    sources_equal = all(
        hashes == reference_hashes
        for group_hashes in source_hashes.values()
        for hashes in group_hashes
    )
    groups = []
    for group in GROUPS:
        subset = [row for row in rows if row["group"] == group]
        successes = sum(row["success"] for row in subset)
        eligible = sum(row["evidence_eligible"] for row in subset)
        lower, upper = wilson(successes, len(subset))
        groups.append(
            {
                "group": group,
                "runs": len(subset),
                "successes": successes,
                "success_rate": successes / len(subset),
                "success_wilson_95": [lower, upper],
                "evidence_eligible_runs": eligible,
                "final_error_median_m": median(row["final_error_m"] for row in subset),
                "final_error_mean_m": mean(row["final_error_m"] for row in subset),
                "final_proxy_count_median": median(
                    row["final_proxy_count"] for row in subset
                ),
                "deadline_misses_total": sum(row["deadline_misses"] for row in subset),
                "penetrating_cycles_total": sum(
                    row["exact_penetrating_cycles"] for row in subset
                ),
                "observability_passed_runs": sum(
                    row["observability_passed"] for row in subset
                ),
                "realtime_passed_runs": sum(
                    row["realtime_controller_passed"] for row in subset
                ),
                "mvt_structure_passed_runs": sum(
                    row["mvt_structure_passed"] is True for row in subset
                ),
                "coverage_and_oracle_all_passed": all(
                    row["coverage_passed"] and row["mvt_oracle_passed"]
                    for row in subset
                ),
            }
        )
    report = {
        "experiment": "07.2",
        "gate": "E7-D6 drawer formal matrix",
        "mock_data": False,
        "run_count": len(rows),
        "all_groups_five_runs": len(rows) == 20,
        "source_hashes_equal_across_all_runs": sources_equal,
        "groups": groups,
        "task_outcome": (
            "E0 succeeds in 2/5 seeds; ET30, SA and S succeed in 0/5. "
            "The grounded drawer is therefore not a completed robust scene."
        ),
        "passed_execution_and_reporting_gate": len(rows) == 20 and sources_equal,
    }
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    with (TABLE_ROOT / "T27_drawer_formal_runs_07_2.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULT_ROOT / "drawer_formal_summary_07_2.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed_execution_and_reporting_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


