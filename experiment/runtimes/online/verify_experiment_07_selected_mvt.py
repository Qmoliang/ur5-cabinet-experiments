"""Audit effective MVT occupancy for Experiment 07.2 frozen groups."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "formal_results" / "experiment_07"
TABLE_ROOT = ROOT / "tables" / "experiment_07"
GROUPS = ("S", "SA", "E0", "ET30")
HIERARCHY_GROUPS = ("SA", "E0", "ET30")


def latest_complete_batch(scene: str) -> Path:
    for path in sorted((RESULT_ROOT / scene).glob("smoke_*"), reverse=True):
        manifest_path = path / "batch_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("groups") == list(GROUPS):
            return path
    raise FileNotFoundError(scene)


def main() -> None:
    records = []
    for scene in ("drawer", "birdcage"):
        batch = latest_complete_batch(scene)
        manifest = json.loads((batch / "batch_manifest.json").read_text(encoding="utf-8"))
        for result in manifest["results"]:
            counts = result["final_mvt_level_proxy_counts"]
            total = sum(counts)
            shares = [count / total for count in counts]
            effective_levels = sum(share >= 0.05 for share in shares)
            dominant = max(shares)
            hierarchy_required = result["group"] in HIERARCHY_GROUPS
            structure_passed = effective_levels >= 2 and dominant <= 0.90
            records.append(
                {
                    "scene": scene,
                    "group": result["group"],
                    "hierarchy_required": hierarchy_required,
                    "level_counts": counts,
                    "level_shares": shares,
                    "effective_level_count": effective_levels,
                    "dominant_share": dominant,
                    "structure_passed": structure_passed,
                    "full_oracle_passed": result["all_mvt_oracle_checks_passed"],
                    "batch": str(batch.relative_to(ROOT)),
                }
            )
    passed = all(
        row["full_oracle_passed"]
        and (row["structure_passed"] if row["hierarchy_required"] else True)
        for row in records
    )
    report = {
        "experiment": "07.2",
        "gate": "E7-D5 initial-snapshot MVT structure and oracle",
        "mock_data": False,
        "criterion": {
            "hierarchy_groups": list(HIERARCHY_GROUPS),
            "minimum_share_per_effective_level": 0.05,
            "minimum_effective_levels": 2,
            "maximum_dominant_share": 0.90,
            "all_groups_require_full_oracle": True,
            "S_role": "cap-reduced load baseline; reported but excluded from hierarchy claim",
        },
        "records": records,
        "scope": "initial immutable snapshots; full trajectory control-equivalence audit pending",
        "passed": passed,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "mvt_selected_group_audit_07_2.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (TABLE_ROOT / "T26_mvt_level_smoke_07_2.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        rows = [
            {
                **row,
                "level_counts": json.dumps(row["level_counts"]),
                "level_shares": json.dumps(row["level_shares"]),
            }
            for row in records
        ]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

