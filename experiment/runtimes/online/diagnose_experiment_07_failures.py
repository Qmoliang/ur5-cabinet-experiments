"""Diagnose the failed evidence gates in Experiment 07.2 drawer runs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TABLE = ROOT / "tables" / "experiment_07" / "T27_drawer_formal_runs_07_2.csv"
OUTPUT = ROOT / "formal_results" / "experiment_07" / "drawer_failure_diagnosis_07_2.json"


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    rows = read_csv(TABLE)
    e0_events = []
    et30_limits = []
    for row in rows:
        run = ROOT / row["run"]
        cycles = read_csv(run / "cycles.csv")
        pairs = read_csv(run / "pair_states.csv")
        if row["group"] == "E0":
            sweep = [
                {
                    "cycle": int(item["cycle"]),
                    "time_s": float(item["time_s"]),
                    "error_m": float(item["error_m"]),
                    "worsening_pairs": int(item["proxy_sweep_worsening_pairs"]),
                    "minimum_clearance_m": float(
                        item["proxy_sweep_minimum_clearance_m"]
                    ),
                    "executed_qdot_norm": float(item["executed_qdot_norm"]),
                }
                for item in cycles
                if int(item["proxy_sweep_worsening_pairs"]) > 0
            ]
            observability = json.loads(
                (run / "observability_report.json").read_text(encoding="utf-8")
            )
            if sweep or not observability["passed"]:
                e0_events.append(
                    {
                        "seed": int(row["seed"]),
                        "task_success": row["success"].lower() == "true",
                        "sweep_failures": sweep,
                        "observability": observability,
                    }
                )
        if row["group"] == "ET30":
            last_cycle = int(cycles[-1]["cycle"])
            tail = [item for item in pairs if int(item["cycle"]) > last_cycle - 200]
            limiting = min(tail, key=lambda item: float(item["clearance"]))
            snapshot = np.load(run / "final_causal_proxies.npz")
            proxy_ids = np.asarray(snapshot["proxy_ids"], dtype=np.int64)
            index = int(np.flatnonzero(proxy_ids == int(limiting["proxy_id"]))[0])
            core = np.asarray(snapshot["ellipsoid_shapes"][index], dtype=float)
            axes = np.sqrt(np.maximum(np.linalg.eigvalsh(core), 0.0))
            et30_limits.append(
                {
                    "seed": int(row["seed"]),
                    "final_error_m": float(row["final_error_m"]),
                    "tail_state_counts": dict(Counter(item["state"] for item in tail)),
                    "limiting_pair": {
                        "cycle": int(limiting["cycle"]),
                        "robot_index": int(limiting["robot_index"]),
                        "proxy_id": int(limiting["proxy_id"]),
                        "clearance_m": float(limiting["clearance"]),
                        "state": limiting["state"],
                        "proxy_center_m": np.asarray(
                            snapshot["centers"][index], dtype=float
                        ).tolist(),
                        "core_semi_axes_m": axes.tolist(),
                        "scalar_offset_m": float(
                            snapshot["uncertainty_offsets"][index]
                        ),
                    },
                }
            )
            snapshot.close()
    report = {
        "experiment": "07.2",
        "mock_data": False,
        "e0_evidence_gate_events": e0_events,
        "et30_tail_limits": et30_limits,
        "diagnosis": [
            "E0 seed 0 has one 23.3 micrometre conservative sweep-bound violation at cycle 76 while moving quickly; MuJoCo penetration remains zero.",
            "E0 seed 3 first approaches an unobserved drawer-left patch at cycle 107 and receives it at cycle 152, 50 cycles after the deadline.",
            "ET30 terminal blocking is localized to drawer-bottom/forearm and ceiling/wrist constraints, with separate uncertainty envelopes and NEAR damping; a sole causal attribution to the 3 mm core floor is unsupported. See terminal_diagnosis/ET30_terminal_diagnosis.md for reconstructed QP and frozen-continuation evidence.",
            "Any fix changes a frozen variable and must be registered as Experiment 07.3.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "E0_events": len(e0_events), "ET30_limits": len(et30_limits)}, indent=2))


if __name__ == "__main__":
    main()


