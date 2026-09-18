"""Plot the Experiment 07.2 formal drawer outcome from T27."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "tables" / "experiment_07" / "T27_drawer_formal_runs_07_2.csv"
OUTPUT_DIR = ROOT / "figures" / "experiment_07"
OUTPUT = OUTPUT_DIR / "F14_drawer_formal_outcome_07_2.png"
GROUPS = ("S", "SA", "E0", "ET30")
COLORS = ("#8C8C8C", "#4C78A8", "#54A24B", "#E45756")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    errors = {
        group: np.asarray(
            [1000.0 * float(row["final_error_m"]) for row in rows if row["group"] == group]
        )
        for group in GROUPS
    }
    successes = {
        group: sum(row["success"].lower() == "true" for row in rows if row["group"] == group)
        for group in GROUPS
    }
    eligible = {
        group: sum(
            row["evidence_eligible"].lower() == "true"
            for row in rows
            if row["group"] == group
        )
        for group in GROUPS
    }
    proxies = {
        group: np.median(
            [float(row["final_proxy_count"]) for row in rows if row["group"] == group]
        )
        for group in GROUPS
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    ax = axes[0]
    for index, (group, color) in enumerate(zip(GROUPS, COLORS), start=1):
        jitter = np.linspace(-0.08, 0.08, len(errors[group]))
        ax.scatter(
            index + jitter,
            errors[group],
            s=45,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.hlines(
            np.median(errors[group]),
            index - 0.23,
            index + 0.23,
            color="black",
            linewidth=2.0,
        )
    ax.axhline(1.0, color="#D62728", linestyle="--", linewidth=1.3, label="1 mm target")
    ax.set_yscale("log")
    ax.set_xticks(range(1, 5), GROUPS)
    ax.set_ylabel("Final Cartesian error (mm, log scale)")
    ax.set_title("Five frozen initial configurations")
    ax.grid(axis="y", which="both", alpha=0.22)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    x = np.arange(len(GROUPS))
    width = 0.32
    ax.bar(x - width / 2, [successes[g] for g in GROUPS], width, color=COLORS, label="Task success")
    ax.bar(
        x + width / 2,
        [eligible[g] for g in GROUPS],
        width,
        color="white",
        edgecolor=COLORS,
        linewidth=1.8,
        label="Evidence eligible",
    )
    for i, group in enumerate(GROUPS):
        ax.text(i, 4.78, f"median proxies\n{proxies[group]:.0f}", ha="center", va="top", fontsize=8)
    ax.set_xticks(x, GROUPS)
    ax.set_ylim(0, 5.15)
    ax.set_yticks(range(0, 6))
    ax.set_ylabel("Runs out of 5")
    ax.set_title("Strict outcomes and evidence gates")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle("Experiment 07.2 — Grounded drawer formal results", fontsize=13)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220)
    plt.close(fig)
    manifest = {
        "figure": "F14",
        "mock_data": False,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256(OUTPUT),
        "filters": "all 20 formal drawer runs; no exclusions",
    }
    (OUTPUT_DIR / "F14_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

