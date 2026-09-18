"""Re-audit a completed causal run without replaying or changing its control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from observability_evaluator import (
    evaluate_observed_before_risk,
    load_observability_snapshots,
)
from protocol_drawer_scene import formal_protocol_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--output-name",
        default="observability_report_nearest_witness_v2.json",
    )
    parser.add_argument("--observation-distance", type=float, default=0.10)
    parser.add_argument("--guard-time", type=float, default=0.10)
    parser.add_argument("--truth-spacing", type=float, default=0.012)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    q_history = np.load(run_dir / "q_history.npy")
    scene = formal_protocol_scene()
    q_trajectory = np.vstack((np.asarray(scene.q0, dtype=float), q_history))
    snapshots = load_observability_snapshots(
        str(run_dir / "causal_observability_snapshots.npz")
    )
    report = evaluate_observed_before_risk(
        scene,
        q_trajectory,
        snapshots,
        observation_distance=args.observation_distance,
        guard_time=args.guard_time,
        truth_surface_spacing=args.truth_spacing,
    )
    output = run_dir / args.output_name
    output.write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
