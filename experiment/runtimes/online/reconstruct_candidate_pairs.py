"""Reconstruct every control-cycle AABB candidate pair after control stops."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from model import build_robot_certificate, certificate_world_positions, set_configuration
from native_mvt import BruteForceAABBIndex
from protocol_drawer_scene import NEAR_DISTANCE, SAFETY_MARGIN
from run_protocol_v3_online_ablation import (
    MOTION_BROADPHASE_PADDING,
    MVT_NUMERICAL_OUTWARD_PADDING,
)


def reconstruct(run_dir: Path, *, compact_only: bool = False) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    proxies = np.load(run_dir / "causal_proxy_snapshots.npz")
    publish_cycles = np.asarray(proxies["publish_cycles"], dtype=np.int64)
    offsets = np.asarray(proxies["offsets"], dtype=np.int64)
    centers = np.asarray(proxies["centers"], dtype=float)
    proxy_ids = np.asarray(proxies["proxy_ids"], dtype=np.int64)
    offsets_radius = np.asarray(proxies["uncertainty_offsets"], dtype=float)
    if summary["representation"] == "sphere":
        radii = np.asarray(proxies["sphere_radii"], dtype=float) + offsets_radius
        half = np.repeat(radii[:, None], 3, axis=1)
    else:
        outer = np.asarray(proxies["ellipsoid_outer_shapes"], dtype=float)
        half = np.sqrt(
            np.maximum(np.diagonal(outer, axis1=1, axis2=2), 0.0)
        ) + offsets_radius[:, None]

    with (run_dir / "cycles.csv").open(newline="", encoding="utf-8") as stream:
        cycle_rows = list(csv.DictReader(stream))
    q_after = np.asarray(np.load(run_dir / "q_history.npy"), dtype=float)
    source = np.load(run_dir / "causal_source_configurations.npz")
    q0 = np.asarray(source["q"][0], dtype=float)
    source.close()
    if len(q_after) != len(cycle_rows):
        raise ValueError("q_history and cycles do not match")

    model = mujoco.MjModel.from_xml_path(str(run_dir / "scene.xml"))
    data = mujoco.MjData(model)
    robot = build_robot_certificate(model)
    robot_radii = np.asarray([item.radius for item in robot], dtype=float)
    # Match the online table exactly: float32 proxy AABBs and the query radius
    # each receive the independently audited 5 micrometre outward-only
    # broadphase pad.  The old replay omitted both pads and therefore dropped
    # boundary false positives that the online MVT intentionally retained.
    half = half + MVT_NUMERICAL_OUTWARD_PADDING
    query_padding = (
        NEAR_DISTANCE
        + SAFETY_MARGIN
        + MOTION_BROADPHASE_PADDING
        + MVT_NUMERICAL_OUTWARD_PADDING
    )
    indices: dict[int, BruteForceAABBIndex] = {}
    output_rows = []
    compact_per_cycle: list[np.ndarray] = []
    candidate_pair_rows = 0
    all_counts_match = True
    for cycle, cycle_row in enumerate(cycle_rows):
        snapshot = int(np.searchsorted(publish_cycles, cycle, side="right") - 1)
        snapshot = max(snapshot, 0)
        start, stop = map(int, offsets[snapshot : snapshot + 2])
        if snapshot not in indices:
            indices[snapshot] = BruteForceAABBIndex(
                centers[start:stop],
                half[start:stop],
                query_padding=query_padding,
            )
        q_before = q0 if cycle == 0 else q_after[cycle - 1]
        set_configuration(model, data, q_before)
        positions = certificate_world_positions(data, robot)
        candidate_rows = indices[snapshot].query_spheres(positions, robot_radii)
        count = int(sum(len(row) for row in candidate_rows))
        candidate_pair_rows += count
        all_counts_match &= count == int(cycle_row["broadphase_candidate_pairs"])
        if compact_only:
            local = [
                proxy_ids[start + np.asarray(row, dtype=np.int64)]
                for row in candidate_rows
                if len(row)
            ]
            compact_per_cycle.append(
                np.unique(np.concatenate(local))
                if local
                else np.empty(0, dtype=np.int64)
            )
            continue
        for robot_index, local_indices in enumerate(candidate_rows):
            for obstacle_index in local_indices:
                output_rows.append(
                    {
                        "cycle": cycle,
                        "time_s": cycle * 0.02,
                        "active_generation": snapshot,
                        "robot_index": robot_index,
                        "obstacle_index": int(obstacle_index),
                        "proxy_id": int(proxy_ids[start + obstacle_index]),
                    }
                )
    proxies.close()
    if not all_counts_match:
        raise AssertionError("post-control candidate replay diverged from online counts")
    if compact_only:
        offsets_out = np.zeros(len(compact_per_cycle) + 1, dtype=np.int64)
        offsets_out[1:] = np.cumsum(
            [len(item) for item in compact_per_cycle]
        )
        compact_ids = (
            np.concatenate(compact_per_cycle)
            if compact_per_cycle
            else np.empty(0, dtype=np.int64)
        )
        path = run_dir / "candidate_proxy_ids.npz"
        np.savez(path, offsets=offsets_out, proxy_ids=compact_ids)
        compact_report = {
            "passed": True,
            "source_candidate_pair_rows": int(candidate_pair_rows),
            "cycles": len(compact_per_cycle),
            "unique_cycle_proxy_rows": int(len(compact_ids)),
            "output": str(path.resolve()),
        }
        (run_dir / "candidate_proxy_ids_report.json").write_text(
            json.dumps(compact_report, indent=2), encoding="utf-8"
        )
    else:
        path = run_dir / "candidate_pairs.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    report = {
        "passed": True,
        "method": (
            "post_control_float32_brute_ball_AABB_replay_with_online_"
            "outward_padding"
        ),
        "online_control_or_publication_timing_affected": False,
        "all_per_cycle_candidate_counts_matched": True,
        "cycles": len(cycle_rows),
        "robot_spheres": len(robot),
        "candidate_pair_rows": int(candidate_pair_rows),
        "compact_only": bool(compact_only),
        "output": str(path.resolve()),
    }
    (run_dir / "candidate_replay_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--compact-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            reconstruct(args.run_dir, compact_only=args.compact_only),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
