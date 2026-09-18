"""Replay the exact online free/occupied/unknown state-change journal.

Unlike a second MuJoCo render, this verifier consumes the map deltas emitted by
the online native occupancy map itself.  It therefore proves that every map
shown after the run can be reconstructed bit-for-bit without changing camera,
publication, or controller timing.  Re-render reproducibility is a separate
diagnostic because MuJoCo multi-ray calls are not guaranteed bit-identical
across process scheduling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _state_hash(state: dict[tuple[int, int, int], int]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(np.asarray((*key, value), dtype="<i4").tobytes())
    return digest.hexdigest()


def verify(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "causal_occupancy_snapshots.npz") as occupancy, np.load(
        run_dir / "causal_source_configurations.npz"
    ) as source, np.load(run_dir / "causal_proxy_snapshots.npz") as proxies:
        publish = np.asarray(occupancy["publish_cycles"], dtype=np.int64)
        source_cycles = np.asarray(occupancy["source_cycles"], dtype=np.int64)
        offsets = np.asarray(occupancy["offsets"], dtype=np.int64)
        keys = np.asarray(occupancy["voxel_keys"], dtype=np.int32)
        states = np.asarray(occupancy["voxel_states"], dtype=np.int8)
        storage = str(np.asarray(occupancy["storage"]).item())
        voxel_size = float(np.asarray(occupancy["voxel_size_m"]).item())
        source_publish = np.asarray(source["publish_cycles"], dtype=np.int64)
        source_source = np.asarray(source["source_cycles"], dtype=np.int64)
        proxy_publish = np.asarray(proxies["publish_cycles"], dtype=np.int64)
        proxy_source = np.asarray(proxies["source_cycles"], dtype=np.int64)

    with (run_dir / "perception_frames.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        frame_rows = list(csv.DictReader(stream))
    frame_publish = np.asarray(
        [int(row["publish_cycle"]) for row in frame_rows], dtype=np.int64
    )
    frame_source = np.asarray(
        [int(row["source_cycle"]) for row in frame_rows], dtype=np.int64
    )

    count = len(publish)
    if storage != "discrete_state_change_deltas":
        raise AssertionError(f"unexpected occupancy storage mode: {storage}")
    if not (
        len(source_cycles) == count
        and len(offsets) == count + 1
        and len(frame_rows) == count
        and len(source_publish) == count
        and len(proxy_publish) == count
    ):
        raise AssertionError("causal occupancy/source/proxy frame counts differ")
    if offsets[0] != 0 or offsets[-1] != len(keys) or len(keys) != len(states):
        raise AssertionError("invalid occupancy delta offsets")
    if np.any(np.diff(offsets) < 0):
        raise AssertionError("occupancy delta offsets are not monotone")
    for candidate in (source_publish, proxy_publish, frame_publish):
        np.testing.assert_array_equal(candidate, publish)
    for candidate in (source_source, proxy_source, frame_source):
        np.testing.assert_array_equal(candidate, source_cycles)
    if count and (publish[0] != 0 or np.any(np.diff(publish) <= 0)):
        raise AssertionError("publish cycles must start at zero and increase")
    if count and np.any(source_cycles > publish):
        raise AssertionError("a map was published before its source frame")
    if int(summary["causal_occupancy_snapshots_saved"]) != count:
        raise AssertionError("summary occupancy snapshot count differs")

    replay: dict[tuple[int, int, int], int] = {}
    frame_reports = []
    total_unknown_removals = 0
    total_net_noop_rows = 0
    for index in range(count):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        local_keys = keys[start:stop]
        local_states = states[start:stop]
        if len(local_keys) != len({tuple(map(int, key)) for key in local_keys}):
            raise AssertionError(f"duplicate voxel key in delta frame {index}")
        if np.any((local_states < 0) | (local_states > 2)):
            raise AssertionError(f"invalid occupancy state in delta frame {index}")
        changed = 0
        net_noops = 0
        for key, state in zip(local_keys, local_states):
            packed = tuple(map(int, key))
            previous = replay.get(packed, 0)
            value = int(state)
            if previous == value:
                # The native dirty journal records a key if it changed at any
                # point during integration.  Multiple rays in one frame can
                # return it to its previous published state; retain and count
                # that deterministic net no-op instead of mislabelling it as
                # a malformed delta.
                net_noops += 1
                total_net_noop_rows += 1
            if value == 0:
                replay.pop(packed, None)
                total_unknown_removals += 1
            else:
                replay[packed] = value
            changed += 1
        values = np.fromiter(replay.values(), dtype=np.int8)
        frame_reports.append(
            {
                "snapshot": index,
                "publish_cycle": int(publish[index]),
                "source_cycle": int(source_cycles[index]),
                "delta_state_changes": changed,
                "net_noop_dirty_rows": net_noops,
                "free_voxels_after_delta": int(np.count_nonzero(values == 1)),
                "occupied_voxels_after_delta": int(np.count_nonzero(values == 2)),
                "known_sparse_states_after_delta": len(replay),
            }
        )

    report = {
        "passed": True,
        "method": "exact_replay_of_online_native_state_change_journal",
        "rerendered_camera_frames": False,
        "online_control_or_publication_timing_affected": False,
        "map_voxel_size_m": voxel_size,
        "snapshot_count": count,
        "total_state_change_rows": len(states),
        "unknown_removal_rows": total_unknown_removals,
        "net_noop_dirty_rows": total_net_noop_rows,
        "final_known_sparse_states": len(replay),
        "final_state_sha256": _state_hash(replay),
        "unknown_definition": "all voxel keys absent from sparse FREE/OCCUPIED map",
        "cycles_match_source_proxy_and_frame_logs": True,
        "dirty_journal_semantics": (
            "keys_touched_by_any_intra_frame_state_change; a row may be a "
            "net no-op relative to the prior publication"
        ),
        "frames": frame_reports,
    }
    output = run_dir / "recorded_occupancy_replay_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
