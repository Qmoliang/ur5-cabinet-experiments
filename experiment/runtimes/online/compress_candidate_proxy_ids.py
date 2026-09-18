"""Compress a candidate-pair CSV to per-cycle unique proxy-ID ragged arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compress(run_dir: Path, chunk_rows: int = 1_000_000) -> dict:
    source = run_dir / "candidate_pairs.csv"
    report = json.loads(
        (run_dir / "candidate_replay_report.json").read_text(encoding="utf-8")
    )
    per_cycle: list[np.ndarray] = []
    pending_cycle = None
    pending_ids = np.empty(0, dtype=np.int64)
    source_rows = 0
    for chunk in pd.read_csv(
        source,
        usecols=["cycle", "proxy_id"],
        dtype={"cycle": np.int32, "proxy_id": np.int64},
        chunksize=chunk_rows,
    ):
        source_rows += len(chunk)
        unique = chunk.drop_duplicates(["cycle", "proxy_id"])
        for cycle, group in unique.groupby("cycle", sort=True):
            cycle = int(cycle)
            ids = np.sort(group["proxy_id"].to_numpy(dtype=np.int64))
            if pending_cycle is None:
                pending_cycle = cycle
                pending_ids = ids
            elif cycle == pending_cycle:
                pending_ids = np.union1d(pending_ids, ids)
            else:
                while len(per_cycle) < pending_cycle:
                    per_cycle.append(np.empty(0, dtype=np.int64))
                if pending_cycle != len(per_cycle):
                    raise RuntimeError("candidate CSV cycles are not ordered")
                per_cycle.append(pending_ids)
                pending_cycle = cycle
                pending_ids = ids
    if pending_cycle is not None:
        while len(per_cycle) < pending_cycle:
            per_cycle.append(np.empty(0, dtype=np.int64))
        if pending_cycle != len(per_cycle):
            raise RuntimeError("candidate CSV final cycle is not ordered")
        per_cycle.append(pending_ids)
    if source_rows != int(report["candidate_pair_rows"]):
        raise RuntimeError("candidate CSV row count disagrees with replay report")
    while len(per_cycle) < int(report["cycles"]):
        per_cycle.append(np.empty(0, dtype=np.int64))
    if len(per_cycle) != int(report["cycles"]):
        raise RuntimeError("candidate CSV cycle count disagrees with replay report")
    offsets = np.zeros(len(per_cycle) + 1, dtype=np.int64)
    if per_cycle:
        offsets[1:] = np.cumsum([len(item) for item in per_cycle])
        proxy_ids = np.concatenate(per_cycle)
    else:
        proxy_ids = np.empty(0, dtype=np.int64)
    output = run_dir / "candidate_proxy_ids.npz"
    np.savez(
        output,
        offsets=offsets,
        proxy_ids=proxy_ids,
    )
    result = {
        "passed": True,
        "source_candidate_pair_rows": int(source_rows),
        "cycles": len(per_cycle),
        "unique_cycle_proxy_rows": int(len(proxy_ids)),
        "output": str(output.resolve()),
    }
    (run_dir / "candidate_proxy_ids_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    args = parser.parse_args()
    print(json.dumps(compress(args.run_dir, args.chunk_rows), indent=2))


if __name__ == "__main__":
    main()
