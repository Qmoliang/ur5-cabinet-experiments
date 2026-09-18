"""Audit unique CenterVox witnesses of retained ellipsoid proxies.

This is a read-only diagnostic for a completed formal run.  It reconstructs
the exact sufficient set-inclusion relation used by the online irredundant
cover, then reports the cells that make selected proxies non-deletable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from irredundant_proxy_cover import ellipsoid_candidate_coverage


def resolve_run(path: Path) -> Path:
    if (path / "summary.json").exists():
        return path
    children = [item for item in path.iterdir() if (item / "summary.json").exists()]
    if len(children) != 1:
        raise ValueError(f"expected exactly one run below {path}, found {len(children)}")
    return children[0]


def axes(shape: np.ndarray) -> list[float]:
    return np.sqrt(np.maximum(np.linalg.eigvalsh(shape), 0.0)).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--proxy-id", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = resolve_run(args.run)
    archive = np.load(run / "final_causal_proxies.npz", allow_pickle=False)
    centers = np.asarray(archive["centers"], dtype=float)
    core_shapes = np.asarray(archive["ellipsoid_shapes"], dtype=float)
    uncertainty = np.asarray(archive["proxy_uncertainty_shapes"], dtype=float)
    outer_shapes = np.asarray(archive["ellipsoid_outer_shapes"], dtype=float)
    offsets = np.asarray(archive["uncertainty_offsets"], dtype=float)
    proxy_ids = np.asarray(archive["proxy_ids"], dtype=np.int64)
    points = np.asarray(archive["filtered_points"], dtype=float)
    cell_u = np.asarray(archive["filtered_uncertainty_shapes"], dtype=float)
    cell_offsets = np.asarray(archive["filtered_point_offsets"], dtype=float)
    owner = np.asarray(archive["filtered_cluster_indices"], dtype=np.int64)
    archive.close()

    history_archive = np.load(
        run / "causal_centervox_snapshots.npz", allow_pickle=False
    )
    publish_cycles = np.asarray(history_archive["publish_cycles"], dtype=np.int64)
    source_cycles = np.asarray(history_archive["source_cycles"], dtype=np.int64)
    history_offsets = np.asarray(history_archive["offsets"], dtype=np.int64)
    history_points = np.asarray(history_archive["points"], dtype=float)
    history_u = np.asarray(history_archive["uncertainty_shapes"], dtype=float)
    filter_size = float(history_archive["centervox_size_m"])
    history_archive.close()
    origin = np.array([-1.2, -1.2, 0.0], dtype=float)

    def voxel_key(point: np.ndarray) -> np.ndarray:
        return np.floor((point - origin) / filter_size).astype(np.int64)

    def cell_history(cell_index: int) -> list[dict]:
        target_key = voxel_key(points[cell_index])
        rows: list[dict] = []
        for snapshot_index, (start, stop) in enumerate(
            zip(history_offsets[:-1], history_offsets[1:])
        ):
            local_points = history_points[start:stop]
            keys = np.floor(
                (local_points - origin[None, :]) / filter_size
            ).astype(np.int64)
            matches = np.flatnonzero(np.all(keys == target_key[None, :], axis=1))
            if len(matches) != 1:
                continue
            absolute = int(start + matches[0])
            rows.append(
                {
                    "publish_cycle": int(publish_cycles[snapshot_index]),
                    "source_cycle": int(source_cycles[snapshot_index]),
                    "point_m": history_points[absolute].tolist(),
                    "uncertainty_axes_m": axes(history_u[absolute]),
                }
            )
        return rows

    candidates = SimpleNamespace(
        centers=centers,
        ellipsoid_shapes=core_shapes,
        base_ellipsoid_shapes=core_shapes,
        proxy_uncertainty_shapes=uncertainty,
        proxy_offset_radii=offsets,
    )
    coverage = ellipsoid_candidate_coverage(
        candidates,
        cell_points=points,
        cell_uncertainty_shapes=cell_u,
        cell_offsets=cell_offsets,
    )
    cover_count = np.zeros(len(points), dtype=np.int64)
    for covered in coverage:
        cover_count[covered] += 1
    if np.any(cover_count <= 0):
        raise AssertionError("selected final proxies do not cover every final cell")

    reports = []
    for requested_id in args.proxy_id:
        matches = np.flatnonzero(proxy_ids == requested_id)
        if len(matches) != 1:
            raise ValueError(f"proxy id {requested_id} has {len(matches)} matches")
        index = int(matches[0])
        covered = coverage[index]
        unique = covered[cover_count[covered] == 1]
        assigned = np.flatnonzero(owner == index)

        def cell_rows(indices: np.ndarray, limit: int = 20) -> list[dict]:
            ordered = sorted(
                indices.tolist(),
                key=lambda item: (-max(axes(cell_u[item])), item),
            )[:limit]
            return [
                {
                    "cell_index": int(item),
                    "point_m": points[item].tolist(),
                    "uncertainty_axes_m": axes(cell_u[item]),
                    "scalar_offset_m": float(cell_offsets[item]),
                    "total_max_support_m": float(
                        max(axes(cell_u[item])) + cell_offsets[item]
                    ),
                    "cover_multiplicity": int(cover_count[item]),
                }
                for item in ordered
            ]

        reports.append(
            {
                "proxy_id": int(requested_id),
                "selected_index": index,
                "center_m": centers[index].tolist(),
                "core_axes_m": axes(core_shapes[index]),
                "uncertainty_axes_m": axes(uncertainty[index]),
                "outer_axes_m": axes(outer_shapes[index]),
                "scalar_offset_m": float(offsets[index]),
                "covered_cell_count": int(len(covered)),
                "unique_witness_count": int(len(unique)),
                "assigned_cell_count": int(len(assigned)),
                "largest_unique_witnesses": cell_rows(unique),
                "largest_assigned_cells": cell_rows(assigned),
                "largest_unique_witness_histories": [
                    {
                        "cell_index": int(item),
                        "voxel_key": voxel_key(points[item]).tolist(),
                        "history": cell_history(int(item)),
                    }
                    for item in sorted(
                        unique.tolist(),
                        key=lambda cell: (-max(axes(cell_u[cell])), cell),
                    )[:4]
                ],
            }
        )

    result = {
        "diagnostic_only": True,
        "feeds_back_to_formal_run": False,
        "source_run": str(run.resolve()),
        "selected_proxy_count": int(len(centers)),
        "centervox_cell_count": int(len(points)),
        "all_cells_covered": True,
        "minimum_cell_cover_multiplicity": int(np.min(cover_count)),
        "maximum_cell_cover_multiplicity": int(np.max(cover_count)),
        "proxies": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
