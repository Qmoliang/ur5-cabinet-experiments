"""Verify that brute AABB, scalar MVT, and AVX2 MVT solve one QP sequence.

The historical ``full_scan`` controller deliberately has no action-distance
broadphase and is therefore a different constraint problem.  This verifier
uses ``aabb_full_scan`` as the O(N) oracle so only the candidate-search data
structure changes across the three acceleration modes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from model import build_model, build_robot_certificate, certificate_world_positions, set_configuration
from native_mvt import BruteForceAABBIndex, NativeMultilevelMVT
from protocol_drawer_scene import NEAR_DISTANCE, SAFETY_MARGIN, formal_protocol_scene
from run_protocol_v3_known_ablation import MVT_BASE_VOXEL_SIZE


MODES = ("aabb_full_scan", "mvt_scalar", "mvt_simd")
LOGICAL_CYCLE_COLUMNS = (
    "cycle",
    "representation",
    "status",
    "qp_row_sha256",
    "broadphase_candidate_pairs",
    "active_obstacle_rows",
    "redundant_proxies_removed",
    "normal_pairs",
    "near_penalty_terms",
    "contact_repulsion_rows",
    "workspace_rows",
    "limiting_robot_index",
    "limiting_obstacle_index",
    "limiting_proxy_id",
)


def _run_dir(root: Path, representation: str, mode: str) -> Path:
    return root / f"K-{representation.capitalize()}-LiuQP-{mode}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _logical_rows(rows: list[dict[str, str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(row[column] for column in LOGICAL_CYCLE_COLUMNS) for row in rows)


def _proxy_half(snapshot: np.lib.npyio.NpzFile, representation: str) -> np.ndarray:
    if representation == "sphere":
        radii = np.asarray(snapshot["sphere_radii"], dtype=float)
        return np.repeat(radii[:, None], 3, axis=1)
    shapes = np.asarray(snapshot["ellipsoid_shapes"], dtype=float)
    return np.sqrt(np.maximum(np.diagonal(shapes, axis1=1, axis2=2), 0.0))


def verify(root: Path) -> dict:
    scene = formal_protocol_scene()
    model = build_model(scene)
    data = mujoco.MjData(model)
    robot = build_robot_certificate(model)
    robot_radii = np.asarray([item.radius for item in robot], dtype=float)
    query_padding = NEAR_DISTANCE + SAFETY_MARGIN
    maximum_query_half = float(np.max(robot_radii) + query_padding)
    report = {
        "root": str(root.resolve()),
        "reference_mode": "aabb_full_scan",
        "excluded_as_non_equivalent": {
            "mode": "full_scan",
            "reason": "no action-distance AABB predicate; it solves a different QP row set",
        },
        "representations": {},
    }

    for representation in ("sphere", "ellipsoid"):
        directories = {
            mode: _run_dir(root, representation, mode) for mode in MODES
        }
        summaries = {
            mode: json.loads((path / "summary.json").read_text(encoding="utf-8"))
            for mode, path in directories.items()
        }
        trajectories = {
            mode: np.load(path / "trajectory.npz")
            for mode, path in directories.items()
        }
        cycles = {
            mode: _read_csv(path / "cycles.csv")
            for mode, path in directories.items()
        }
        pairs = {
            mode: _read_csv(path / "pair_states.csv")
            for mode, path in directories.items()
        }
        reference = trajectories["aabb_full_scan"]
        trajectory_equal = {
            mode: {
                key: bool(np.array_equal(reference[key], trajectories[mode][key]))
                for key in ("q", "qdot", "ee", "error")
            }
            for mode in MODES[1:]
        }
        logical_rows_equal = {
            mode: _logical_rows(cycles[mode]) == _logical_rows(cycles["aabb_full_scan"])
            for mode in MODES[1:]
        }
        pair_rows_equal = {
            mode: pairs[mode] == pairs["aabb_full_scan"] for mode in MODES[1:]
        }
        source_hashes_equal = all(
            summaries[mode]["hashes"]["source_sha256"]
            == summaries["aabb_full_scan"]["hashes"]["source_sha256"]
            for mode in MODES[1:]
        )
        immutable_inputs_equal = all(
            all(
                summaries[mode]["hashes"][key]
                == summaries["aabb_full_scan"]["hashes"][key]
                for key in (
                    "scene_xml_sha256",
                    "proxy_snapshot_sha256",
                    "robot_certificate_sha256",
                )
            )
            for mode in MODES[1:]
        )

        proxy_path = directories["aabb_full_scan"] / "matched_environment_proxies.npz"
        with np.load(proxy_path) as snapshot:
            centers = np.asarray(snapshot["centers"], dtype=float)
            half = _proxy_half(snapshot, representation)
        oracle = BruteForceAABBIndex(
            centers, half, query_padding=query_padding
        )
        scalar = NativeMultilevelMVT(
            centers,
            half,
            base_voxel_size=MVT_BASE_VOXEL_SIZE,
            maximum_query_half_extent=maximum_query_half,
            query_padding=query_padding,
            simd=False,
        )
        simd = NativeMultilevelMVT(
            centers,
            half,
            base_voxel_size=MVT_BASE_VOXEL_SIZE,
            maximum_query_half_extent=maximum_query_half,
            query_padding=query_padding,
            simd=True,
        )
        query_count = 0
        candidate_id_order_equal = True
        try:
            for q in reference["q"]:
                set_configuration(model, data, q)
                positions = certificate_world_positions(data, robot)
                oracle_rows = oracle.query_spheres(positions, robot_radii)
                scalar_rows = scalar.query_spheres(positions, robot_radii)
                simd_rows = simd.query_spheres(positions, robot_radii)
                for expected, scalar_row, simd_row in zip(
                    oracle_rows, scalar_rows, simd_rows
                ):
                    query_count += 1
                    if not (
                        np.array_equal(expected, scalar_row)
                        and np.array_equal(expected, simd_row)
                    ):
                        candidate_id_order_equal = False
                        break
                if not candidate_id_order_equal:
                    break
        finally:
            scalar.close()
            simd.close()

        timing = {
            mode: summaries[mode]["timing_ms"] for mode in MODES
        }
        brute_p99 = timing["aabb_full_scan"]["controller_p99"]
        scalar_p99 = timing["mvt_scalar"]["controller_p99"]
        simd_p99 = timing["mvt_simd"]["controller_p99"]
        passed = bool(
            candidate_id_order_equal
            and source_hashes_equal
            and immutable_inputs_equal
            and all(all(values.values()) for values in trajectory_equal.values())
            and all(logical_rows_equal.values())
            and all(pair_rows_equal.values())
        )
        report["representations"][representation] = {
            "passed": passed,
            "candidate_id_and_order_equal": candidate_id_order_equal,
            "candidate_queries_checked": query_count,
            "trajectory_arrays_exactly_equal": trajectory_equal,
            "logical_cycle_rows_equal": logical_rows_equal,
            "pair_state_rows_equal": pair_rows_equal,
            "source_hashes_equal": source_hashes_equal,
            "immutable_input_hashes_equal": immutable_inputs_equal,
            "timing_ms": timing,
            "controller_p99_speedup_over_aabb_full_scan": {
                "mvt_scalar": brute_p99 / scalar_p99,
                "mvt_simd": brute_p99 / simd_p99,
            },
            "simd_over_scalar_controller_p99": scalar_p99 / simd_p99,
        }
        for archive in trajectories.values():
            archive.close()

    report["passed"] = all(
        item["passed"] for item in report["representations"].values()
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = verify(args.root)
    output = args.root / "acceleration_equivalence_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
