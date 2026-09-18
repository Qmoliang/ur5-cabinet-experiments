"""Formal known-environment protocol-v3 sphere/ellipsoid LiuQP ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import mujoco
import numpy as np

from model import (
    DT,
    JOINT_NAMES,
    attachment_position,
    build_model,
    build_robot_certificate,
    build_xml,
    certificate_world_positions,
    set_configuration,
)
from native_mvt import BruteForceAABBIndex, NativeMultilevelMVT
from protocol_drawer_scene import (
    CONTACT_DISTANCE,
    KNOWN_PROXY_CELL_SIZE,
    NEAR_DISTANCE,
    SAFETY_MARGIN,
    SUCCESS_TOLERANCE,
    formal_protocol_scene,
)
from protocol_known_proxies import build_known_matched_proxy_tree
from protocol_liuqp_controller import ProtocolLiuQPController


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "formal_results" / "protocol_v3" / "known_ablation"
SUCCESS_HOLD_CYCLES = 10
MVT_BASE_VOXEL_SIZE = 0.015
MVT_QUERY_PADDING = NEAR_DISTANCE + SAFETY_MARGIN


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _velocity_sign_flip_rate(qdot: np.ndarray) -> float:
    if len(qdot) < 2:
        return 0.0
    previous = np.sign(qdot[:-1])
    current = np.sign(qdot[1:])
    valid = (previous != 0.0) & (current != 0.0)
    return float(np.sum((previous != current) & valid) / max(np.sum(valid), 1))


def _source_hashes() -> dict[str, str]:
    names = (
        "run_protocol_v3_known_ablation.py",
        "protocol_liuqp_controller.py",
        "protocol_drawer_scene.py",
        "protocol_known_proxies.py",
        "ellipsoid_model.py",
        "model.py",
        "native_mvt.py",
        "native_ellipsoid_support.py",
        "build_native_mvt.bat",
        "native_mvt/native_mvt.cpp",
        "native_mvt/native_mvt.dll",
    )
    return {
        name: _sha256((ROOT / name).read_bytes())
        for name in names
    }


def _known_proxy_half_extents(proxies, representation: str) -> np.ndarray:
    if representation == "sphere":
        return np.repeat(proxies.sphere_radii[:, None], 3, axis=1)
    return np.sqrt(
        np.maximum(
            np.diagonal(proxies.ellipsoid_shapes, axis1=1, axis2=2),
            0.0,
        )
    )


def _build_known_mvt(
    proxies,
    representation: str,
    robot,
    data,
    *,
    simd: bool,
) -> NativeMultilevelMVT:
    half = _known_proxy_half_extents(proxies, representation)
    robot_radii = np.asarray([item.radius for item in robot], dtype=float)
    table = NativeMultilevelMVT(
        proxies.centers,
        half,
        base_voxel_size=MVT_BASE_VOXEL_SIZE,
        maximum_query_half_extent=float(
            np.max(robot_radii) + MVT_QUERY_PADDING
        ),
        query_padding=MVT_QUERY_PADDING,
        simd=simd,
    )
    positions = certificate_world_positions(data, robot)
    for center, radius in zip(positions, robot_radii):
        query_radius = float(radius) + MVT_QUERY_PADDING
        lo = proxies.centers - half
        hi = proxies.centers + half
        delta = np.maximum(np.maximum(lo - center, center - hi), 0.0)
        expected = np.flatnonzero(
            np.einsum("ij,ij->i", delta, delta)
            <= query_radius * query_radius
        )
        actual = table.query_sphere(center, float(radius))
        missing = np.setdiff1d(expected, actual, assume_unique=True)
        if len(missing):
            table.close()
            raise AssertionError(
                f"multilevel MVT omitted {len(missing)} initial oracle candidates"
            )
    return table


def _build_known_aabb_full_scan(
    proxies,
    representation: str,
) -> BruteForceAABBIndex:
    """O(N) oracle with the exact same ball--AABB predicate used by MVT."""

    return BruteForceAABBIndex(
        proxies.centers,
        _known_proxy_half_extents(proxies, representation),
        query_padding=MVT_QUERY_PADDING,
    )


def _audit_mvt_trajectory(
    table: NativeMultilevelMVT,
    proxies,
    representation: str,
    robot,
    model,
    data,
    q_history: np.ndarray,
) -> int:
    half = _known_proxy_half_extents(proxies, representation)
    robot_radii = np.asarray([item.radius for item in robot], dtype=float)
    missing = 0
    for q in q_history:
        set_configuration(model, data, q)
        positions = certificate_world_positions(data, robot)
        for center, radius in zip(positions, robot_radii):
            query_radius = float(radius) + MVT_QUERY_PADDING
            lo = proxies.centers - half
            hi = proxies.centers + half
            delta = np.maximum(np.maximum(lo - center, center - hi), 0.0)
            expected = np.flatnonzero(
                np.einsum("ij,ij->i", delta, delta)
                <= query_radius * query_radius
            )
            actual = table.query_sphere(center, float(radius))
            missing += len(np.setdiff1d(expected, actual, assume_unique=True))
    return int(missing)


def run_one(
    representation: str,
    output_root: Path = DEFAULT_OUTPUT,
    *,
    maximum_cycles: int | None = None,
    index_mode: str = "full_scan",
) -> dict:
    if representation not in {"sphere", "ellipsoid"}:
        raise ValueError("representation must be sphere or ellipsoid")
    if index_mode not in {
        "full_scan",
        "aabb_full_scan",
        "mvt_scalar",
        "mvt_simd",
    }:
        raise ValueError("unknown index_mode")
    scene = formal_protocol_scene()
    model = build_model(scene)
    data = mujoco.MjData(model)
    q0 = np.asarray(scene.q0, dtype=float)
    set_configuration(model, data, q0)
    robot = build_robot_certificate(model)
    proxies = build_known_matched_proxy_tree(
        scene.boxes,
        cell_size=KNOWN_PROXY_CELL_SIZE,
    )
    obstacle_index = (
        None
        if index_mode == "full_scan"
        else (
            _build_known_aabb_full_scan(proxies, representation)
            if index_mode == "aabb_full_scan"
            else _build_known_mvt(
                proxies,
                representation,
                robot,
                data,
                simd=index_mode == "mvt_simd",
            )
        )
    )
    common = dict(
        model=model,
        data=data,
        scene=scene,
        robot_spheres=robot,
        obstacle_centers=proxies.centers,
        representation=representation,
        proxy_ids=proxies.proxy_ids,
        safety_margin=SAFETY_MARGIN,
        near_distance=NEAR_DISTANCE,
        contact_distance=CONTACT_DISTANCE,
        obstacle_index=obstacle_index,
        redundant_plane_pruning=True,
    )
    if representation == "sphere":
        controller = ProtocolLiuQPController(
            **common,
            obstacle_radii=proxies.sphere_radii,
        )
    else:
        controller = ProtocolLiuQPController(
            **common,
            obstacle_shapes=proxies.ellipsoid_shapes,
        )

    target = np.asarray(scene.waypoints[-1], dtype=float)
    formal_cycles = int(round(scene.duration / DT))
    cycle_limit = formal_cycles if maximum_cycles is None else int(maximum_cycles)
    if cycle_limit <= 0 or cycle_limit > formal_cycles:
        raise ValueError("maximum_cycles must be in the formal duration")
    run_label = f"K-{representation.capitalize()}-LiuQP"
    if index_mode != "full_scan":
        run_label += f"-{index_mode}"
    run_dir = output_root / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    xml = build_xml(scene).encode("utf-8")
    (run_dir / "scene.xml").write_bytes(xml)
    np.savez_compressed(
        run_dir / "matched_environment_proxies.npz",
        proxy_ids=proxies.proxy_ids,
        centers=proxies.centers,
        sphere_radii=proxies.sphere_radii,
        ellipsoid_shapes=proxies.ellipsoid_shapes,
        cell_half_extents=proxies.cell_half_extents,
        owner_indices=proxies.owner_indices,
        cell_indices=proxies.cell_indices,
    )

    cycles: list[dict] = []
    pairs: list[dict] = []
    q_history = []
    qdot_history = []
    ee_history = []
    error_history = []
    success_hold = 0
    first_success_cycle = None
    maximum_penetration = 0.0
    run_started = time.perf_counter()
    for cycle in range(cycle_limit):
        cycle_started = time.perf_counter()
        q_before = data.qpos[: len(JOINT_NAMES)].copy()
        qdot, metrics = controller.solve(target)
        # This is the only state update.  There is no collision backtracking,
        # line search, random perturbation, target change, or route input.
        q_after = q_before + qdot * DT
        set_configuration(model, data, q_after)
        ee = attachment_position(model, data)
        error = float(np.linalg.norm(target - ee))
        penetrating = [
            index
            for index in range(data.ncon)
            if data.contact[index].dist < -1.0e-8
        ]
        minimum_contact_distance = (
            min(float(data.contact[index].dist) for index in range(data.ncon))
            if data.ncon
            else None
        )
        if minimum_contact_distance is not None:
            maximum_penetration = min(
                maximum_penetration,
                minimum_contact_distance,
            )
        if error <= SUCCESS_TOLERANCE and not penetrating:
            success_hold += 1
            if first_success_cycle is None:
                first_success_cycle = cycle
        else:
            success_hold = 0
            if error > SUCCESS_TOLERANCE:
                first_success_cycle = None
        full_cycle_ms = (time.perf_counter() - cycle_started) * 1000.0
        row = {
            "cycle": cycle,
            "time_s": cycle * DT,
            "representation": representation,
            "status": metrics.status,
            "ee_x": float(ee[0]),
            "ee_y": float(ee[1]),
            "ee_z": float(ee[2]),
            "error_m": error,
            "qdot_norm": float(np.linalg.norm(qdot)),
            "exact_contact_count": int(data.ncon),
            "exact_penetrating_contact_count": len(penetrating),
            "minimum_contact_distance_m": minimum_contact_distance,
            "full_cycle_ms": full_cycle_ms,
            **metrics.as_dict(),
        }
        cycles.append(row)
        for record in controller.last_pair_records:
            pairs.append(
                {
                    "cycle": cycle,
                    "time_s": cycle * DT,
                    **record.__dict__,
                }
            )
        q_history.append(q_after.copy())
        qdot_history.append(qdot.copy())
        ee_history.append(ee.copy())
        error_history.append(error)
        if success_hold >= SUCCESS_HOLD_CYCLES:
            break

    q_array = np.asarray(q_history)
    qdot_array = np.asarray(qdot_history)
    ee_array = np.asarray(ee_history)
    error_array = np.asarray(error_history)
    full_ms = np.asarray([row["full_cycle_ms"] for row in cycles])
    total_ms = np.asarray([row["total_controller_ms"] for row in cycles])
    solve_ms = np.asarray([row["solve_ms"] for row in cycles])
    geometry_ms = np.asarray([row["geometry_ms"] for row in cycles])
    solved = np.asarray(
        [str(row["status"]).startswith("solved") for row in cycles]
    )
    candidate_pairs = np.asarray(
        [row["broadphase_candidate_pairs"] for row in cycles], dtype=float
    )
    mvt_missing = (
        0
        if obstacle_index is None or index_mode == "aabb_full_scan"
        else _audit_mvt_trajectory(
            obstacle_index,
            proxies,
            representation,
            robot,
            model,
            data,
            q_array,
        )
    )
    if mvt_missing:
        raise AssertionError(
            f"multilevel MVT omitted {mvt_missing} trajectory oracle candidates"
        )
    success = bool(success_hold >= SUCCESS_HOLD_CYCLES)
    tail_cycles = min(len(error_array), int(round(5.0 / DT)))
    tail_improvement = float(
        np.max(error_array[-tail_cycles:]) - np.min(error_array[-tail_cycles:])
    )
    summary = {
        "formal": maximum_cycles is None,
        "scene": scene.name,
        "group": run_label,
        "representation": representation,
        "index_mode": index_mode,
        "broadphase_predicate": (
            "none_all_proxies"
            if index_mode == "full_scan"
            else "inclusive_float32_proxy_query_aabb_intersection"
        ),
        "broadphase_search": (
            "none_all_proxies"
            if index_mode == "full_scan"
            else (
                "brute_force_O_N"
                if index_mode == "aabb_full_scan"
                else "multilevel_27_cell_MVT"
            )
        ),
        "mvt_multilevel_enabled": index_mode in {"mvt_scalar", "mvt_simd"},
        "mvt_simd_enabled": index_mode == "mvt_simd",
        "success": success,
        "success_hold_cycles": SUCCESS_HOLD_CYCLES,
        "first_success_s": (
            None if first_success_cycle is None else first_success_cycle * DT
        ),
        "executed_cycles": len(cycles),
        "duration_s": len(cycles) * DT,
        "final_error_m": float(error_array[-1]),
        "minimum_error_m": float(np.min(error_array)),
        "minimum_ee_x_m": float(np.min(ee_array[:, 0])),
        "maximum_ee_x_m": float(np.max(ee_array[:, 0])),
        "tail_5s_error_range_m": tail_improvement,
        "stalled_last_5s": bool(
            not success and tail_cycles >= int(round(4.5 / DT))
            and tail_improvement < 0.001
        ),
        "qdot_rms": float(np.sqrt(np.mean(qdot_array**2))),
        "velocity_sign_flip_rate": _velocity_sign_flip_rate(qdot_array),
        "qp_solved_fraction": float(np.mean(solved)),
        "exact_penetrating_cycles": int(
            sum(row["exact_penetrating_contact_count"] > 0 for row in cycles)
        ),
        "maximum_exact_penetration_m": float(maximum_penetration),
        "robot_sphere_count": len(robot),
        "environment_proxy_count": len(proxies.centers),
        "raw_pair_count_per_cycle": len(robot) * len(proxies.centers),
        "broadphase_candidate_pairs_p50": _percentile(candidate_pairs, 50),
        "broadphase_candidate_pairs_p95": _percentile(candidate_pairs, 95),
        "broadphase_candidate_pairs_p99": _percentile(candidate_pairs, 99),
        "broadphase_rejection_fraction_p50": float(
            1.0
            - _percentile(candidate_pairs, 50)
            / (len(robot) * len(proxies.centers))
        ),
        "all_mvt_trajectory_oracle_checks_passed": mvt_missing == 0,
        "mvt_levels": (
            0
            if index_mode not in {"mvt_scalar", "mvt_simd"}
            else obstacle_index.stats.level_count
        ),
        "mvt_index_references": (
            0
            if index_mode not in {"mvt_scalar", "mvt_simd"}
            else obstacle_index.stats.index_references
        ),
        "mvt_cell_lookups_per_robot_query": (
            0
            if index_mode not in {"mvt_scalar", "mvt_simd"}
            else obstacle_index.stats.cell_lookups_per_query
        ),
        "mvt_simd_width": (
            obstacle_index.stats.simd_width
            if index_mode == "mvt_simd"
            else 1
        ),
        "closest_point_newton_iterations_total": int(
            sum(row["closest_point_newton_iterations"] for row in cycles)
        ),
        "closest_point_bisection_iterations_total": int(
            sum(row["closest_point_bisection_iterations"] for row in cycles)
        ),
        "closest_point_max_residual": float(
            max(row["closest_point_max_residual"] for row in cycles)
        ),
        "multiplier_warm_start_hits_total": int(
            sum(row["multiplier_warm_start_hits"] for row in cycles)
        ),
        "timing_ms": {
            "full_cycle_p50": _percentile(full_ms, 50),
            "full_cycle_p95": _percentile(full_ms, 95),
            "full_cycle_p99": _percentile(full_ms, 99),
            "controller_p50": _percentile(total_ms, 50),
            "controller_p95": _percentile(total_ms, 95),
            "controller_p99": _percentile(total_ms, 99),
            "geometry_p50": _percentile(geometry_ms, 50),
            "geometry_p95": _percentile(geometry_ms, 95),
            "geometry_p99": _percentile(geometry_ms, 99),
            "qp_solve_p50": _percentile(solve_ms, 50),
            "qp_solve_p95": _percentile(solve_ms, 95),
            "qp_solve_p99": _percentile(solve_ms, 99),
        },
        "frozen": {
            "control_hz": 1.0 / DT,
            "duration_limit_s": scene.duration,
            "safety_margin_m": SAFETY_MARGIN,
            "near_distance_m": NEAR_DISTANCE,
            "contact_distance_m": CONTACT_DISTANCE,
            "success_tolerance_m": SUCCESS_TOLERANCE,
            "target": target.tolist(),
            "q0": q0.tolist(),
            "known_proxy_cell_size_m": KNOWN_PROXY_CELL_SIZE,
            "route_input": None,
            "orientation_target": None,
            "posture_target": None,
            "random_dither": None,
            "truth_collision_backtracking": False,
            "camera_or_centervox": False,
            "vcc_inspired_mvt_aabb": index_mode != "full_scan",
            "vcc_inspired_simd": index_mode == "mvt_simd",
        },
        "hashes": {
            "scene_xml_sha256": _sha256(xml),
            "proxy_snapshot_sha256": proxies.snapshot_sha256,
            "robot_certificate_sha256": _sha256(
                b"".join(
                    np.ascontiguousarray(value).tobytes()
                    for value in (
                        np.asarray([item.body_id for item in robot]),
                        np.asarray([item.source_geom_id for item in robot]),
                        np.asarray([item.local_center for item in robot]),
                        np.asarray([item.radius for item in robot]),
                    )
                )
            ),
            "source_sha256": _source_hashes(),
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
        },
        "wall_elapsed_s": float(time.perf_counter() - run_started),
    }
    _write_csv(run_dir / "cycles.csv", cycles)
    _write_csv(run_dir / "pair_states.csv", pairs)
    np.savez_compressed(
        run_dir / "trajectory.npz",
        q=q_array,
        qdot=qdot_array,
        ee=ee_array,
        error=error_array,
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if obstacle_index is not None:
        obstacle_index.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representation",
        choices=("sphere", "ellipsoid", "both"),
        default="both",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--index-mode",
        choices=("full_scan", "aabb_full_scan", "mvt_scalar", "mvt_simd"),
        default="full_scan",
    )
    parser.add_argument(
        "--smoke-cycles",
        type=int,
        default=None,
        help="Non-formal implementation smoke test; omit for frozen duration.",
    )
    args = parser.parse_args()
    representations = (
        ("sphere", "ellipsoid")
        if args.representation == "both"
        else (args.representation,)
    )
    summaries = [
        run_one(
            value,
            args.output,
            maximum_cycles=args.smoke_cycles,
            index_mode=args.index_mode,
        )
        for value in representations
    ]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
