"""Audit what the frozen v4.4 sphere run does and does not prove.

The audit reconstructs the first failed control cycle from immutable logs,
checks the reconstructed kinematics against the recorded end-effector and
pair clearance, measures the local fixed-line obstruction, and separately
tests the entire physical y-z section.  The distinction prevents a local QP
deadlock from being misreported as a globally closed geometric passage.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from certify_formal_paired_cross_section import _sphere_section
from model import (
    attachment_position,
    build_robot_certificate,
    certificate_world_state,
    set_configuration,
)


SAFETY_MARGIN_M = 0.006


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def xml_vector(element: ET.Element, name: str) -> np.ndarray:
    return np.asarray([float(value) for value in element.attrib[name].split()])


def opening_from_xml(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    geoms = {element.attrib.get("name"): element for element in root.iter("geom")}
    sites = {element.attrib.get("name"): element for element in root.iter("site")}
    left = geoms["drawer_left"]
    right = geoms["drawer_right"]
    bottom = geoms["drawer_bottom"]
    ceiling = geoms["drawer_ceiling"]
    left_pos, left_size = xml_vector(left, "pos"), xml_vector(left, "size")
    right_pos, right_size = xml_vector(right, "pos"), xml_vector(right, "size")
    bottom_pos, bottom_size = xml_vector(bottom, "pos"), xml_vector(bottom, "size")
    ceiling_pos, ceiling_size = xml_vector(ceiling, "pos"), xml_vector(ceiling, "size")
    lower = np.array(
        [left_pos[1] + left_size[1], bottom_pos[2] + bottom_size[2]],
        dtype=float,
    )
    upper = np.array(
        [right_pos[1] - right_size[1], ceiling_pos[2] - ceiling_size[2]],
        dtype=float,
    )
    target = xml_vector(sites["liuqp_target"], "pos")
    return lower, upper, target


def merge_intervals(intervals: list[tuple[float, float]], lo: float, hi: float):
    clipped = sorted((max(lo, a), min(hi, b)) for a, b in intervals if b >= lo and a <= hi)
    merged: list[list[float]] = []
    for a, b in clipped:
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    uncovered: list[list[float]] = []
    cursor = lo
    for a, b in merged:
        if a > cursor:
            uncovered.append([cursor, a])
        cursor = max(cursor, b)
    if cursor < hi:
        uncovered.append([cursor, hi])
    return merged, uncovered


def vertical_line_audit(
    centers: np.ndarray,
    effective_radii: np.ndarray,
    x: float,
    y: float,
    robot_radius: float,
    z_lo: float,
    z_hi: float,
) -> dict:
    intervals: list[tuple[float, float]] = []
    sources: list[dict] = []
    total = effective_radii + float(robot_radius) + SAFETY_MARGIN_M
    rho2 = (centers[:, 0] - x) ** 2 + (centers[:, 1] - y) ** 2
    hit = np.flatnonzero(rho2 <= total * total)
    for index in hit:
        dz = float(np.sqrt(max(total[index] ** 2 - rho2[index], 0.0)))
        interval = (float(centers[index, 2] - dz), float(centers[index, 2] + dz))
        if interval[1] >= z_lo and interval[0] <= z_hi:
            intervals.append(interval)
            sources.append(
                {
                    "proxy_index": int(index),
                    "center_m": centers[index].tolist(),
                    "effective_obstacle_radius_m": float(effective_radii[index]),
                    "forbidden_z_interval_m": list(interval),
                }
            )
    merged, uncovered = merge_intervals(intervals, z_lo, z_hi)
    return {
        "x_m": float(x),
        "y_m": float(y),
        "physical_z_interval_m": [float(z_lo), float(z_hi)],
        "intersecting_proxy_count": int(len(sources)),
        "merged_forbidden_intervals_m": merged,
        "uncovered_intervals_m": uncovered,
        "strictly_closed_on_this_fixed_xy_line": bool(not uncovered),
        "maximum_uncovered_interval_width_m": float(
            max((b - a for a, b in uncovered), default=0.0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grid-size", type=int, default=301)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    cycles = read_rows(run_dir / "cycles.csv")
    status_counts = Counter(row["status"] for row in cycles)
    failed = [row for row in cycles if row["status"] != "solved"]
    if not failed:
        raise AssertionError("frozen sphere run has no failed QP cycle")
    first = failed[0]
    cycle = int(first["cycle"])
    generation = int(first["active_generation"])

    archive = np.load(run_dir / "causal_proxy_snapshots.npz")
    starts = np.asarray(archive["offsets"], dtype=np.int64)
    start, stop = map(int, starts[generation : generation + 2])
    snapshot = {name: np.asarray(archive[name][start:stop]) for name in (
        "proxy_ids", "centers", "sphere_radii", "uncertainty_offsets"
    )}

    q_history = np.asarray(np.load(run_dir / "q_history.npy"), dtype=float)
    model = mujoco.MjModel.from_xml_path(str(run_dir / "scene.xml"))
    data = mujoco.MjData(model)
    set_configuration(model, data, q_history[cycle])
    certificate = build_robot_certificate(model)
    robot_centers, _, robot_radii = certificate_world_state(model, data, certificate)
    robot_index = int(first["limiting_robot_index"])
    obstacle_index = int(first["limiting_obstacle_index"])
    if robot_index < 0 or obstacle_index < 0:
        raise AssertionError("first failed cycle did not record a limiting pair")

    lower, upper, target = opening_from_xml(run_dir / "scene.xml")
    recorded_ee = np.asarray(
        [float(first["ee_x"]), float(first["ee_y"]), float(first["ee_z"])],
        dtype=float,
    )
    reconstructed_ee = attachment_position(model, data)
    ee_reconstruction_error = float(np.linalg.norm(recorded_ee - reconstructed_ee))

    pair_rows = []
    with (run_dir / "pair_states.csv").open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            row_cycle = int(row["cycle"])
            if row_cycle == cycle:
                pair_rows.append(row)
            elif row_cycle > cycle and pair_rows:
                break
    matching = [
        row for row in pair_rows
        if int(row["robot_index"]) == robot_index
        and int(row["obstacle_index"]) == obstacle_index
    ]
    if len(matching) != 1:
        raise AssertionError("limiting pair was not uniquely present in pair_states")
    pair = matching[0]
    center_gap = float(
        np.linalg.norm(robot_centers[robot_index] - snapshot["centers"][obstacle_index])
        - robot_radii[robot_index]
        - snapshot["sphere_radii"][obstacle_index]
        - snapshot["uncertainty_offsets"][obstacle_index]
    )
    logged_surface_gap = float(pair["surface_clearance"])
    gap_reconstruction_error = abs(center_gap - logged_surface_gap)
    if ee_reconstruction_error > 1.0e-8 or gap_reconstruction_error > 1.0e-8:
        raise AssertionError(
            "current kinematic reconstruction does not match frozen logs: "
            f"ee={ee_reconstruction_error}, gap={gap_reconstruction_error}"
        )

    effective = snapshot["sphere_radii"] + snapshot["uncertainty_offsets"]
    local_line = vertical_line_audit(
        snapshot["centers"],
        effective,
        float(robot_centers[robot_index, 0]),
        float(robot_centers[robot_index, 1]),
        float(robot_radii[robot_index]),
        float(lower[1]),
        float(upper[1]),
    )
    full_section = _sphere_section(
        snapshot,
        float(robot_centers[robot_index, 0]),
        lower,
        upper,
        float(robot_radii[robot_index]),
        args.grid_size,
    )

    zero = np.asarray([float(row["executed_qdot_norm"]) <= 1.0e-12 for row in cycles])
    longest = 0
    current = 0
    for value in zero:
        current = current + 1 if value else 0
        longest = max(longest, current)
    rows_after = cycles[cycle:]
    post_failure_status = Counter(row["status"] for row in rows_after)
    result = {
        "analysis": "frozen_v4.4_first_failure_local_vs_global_geometry",
        "run_dir": str(run_dir),
        "target_m": target.tolist(),
        "status_counts_all_cycles": dict(status_counts),
        "first_failed_cycle": cycle,
        "first_failed_time_s": float(first["time_s"]),
        "active_generation": generation,
        "snapshot_source_cycle": int(first["snapshot_source_cycle"]),
        "snapshot_age_ms": float(first["snapshot_age_ms"]),
        "post_failure_status_counts": dict(post_failure_status),
        "post_failure_cycles": int(len(rows_after)),
        "post_failure_zero_velocity_cycles": int(
            sum(float(row["executed_qdot_norm"]) <= 1.0e-12 for row in rows_after)
        ),
        "longest_zero_velocity_run_cycles": int(longest),
        "reconstruction_validation": {
            "robot_certificate_spheres": int(len(certificate)),
            "ee_position_error_m": ee_reconstruction_error,
            "limiting_surface_gap_error_m": gap_reconstruction_error,
            "passed_1e-8": True,
        },
        "limiting_pair": {
            "robot_index": robot_index,
            "robot_body": certificate[robot_index].body_name,
            "robot_center_m": robot_centers[robot_index].tolist(),
            "robot_radius_m": float(robot_radii[robot_index]),
            "obstacle_index": obstacle_index,
            "proxy_id": int(snapshot["proxy_ids"][obstacle_index]),
            "obstacle_center_m": snapshot["centers"][obstacle_index].tolist(),
            "obstacle_sphere_radius_m": float(snapshot["sphere_radii"][obstacle_index]),
            "uncertainty_offset_m": float(snapshot["uncertainty_offsets"][obstacle_index]),
            "raw_surface_gap_m": logged_surface_gap,
            "hard_barrier_clearance_m": float(pair["clearance"]),
            "state": pair["state"],
            "qp_collision_row": pair["qp_collision_row"].lower() == "true",
            "near_penalty": pair["near_penalty"].lower() == "true",
            "contact_recovery_row": pair["contact_recovery_row"].lower() == "true",
        },
        "local_fixed_xy_vertical_line": local_line,
        "entire_physical_yz_section_at_same_x": full_section,
        "allowed_conclusion": (
            "the recorded LiuQP state enters a persistent hard-QP failure and the "
            "current limiting robot-sphere line is locally blocked"
        ),
        "forbidden_conclusion": (
            "the entire drawer-front geometry is closed to every perturbed "
            "configuration; the full-section audit must be strictly_closed for that"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
