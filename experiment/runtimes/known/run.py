"""Run the known-environment full-volume LiuQP comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "src" / "core"
SRC = ROOT / "src"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(SRC))

from controller import ProtocolLiuQPController  # noqa: E402
from known_volume import build_volume_cover, coverage_audit  # noqa: E402
from robot import (  # noqa: E402
    DT,
    JOINT_NAMES,
    BoxObstacle,
    SceneDefinition,
    attachment_position,
    build_model,
    build_robot_certificate,
    set_configuration,
)
from voxel_index import NativeMultilevelMVT  # noqa: E402

MAXIMUM_CELL_SPAN = 0.075
MAXIMUM_CYCLES = 3000
SUCCESS_TOLERANCE = 0.001
SUCCESS_HOLD_CYCLES = 50
SAFETY_MARGIN = 0.006
NEAR_DISTANCE = 0.04
CONTACT_DISTANCE = 0.0
MVT_BASE_VOXEL_SIZE = 0.015
MVT_OUTWARD_PADDING = 5e-6


def load_scene() -> SceneDefinition:
    payload = json.loads((ROOT / "assets" / "scene.json").read_text(encoding="utf-8"))
    return SceneDefinition(
        name=payload["name"],
        description=payload["description"],
        boxes=tuple(
            BoxObstacle(
                name=item["name"],
                center=tuple(item["center"]),
                half_size=tuple(item["half_size"]),
                material=item["material"],
            )
            for item in payload["boxes"]
        ),
        q0=tuple(payload["q0"]),
        waypoints=tuple(tuple(item) for item in payload["waypoints"]),
        duration=float(payload["duration"]),
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    paths = [
        ROOT / "run.py",
        ROOT / "src" / "known_volume.py",
        CORE / "controller.py",
        CORE / "geometry.py",
        CORE / "native_support.py",
        CORE / "robot.py",
        CORE / "voxel_index.py",
        ROOT / "native" / "collision.dll",
        ROOT / "assets" / "drawer.xml",
        ROOT / "assets" / "scene.json",
    ]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in paths}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_or_none(value: float):
    return float(value) if np.isfinite(value) else None


def run(
    representation: str,
    *,
    maximum_cycles: int = MAXIMUM_CYCLES,
    osqp_max_iterations: int = 6000,
) -> dict:
    np.random.seed(0)
    if representation not in {"sphere", "ellipsoid"}:
        raise ValueError("representation must be sphere or ellipsoid")
    if maximum_cycles <= 0:
        raise ValueError("maximum_cycles must be positive")
    if osqp_max_iterations <= 0:
        raise ValueError("osqp_max_iterations must be positive")

    output = ROOT / "results" / representation
    output.mkdir(parents=True, exist_ok=True)
    scene = load_scene()
    cover = build_volume_cover(scene.boxes, representation, MAXIMUM_CELL_SPAN)
    audit = coverage_audit(cover)
    if not audit["complete_solid_cell_volume_covered"]:
        raise AssertionError("full-volume proxy construction failed its 8-corner proof")

    model = build_model(scene)
    data = mujoco.MjData(model)
    initial_q = np.asarray(scene.q0, dtype=float)
    set_configuration(model, data, initial_q)
    initial_ee = attachment_position(model, data)
    target = np.asarray(scene.waypoints[-1], dtype=float)
    robot = build_robot_certificate(model)
    robot_radii = np.asarray([item.radius for item in robot], dtype=float)

    query_padding = NEAR_DISTANCE + SAFETY_MARGIN + MVT_OUTWARD_PADDING
    index = NativeMultilevelMVT(
        cover.centers,
        cover.aabb_half_extents + MVT_OUTWARD_PADDING,
        base_voxel_size=MVT_BASE_VOXEL_SIZE,
        maximum_query_half_extent=float(np.max(robot_radii) + query_padding),
        query_padding=query_padding,
        simd=True,
    )
    common = dict(
        model=model,
        data=data,
        scene=scene,
        robot_spheres=robot,
        obstacle_centers=cover.centers,
        representation=representation,
        obstacle_offsets=np.zeros(len(cover.centers)),
        proxy_ids=cover.proxy_ids,
        safety_margin=SAFETY_MARGIN,
        near_distance=NEAR_DISTANCE,
        contact_distance=CONTACT_DISTANCE,
        obstacle_index=index,
        redundant_plane_pruning=True,
        ellipsoid_pair_threads=8,
        ellipsoid_pair_affinity_mask=0,
    )
    if representation == "sphere":
        controller = ProtocolLiuQPController(**common, obstacle_radii=cover.sphere_radii)
    else:
        controller = ProtocolLiuQPController(**common, obstacle_shapes=cover.ellipsoid_shapes)
    controller.task_gain = 2.0
    controller.max_task_speed = 0.18
    controller.osqp_adaptive_row_threshold = 512
    controller.osqp_absolute_tolerance = 1e-4
    controller.osqp_relative_tolerance = 1e-4
    controller.osqp_max_iterations = int(osqp_max_iterations)

    cycles: list[dict] = []
    q_history = [initial_q.copy()]
    ee_history = [initial_ee.copy()]
    success_hold = 0
    maximum_success_hold = 0
    first_success_cycle = None
    penetrating_cycles = 0
    maximum_penetration = 0.0
    started = time.perf_counter()
    try:
        for cycle in range(maximum_cycles):
            cycle_started = time.perf_counter()
            q_before = data.qpos[: len(JOINT_NAMES)].copy()
            qdot, metrics = controller.solve(target)
            q_after = q_before + qdot * DT
            set_configuration(model, data, q_after)
            ee = attachment_position(model, data)
            error = float(np.linalg.norm(target - ee))
            penetrating = [
                index_contact
                for index_contact in range(data.ncon)
                if float(data.contact[index_contact].dist) < -1e-8
            ]
            minimum_contact = min(
                (float(data.contact[index_contact].dist) for index_contact in range(data.ncon)),
                default=float("inf"),
            )
            if penetrating:
                penetrating_cycles += 1
            maximum_penetration = min(maximum_penetration, minimum_contact)
            if error < SUCCESS_TOLERANCE and not penetrating:
                success_hold += 1
                maximum_success_hold = max(maximum_success_hold, success_hold)
                if success_hold == SUCCESS_HOLD_CYCLES and first_success_cycle is None:
                    first_success_cycle = cycle - SUCCESS_HOLD_CYCLES + 1
            else:
                success_hold = 0

            elapsed_ms = (time.perf_counter() - cycle_started) * 1000.0
            row = {
                "cycle": cycle,
                "time_s": cycle * DT,
                "representation": representation,
                "ee_x": float(ee[0]),
                "ee_y": float(ee[1]),
                "ee_z": float(ee[2]),
                "error_m": error,
                "status": metrics.status,
                "exact_contact_count": int(data.ncon),
                "exact_penetrating_contact_count": len(penetrating),
                "minimum_contact_distance_m": finite_or_none(minimum_contact),
                "control_wall_ms": elapsed_ms,
                **metrics.as_dict(),
            }
            cycles.append(row)
            q_history.append(q_after.copy())
            ee_history.append(ee.copy())
    finally:
        index.close()
    wall_duration = time.perf_counter() - started

    np.save(output / "q_history.npy", np.asarray(q_history))
    np.save(output / "ee_history.npy", np.asarray(ee_history))
    np.savez_compressed(
        output / "proxies.npz",
        representation=np.asarray(representation),
        centers=cover.centers,
        cell_half_extents=cover.cell_half_extents,
        box_indices=cover.box_indices,
        proxy_ids=cover.proxy_ids,
        sphere_radii=cover.sphere_radii,
        ellipsoid_shapes=cover.ellipsoid_shapes,
        aabb_half_extents=cover.aabb_half_extents,
    )
    write_csv(output / "cycles.csv", cycles)
    (output / "coverage.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "scene.xml").write_bytes((ROOT / "assets" / "drawer.xml").read_bytes())

    errors = np.asarray([row["error_m"] for row in cycles])
    controller_ms = np.asarray([row["total_controller_ms"] for row in cycles])
    wall_ms = np.asarray([row["control_wall_ms"] for row in cycles])
    candidate_pairs = np.asarray([row["broadphase_candidate_pairs"] for row in cycles])
    qp_iterations = np.asarray([row["qp_iterations"] for row in cycles])
    summary = {
        "experiment": "known_environment_complete_solid_volume_cover_liuqp",
        "representation": representation,
        "environment_known": True,
        "depth_camera_used": False,
        "point_cloud_used": False,
        "online_voxel_map_used": False,
        "path_planner": None,
        "intermediate_targets": None,
        "random_dither": False,
        "truth_collision_backtracking": False,
        "scene_name": scene.name,
        "maximum_cell_span_m": MAXIMUM_CELL_SPAN,
        "solid_partition_shared_between_representations": True,
        "proxy_count": len(cover.centers),
        "robot_certificate_spheres": len(robot),
        "complete_solid_cell_volume_covered": audit["complete_solid_cell_volume_covered"],
        "coverage_maximum_corner_measure": audit["maximum_corner_measure"],
        "primitive_to_box_volume_sum_ratio": audit["primitive_to_box_volume_sum_ratio"],
        "ellipsoid_axes_rule": "sqrt(3) times each solid sub-box half extent" if representation == "ellipsoid" else None,
        "sphere_radius_rule": "solid sub-box half diagonal" if representation == "sphere" else None,
        "mvt_multilevel_enabled": True,
        "mvt_simd_enabled": True,
        "mvt_level_count": index.stats.level_count,
        "mvt_index_references": index.stats.index_references,
        "mvt_cell_lookups_per_robot_query": index.stats.cell_lookups_per_query,
        "cycles": len(cycles),
        "simulated_duration_s": len(cycles) * DT,
        "wall_duration_s": wall_duration,
        "initial_ee_m": initial_ee.tolist(),
        "target_m": target.tolist(),
        "final_error_m": float(errors[-1]),
        "minimum_error_m": float(np.min(errors)),
        "maximum_ee_x_m": float(np.max(np.asarray(ee_history)[:, 0])),
        "success_tolerance_m": SUCCESS_TOLERANCE,
        "success_hold_cycles": SUCCESS_HOLD_CYCLES,
        "maximum_success_hold_cycles": maximum_success_hold,
        "success": maximum_success_hold >= SUCCESS_HOLD_CYCLES,
        "first_sustained_success_time_s": None if first_success_cycle is None else first_success_cycle * DT,
        "exact_penetrating_cycles": penetrating_cycles,
        "maximum_penetration_m": maximum_penetration,
        "controller_ms_p50": float(np.percentile(controller_ms, 50)),
        "controller_ms_p95": float(np.percentile(controller_ms, 95)),
        "controller_ms_p99": float(np.percentile(controller_ms, 99)),
        "control_wall_ms_p99": float(np.percentile(wall_ms, 99)),
        "broadphase_candidate_pairs_p50": float(np.percentile(candidate_pairs, 50)),
        "broadphase_candidate_pairs_p99": float(np.percentile(candidate_pairs, 99)),
        "qp_iterations_p50": float(np.percentile(qp_iterations, 50)),
        "qp_iterations_p99": float(np.percentile(qp_iterations, 99)),
        "safety_margin_m": SAFETY_MARGIN,
        "near_distance_m": NEAR_DISTANCE,
        "contact_distance_m": CONTACT_DISTANCE,
        "osqp_max_iterations": int(controller.osqp_max_iterations),
        "controller_origin": "frozen original v4.3/v4.4 shared LiuQP controller",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "source_hashes": source_hashes(),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("representation", choices=("sphere", "ellipsoid"))
    parser.add_argument("--cycles", type=int, default=MAXIMUM_CYCLES)
    parser.add_argument("--osqp-max-iterations", type=int, default=6000)
    args = parser.parse_args()
    run(
        args.representation,
        maximum_cycles=args.cycles,
        osqp_max_iterations=args.osqp_max_iterations,
    )


if __name__ == "__main__":
    main()
