"""Path-free static D2 geometry gate for the formal protocol-v3 scene."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares

from ellipsoid_model import closest_point_on_ellipsoid
from model import (
    DT,
    JOINT_NAMES,
    attachment_position,
    build_model,
    build_robot_certificate,
    build_xml,
    certificate_world_state,
    set_configuration,
)
from protocol_drawer_scene import (
    DRAWER_Y,
    DRAWER_Z,
    FRONT_X,
    INNER_HALF_Y,
    INNER_HALF_Z,
    KNOWN_PROXY_CELL_SIZE,
    SAFETY_MARGIN,
    SUCCESS_TOLERANCE,
    formal_protocol_scene,
    protocol_drawer_v1_scene,
)
from protocol_known_proxies import (
    audit_known_proxy_cell_coverage,
    build_known_matched_proxy_tree,
)
from protocol_liuqp_controller import ProtocolLiuQPController


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "formal_results" / "final_two_camera" / "static_geometry_gate_final"
)
RNG_SEED = 90217
IK_TOLERANCE = 0.0015
SPHERE_CLOSURE_REQUIRED = 0.005
ELLIPSOID_OPEN_REQUIRED = 0.002
GRID_SIZE = 161


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _collision_free(model, data, q: np.ndarray) -> bool:
    set_configuration(model, data, q)
    return all(data.contact[index].dist >= -1.0e-8 for index in range(data.ncon))


def _contact_records(model, data):
    return [
        {
            "geom1": model.geom(int(data.contact[index].geom1)).name,
            "geom2": model.geom(int(data.contact[index].geom2)).name,
            "distance_m": float(data.contact[index].dist),
        }
        for index in range(data.ncon)
    ]


def _solve_position_ik(model, data, seed, target, lower, upper):
    seed = np.clip(np.asarray(seed), lower + 1.0e-5, upper - 1.0e-5)

    def residual(q):
        set_configuration(model, data, q)
        return attachment_position(model, data) - target

    result = least_squares(
        residual,
        seed,
        bounds=(lower + 1.0e-5, upper - 1.0e-5),
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
        max_nfev=240,
    )
    return result.x.copy(), float(np.linalg.norm(residual(result.x)))


def _find_contact_free_ik(
    model,
    data,
    target,
    lower,
    upper,
    rng,
    attempts,
    maximum_records,
):
    records = []
    for attempt in range(attempts):
        seed = rng.uniform(lower + 0.01, upper - 0.01)
        q, error = _solve_position_ik(
            model, data, seed, target, lower, upper
        )
        if error > IK_TOLERANCE or not _collision_free(model, data, q):
            continue
        if all(np.linalg.norm(q - np.asarray(row["q"])) > 0.08 for row in records):
            records.append(
                {
                    "attempt": attempt,
                    "error_m": error,
                    "q": q.tolist(),
                    "achieved_m": attachment_position(model, data).tolist(),
                    "contacts": _contact_records(model, data),
                }
            )
        if len(records) >= maximum_records:
            break
    return records


def _effective_attachment_sphere(model, robot):
    site_id = model.site("attachment_site").id
    body_id = int(model.site_bodyid[site_id])
    site_local = model.site_pos[site_id]
    candidates = []
    for index, proxy in enumerate(robot):
        if proxy.body_id != body_id:
            continue
        offset = float(np.linalg.norm(proxy.local_center - site_local))
        candidates.append((float(proxy.radius - offset), index, offset))
    return max(candidates)


def _sphere_cross_section(proxies, effective_radius):
    ys = np.linspace(
        DRAWER_Y - INNER_HALF_Y,
        DRAWER_Y + INNER_HALF_Y,
        GRID_SIZE,
    )
    zs = np.linspace(
        DRAWER_Z - INNER_HALF_Z,
        DRAWER_Z + INNER_HALF_Z,
        GRID_SIZE,
    )
    yz = np.array(np.meshgrid(ys, zs, indexing="ij")).reshape(2, -1).T
    cover = 0.5 * math.sqrt(
        (ys[1] - ys[0]) ** 2 + (zs[1] - zs[0]) ** 2
    )
    scans = []
    fields = {}
    for plane_x in np.arange(FRONT_X, 0.651, 0.01):
        points = np.c_[np.full(len(yz), plane_x), yz]
        clearance = np.full(len(points), np.inf)
        for indices in np.array_split(np.arange(len(points)), 32):
            clearance[indices] = np.min(
                np.linalg.norm(
                    points[indices, None, :]
                    - proxies.centers[None, :, :],
                    axis=2,
                )
                - proxies.sphere_radii[None, :]
                - effective_radius
                - SAFETY_MARGIN,
                axis=1,
            )
        row = {
            "plane_x_m": float(plane_x),
            "sampled_max_clearance_m": float(np.max(clearance)),
            "lipschitz_cover_radius_m": float(cover),
            "continuous_upper_bound_m": float(np.max(clearance) + cover),
        }
        scans.append(row)
        fields[float(plane_x)] = clearance.reshape(GRID_SIZE, GRID_SIZE)
    selected = next(
        (
            row
            for row in scans
            if row["continuous_upper_bound_m"]
            <= -SPHERE_CLOSURE_REQUIRED
        ),
        min(scans, key=lambda row: row["continuous_upper_bound_m"]),
    )
    return ys, zs, scans, selected, fields[selected["plane_x_m"]]


def _full_ellipsoid_clearance(model, data, robot, proxies, q):
    set_configuration(model, data, q)
    positions, _, radii = certificate_world_state(model, data, robot)
    maximum_axes = np.sqrt(
        np.max(np.linalg.eigvalsh(proxies.ellipsoid_shapes), axis=1)
    )
    best = math.inf
    limiting = None
    maximum_residual = 0.0
    exact_pair_count = 0
    for robot_index, (position, radius) in enumerate(zip(positions, radii)):
        lower_bounds = (
            np.linalg.norm(proxies.centers - position, axis=1)
            - maximum_axes
            - float(radius)
            - SAFETY_MARGIN
        )
        for obstacle_index in np.argsort(lower_bounds):
            if lower_bounds[obstacle_index] >= best:
                break
            closest = closest_point_on_ellipsoid(
                proxies.centers[obstacle_index],
                proxies.ellipsoid_shapes[obstacle_index],
                position,
            )
            exact_pair_count += 1
            clearance = (
                float(np.linalg.norm(closest.surface_point - position))
                - float(radius)
                - SAFETY_MARGIN
            )
            maximum_residual = max(maximum_residual, closest.residual)
            if clearance < best:
                best = clearance
                limiting = (robot_index, int(obstacle_index))
    return float(best), limiting, float(maximum_residual), exact_pair_count


def _find_safe_crossing(
    model,
    data,
    robot,
    proxies,
    point,
    lower,
    upper,
    rng,
    attempts,
):
    best = None
    for attempt in range(attempts):
        seed = rng.uniform(lower + 0.01, upper - 0.01)
        q, error = _solve_position_ik(
            model, data, seed, point, lower, upper
        )
        if error > IK_TOLERANCE or not _collision_free(model, data, q):
            continue
        clearance, limiting, residual, exact_pairs = _full_ellipsoid_clearance(
            model, data, robot, proxies, q
        )
        record = {
            "attempt": attempt,
            "error_m": error,
            "q": q.tolist(),
            "achieved_m": attachment_position(model, data).tolist(),
            "exact_contact_free": True,
            "full_pair_min_clearance_m": clearance,
            "limiting_pair": list(limiting),
            "maximum_closest_point_residual": residual,
            "exact_pairs_evaluated_after_safe_bounds": exact_pairs,
        }
        if best is None or clearance > best["full_pair_min_clearance_m"]:
            best = record
        if clearance >= ELLIPSOID_OPEN_REQUIRED:
            break
    return best


def _no_obstacle_qp_gate(scene):
    empty_scene = replace(
        scene,
        name=f"{scene.name}_no_obstacles",
        boxes=(),
    )
    model = build_model(empty_scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(scene.q0, dtype=float))
    robot = build_robot_certificate(model)
    controller = ProtocolLiuQPController(
        model,
        data,
        empty_scene,
        robot,
        np.empty((0, 3)),
        representation="sphere",
        obstacle_radii=np.empty(0),
        proxy_ids=np.empty(0, dtype=np.int64),
        safety_margin=SAFETY_MARGIN,
    )
    target = np.asarray(scene.waypoints[-1], dtype=float)
    errors = []
    statuses = []
    q_history = []
    for _ in range(int(scene.duration / DT)):
        qdot, metrics = controller.solve(target)
        statuses.append(metrics.status)
        data.qpos[: len(JOINT_NAMES)] += qdot * DT
        set_configuration(model, data, data.qpos[: len(JOINT_NAMES)])
        q_history.append(data.qpos[: len(JOINT_NAMES)].copy())
        errors.append(
            float(np.linalg.norm(attachment_position(model, data) - target))
        )
        if errors[-1] <= SUCCESS_TOLERANCE:
            break
    return (
        {
            "success": bool(errors[-1] <= SUCCESS_TOLERANCE),
            "cycles": len(errors),
            "first_success_s": (
                float((len(errors) - 1) * DT)
                if errors[-1] <= SUCCESS_TOLERANCE
                else None
            ),
            "final_error_m": float(errors[-1]),
            "solved_fraction": float(
                np.mean([value.startswith("solved") for value in statuses])
            ),
        },
        np.asarray(q_history),
        np.asarray(errors),
    )


def audit():
    started = time.perf_counter()
    scene = formal_protocol_scene()
    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(scene.q0, dtype=float))
    lower = np.array(
        [model.jnt_range[model.joint(name).id, 0] for name in JOINT_NAMES]
    )
    upper = np.array(
        [model.jnt_range[model.joint(name).id, 1] for name in JOINT_NAMES]
    )
    robot = build_robot_certificate(model)
    proxies = build_known_matched_proxy_tree(
        scene.boxes,
        cell_size=KNOWN_PROXY_CELL_SIZE,
    )
    coverage = audit_known_proxy_cell_coverage(proxies)
    target = np.asarray(scene.waypoints[-1], dtype=float)

    target_ik = _find_contact_free_ik(
        model,
        data,
        target,
        lower,
        upper,
        np.random.default_rng(RNG_SEED + 1),
        attempts=900,
        maximum_records=10,
    )
    effective_radius, effective_index, effective_offset = (
        _effective_attachment_sphere(model, robot)
    )
    ys, zs, scans, selected, sphere_field = _sphere_cross_section(
        proxies,
        effective_radius,
    )
    crossing_point = np.array(
        [selected["plane_x_m"], DRAWER_Y, DRAWER_Z],
        dtype=float,
    )
    crossing = _find_safe_crossing(
        model,
        data,
        robot,
        proxies,
        crossing_point,
        lower,
        upper,
        np.random.default_rng(RNG_SEED),
        attempts=500,
    )
    no_obstacle, no_obstacle_q, no_obstacle_error = _no_obstacle_qp_gate(
        scene
    )

    rejected = protocol_drawer_v1_scene()
    rejected_proxies = build_known_matched_proxy_tree(
        rejected.boxes,
        cell_size=KNOWN_PROXY_CELL_SIZE,
    )
    _, _, rejected_scan, rejected_best, _ = _sphere_cross_section(
        rejected_proxies,
        effective_radius,
    )

    xml = build_xml(scene).encode("utf-8")
    config = json.dumps(
        {
            "scene": scene.name,
            "q0": scene.q0,
            "target": scene.waypoints[-1],
            "cell_size": KNOWN_PROXY_CELL_SIZE,
            "safety": SAFETY_MARGIN,
            "seed": RNG_SEED,
            "grid_size": GRID_SIZE,
            "sphere_closure_required": SPHERE_CLOSURE_REQUIRED,
            "ellipsoid_open_required": ELLIPSOID_OPEN_REQUIRED,
        },
        sort_keys=True,
    ).encode("utf-8")
    passed = bool(
        len(target_ik) > 0
        and no_obstacle["success"]
        and selected["continuous_upper_bound_m"]
        <= -SPHERE_CLOSURE_REQUIRED
        and crossing is not None
        and crossing["full_pair_min_clearance_m"]
        >= ELLIPSOID_OPEN_REQUIRED
        and coverage["sphere_covers_all_cells"]
        and coverage["ellipsoid_covers_all_cells"]
    )
    result = {
        "passed": passed,
        "scene": scene.name,
        "evidence_role": "path_free_static_geometry_gate",
        "no_path_planner": True,
        "continuous_reachability_deferred_to_actual_ellipsoid_liuqp": True,
        "collision_free_goal_ik_count": len(target_ik),
        "goal_ik": target_ik,
        "no_obstacle_liuqp": no_obstacle,
        "robot_sphere_count": len(robot),
        "environment_proxy_count": len(proxies.centers),
        "coverage": coverage,
        "orientation_independent_effective_sphere": {
            "robot_proxy_index": int(effective_index),
            "proxy_radius_m": float(robot[effective_index].radius),
            "offset_from_attachment_m": float(effective_offset),
            "effective_radius_m": float(effective_radius),
            "proof": "triangle inequality for every wrist rotation",
        },
        "sphere_cross_section": {
            "selected": selected,
            "required_closure_margin_m": SPHERE_CLOSURE_REQUIRED,
            "strictly_closed": bool(
                selected["continuous_upper_bound_m"]
                <= -SPHERE_CLOSURE_REQUIRED
            ),
            "scan": scans,
        },
        "ellipsoid_crossing": crossing,
        "required_ellipsoid_open_margin_m": ELLIPSOID_OPEN_REQUIRED,
        "rejected_v1": {
            "scene": rejected.name,
            "reason": "sphere continuous cross-section remained open",
            "best_section": rejected_best,
            "scan": rejected_scan,
        },
        "hashes": {
            "scene_xml_sha256": _sha256(xml),
            "proxy_snapshot_sha256": proxies.snapshot_sha256,
            "config_sha256": _sha256(config),
            "audit_source_sha256": _sha256(Path(__file__).read_bytes()),
        },
        "elapsed_s": float(time.perf_counter() - started),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "scene.xml").write_bytes(xml)
    (OUTPUT / "static_geometry_gate.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        OUTPUT / "cross_section_fields.npz",
        y=ys,
        z=zs,
        sphere_clearance=sphere_field,
        selected_x=np.array([selected["plane_x_m"]]),
    )
    np.savez_compressed(
        OUTPUT / "no_obstacle_liuqp.npz",
        q=no_obstacle_q,
        error=no_obstacle_error,
    )
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
