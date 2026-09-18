"""Formal protocol-v3 causal moving-local-depth LiuQP experiment.

The controller receives one final Cartesian target and only proxies built from
depth frames available at that cycle.  There is no intermediate target,
precomputed route, global planner, random perturbation, or MuJoCo-truth
acceptance test.  Execution is additionally gated by the live
free/occupied/unknown swept-volume certificate.
"""

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

from depth_camera_perception import (
    DepthObservation,
    UR5MountedDepthCamera,
    environment_endpoint_mask,
)
from incremental_proxy_manager import IncrementalMatchedProxyManager
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
from native_mvt import NativeMultilevelMVT
from native_occupancy import NativeIncrementalOccupancyMap
from protocol_drawer_scene import (
    CAMERA_HZ,
    CONTACT_DISTANCE,
    INITIAL_CALIBRATED_FREE_PADDING,
    NEAR_DISTANCE,
    SAFETY_MARGIN,
    SUCCESS_TOLERANCE,
    formal_protocol_scene,
)
from protocol_liuqp_controller import ProtocolLiuQPController


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "formal_results" / "protocol_v6" / "online_ablation"
SUCCESS_HOLD_CYCLES = 10
MAP_VOXEL_SIZE = 0.005
CENTERVOX_SIZE = 0.015
PROXY_CLUSTER_SIZE = 0.10
MAXIMUM_AABB_OVERSHOOT = 0.025
MVT_BASE_VOXEL_SIZE = 0.015
# Native MVT stores AABBs and query radii as float32. Expand only the
# broadphase (never the collision geometry) so a float64 boundary contact
# cannot round inward and disappear from the candidate set. Extra rows are
# harmless false positives removed by the exact narrow phase.
MVT_NUMERICAL_OUTWARD_PADDING = 5.0e-6
# NEAR_DISTANCE already exceeds the maximum observed 20 ms certificate-center
# displacement and is itself retained in every query.  Adding another 25 mm
# double-counted motion reach and sent provably non-binding far proxies to the
# exact Q+U narrow phase.
MOTION_BROADPHASE_PADDING = 0.0
CAMERA_PIXEL_STRIDE = 2
PROXY_WORKSPACE_LOWER = np.array([-0.25, -0.15, -0.02])
PROXY_WORKSPACE_UPPER = np.array([0.95, 0.95, 1.15])


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _first_nonfree_swept_voxel(
    occupancy: NativeIncrementalOccupancyMap,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
) -> dict | None:
    """One-shot Python replay of the native conservative swept-ball query."""

    keys, states = occupancy.snapshot_arrays()
    state_by_key = {
        tuple(int(value) for value in key): int(state)
        for key, state in zip(keys, states)
    }
    voxel = np.float32(occupancy.voxel_size)
    start32 = np.asarray(start, dtype=np.float32)
    end32 = np.asarray(end, dtype=np.float32)
    delta = end32 - start32
    distance = float(np.linalg.norm(delta))
    steps = max(1, int(np.ceil(distance / float(np.float32(0.45) * voxel))))
    cover = float(np.float32(radius) + np.float32(0.8660254037844386) * voxel)
    checked = set()
    for step in range(steps + 1):
        center = start32 + np.float32(step / steps) * delta
        lower = np.floor((center - cover) / voxel).astype(np.int32)
        upper = np.floor((center + cover) / voxel).astype(np.int32)
        for x in range(int(lower[0]), int(upper[0]) + 1):
            for y in range(int(lower[1]), int(upper[1]) + 1):
                for z in range(int(lower[2]), int(upper[2]) + 1):
                    voxel_center = (
                        np.array([x + 0.5, y + 0.5, z + 0.5]) * float(voxel)
                    )
                    if np.linalg.norm(voxel_center - center) > cover:
                        continue
                    key = (x, y, z)
                    if key in checked:
                        continue
                    checked.add(key)
                    state = state_by_key.get(key, 0)
                    if state != 1:
                        return {
                            "key": list(key),
                            "center_m": voxel_center.tolist(),
                            "state": "occupied" if state == 2 else "unknown",
                            "distance_to_start_center_m": float(
                                np.linalg.norm(voxel_center - start32)
                            ),
                        }
    return None


def _source_hashes() -> dict[str, str]:
    names = (
        "run_protocol_v3_online_ablation.py",
        "protocol_liuqp_controller.py",
        "protocol_drawer_scene.py",
        "depth_camera_perception.py",
        "incremental_proxy_manager.py",
        "incremental_occupancy_map.py",
        "native_occupancy.py",
        "native_mvt.py",
        "native_mvt/native_mvt.cpp",
        "ellipsoid_model.py",
        "model.py",
    )
    return {name: _sha256((ROOT / name).read_bytes()) for name in names}


def _crop_proxy_observation(
    observation: DepthObservation,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(observation.points, dtype=float)
    keep = environment_endpoint_mask(observation)
    keep &= np.all(points >= PROXY_WORKSPACE_LOWER, axis=1)
    keep &= np.all(points <= PROXY_WORKSPACE_UPPER, axis=1)
    return points[keep], np.asarray(observation.sample_radii, dtype=float)[keep]


def _environment_half_extents(proxies, representation: str) -> np.ndarray:
    offsets = np.asarray(proxies.proxy_offset_radii, dtype=float)
    if representation == "sphere":
        radii = np.asarray(proxies.sphere_radii, dtype=float) + offsets
        return np.repeat(radii[:, None], 3, axis=1)
    shapes = (
        proxies.ellipsoid_outer_shapes
        if proxies.proxy_uncertainty_shapes is not None
        else proxies.base_ellipsoid_shapes
    )
    return (
        np.sqrt(
            np.maximum(
                np.diagonal(
                    np.asarray(shapes), axis1=1, axis2=2
                ),
                0.0,
            )
        )
        + offsets[:, None]
    )


def _build_mvt_only(
    proxies,
    representation: str,
    robot_radii: np.ndarray,
    *,
    simd: bool,
) -> NativeMultilevelMVT:
    half = _environment_half_extents(proxies, representation)
    indexed_half = half + MVT_NUMERICAL_OUTWARD_PADDING
    query_padding = (
        NEAR_DISTANCE
        + SAFETY_MARGIN
        + MOTION_BROADPHASE_PADDING
        + MVT_NUMERICAL_OUTWARD_PADDING
    )
    maximum_query_half = float(np.max(robot_radii) + query_padding)
    table = NativeMultilevelMVT(
        proxies.centers,
        indexed_half,
        base_voxel_size=MVT_BASE_VOXEL_SIZE,
        maximum_query_half_extent=maximum_query_half,
        query_padding=query_padding,
        simd=simd,
    )
    return table


def _build_mvt_and_audit(
    proxies,
    representation: str,
    robot_positions: np.ndarray,
    robot_radii: np.ndarray,
    *,
    simd: bool,
) -> tuple[NativeMultilevelMVT, int]:
    half = _environment_half_extents(proxies, representation)
    query_padding = (
        NEAR_DISTANCE
        + SAFETY_MARGIN
        + MOTION_BROADPHASE_PADDING
        + MVT_NUMERICAL_OUTWARD_PADDING
    )
    table = _build_mvt_only(
        proxies,
        representation,
        robot_radii,
        simd=simd,
    )
    missing = 0
    actual_rows = table.query_spheres(robot_positions, robot_radii)
    # The brute-force oracle deliberately remains independent of the MVT.
    # Its obstacle AABBs are snapshot-invariant, so materialize them once
    # instead of allocating the same two (N, 3) arrays for every robot ball.
    lo = proxies.centers - half
    hi = proxies.centers + half
    for center, radius, actual in zip(
        robot_positions, robot_radii, actual_rows
    ):
        query_radius = float(radius) + query_padding
        delta = np.maximum(
            np.maximum(lo - center, center - hi),
            0.0,
        )
        expected = np.flatnonzero(
            np.einsum("ij,ij->i", delta, delta)
            <= query_radius * query_radius
        )
        missing += len(np.setdiff1d(expected, actual, assume_unique=True))
    if missing:
        table.close()
        raise AssertionError(
            f"multilevel MVT missed {missing} ball-AABB oracle candidates"
        )
    return table, missing


def _controller_uncertainty_shapes(proxies) -> np.ndarray | None:
    """Return U only for the legacy two-ellipsoid support-sum path.

    v4.2 publishes an explicit zero U because all uncertainty is already
    fused into one certified Q.  Treating that zero tensor as a second
    ellipsoid would unnecessarily select the support-normal optimizer instead
    of the exact safeguarded closest-point Newton/bisection implementation.
    """

    shapes = proxies.proxy_uncertainty_shapes
    if shapes is None:
        return None
    shapes = np.asarray(shapes, dtype=float)
    if not np.any(shapes):
        return None
    return shapes


def _new_controller(
    representation: str,
    model,
    data,
    scene,
    robot,
    proxies,
    obstacle_index,
    *,
    ellipsoid_pair_threads: int = 8,
    ellipsoid_pair_affinity_mask: int = 0,
) -> ProtocolLiuQPController:
    common = dict(
        model=model,
        data=data,
        scene=scene,
        robot_spheres=robot,
        obstacle_centers=proxies.centers,
        representation=representation,
        obstacle_offsets=proxies.proxy_offset_radii,
        proxy_ids=proxies.proxy_ids,
        safety_margin=SAFETY_MARGIN,
        near_distance=NEAR_DISTANCE,
        contact_distance=CONTACT_DISTANCE,
        obstacle_index=obstacle_index,
        redundant_plane_pruning=True,
        ellipsoid_pair_threads=ellipsoid_pair_threads,
        ellipsoid_pair_affinity_mask=ellipsoid_pair_affinity_mask,
    )
    if representation == "sphere":
        return ProtocolLiuQPController(
            **common, obstacle_radii=proxies.sphere_radii
        )
    return ProtocolLiuQPController(
        **common,
        obstacle_shapes=proxies.base_ellipsoid_shapes,
        obstacle_uncertainty_shapes=_controller_uncertainty_shapes(proxies),
    )


def run_one(
    representation: str,
    index_mode: str,
    output_root: Path = DEFAULT_OUTPUT,
    *,
    maximum_cycles: int | None = None,
    unknown_policy: str = "monitor",
) -> dict:
    if representation not in {"sphere", "ellipsoid"}:
        raise ValueError("representation must be sphere or ellipsoid")
    if index_mode not in {"full_scan", "mvt_scalar", "mvt_simd"}:
        raise ValueError("unknown index mode")
    if unknown_policy not in {"monitor", "strict"}:
        raise ValueError("unknown_policy must be monitor or strict")
    scene = formal_protocol_scene()
    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(scene.q0, dtype=float))
    robot = build_robot_certificate(model)
    robot_radii = np.asarray([sphere.radius for sphere in robot], dtype=float)
    target = np.asarray(scene.waypoints[-1], dtype=float)
    occupancy = NativeIncrementalOccupancyMap(MAP_VOXEL_SIZE)
    proxy_manager = IncrementalMatchedProxyManager(
        filter_size=CENTERVOX_SIZE,
        cluster_size=PROXY_CLUSTER_SIZE,
        maximum_aabb_overshoot=MAXIMUM_AABB_OVERSHOOT,
    )
    camera = UR5MountedDepthCamera(
        model,
        camera_names=(
            "ur5_depth_wrist",
            "ur5_depth_wrist_right",
            "ur5_depth_shoulder",
        ),
        width=160,
        height=120,
        pixel_stride=CAMERA_PIXEL_STRIDE,
        optical_depth_error_bound=0.003,
        occluding_self_filter=True,
    )
    controller = None
    active_mvt = None
    last_camera_tick = -1
    formal_cycles = int(round(scene.duration / DT))
    cycle_limit = formal_cycles if maximum_cycles is None else int(maximum_cycles)
    if cycle_limit <= 0 or cycle_limit > formal_cycles:
        raise ValueError("maximum_cycles must be in the formal duration")
    suffix = "" if unknown_policy == "monitor" else "-strict_unknown"
    run_name = f"O-CV-{index_mode}-{representation}{suffix}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "scene.xml").write_bytes(build_xml(scene).encode("utf-8"))

    cycles: list[dict] = []
    frames: list[dict] = []
    q_history: list[np.ndarray] = []
    ee_history: list[np.ndarray] = []
    success_hold = 0
    first_success_cycle = None
    first_unknown_blocker = None
    maximum_penetration = 0.0
    run_started = time.perf_counter()
    try:
        for cycle in range(cycle_limit):
            wall_started = time.perf_counter()
            q_before = data.qpos[: len(JOINT_NAMES)].copy()
            robot_before = certificate_world_positions(data, robot)
            if cycle > 0:
                # This volume was accepted by the previous causal sweep.
                occupancy.mark_current_robot_free(robot_before, robot_radii)
            camera_tick = int(np.floor(cycle * DT * CAMERA_HZ + 1.0e-12))
            camera_updated = camera_tick != last_camera_tick
            frame_update_ms = 0.0
            if camera_updated:
                frame_started = time.perf_counter()
                capture_started = time.perf_counter()
                observations = camera.capture(data)
                capture_ms = (time.perf_counter() - capture_started) * 1000.0
                map_started = time.perf_counter()
                map_stats = occupancy.integrate(observations)
                map_update_ms = (time.perf_counter() - map_started) * 1000.0
                if cycle == 0:
                    # Integrate observed occupied endpoints first.  mark-free
                    # never overwrites OCCUPIED, so the calibrated start cell
                    # cannot erase a visible obstacle even if misconfigured.
                    occupancy.mark_current_robot_free(
                        robot_before,
                        robot_radii,
                        padding=INITIAL_CALIBRATED_FREE_PADDING,
                    )
                point_batches = []
                radius_batches = []
                raw_points_by_camera = {
                    observation.camera_name: int(len(observation.points))
                    for observation in observations
                }
                environment_points_by_camera = {
                    observation.camera_name: int(
                        np.count_nonzero(environment_endpoint_mask(observation))
                    )
                    for observation in observations
                }
                self_returns_by_camera = {
                    observation.camera_name: int(
                        len(observation.points)
                        - np.count_nonzero(environment_endpoint_mask(observation))
                    )
                    for observation in observations
                }
                workspace_points_by_camera = {}
                for observation in observations:
                    points, radii = _crop_proxy_observation(observation)
                    workspace_points_by_camera[observation.camera_name] = int(
                        len(points)
                    )
                    if len(points):
                        point_batches.append(points)
                        radius_batches.append(radii)
                if not point_batches:
                    raise RuntimeError("causal camera frame contained no workspace points")
                update_stats = proxy_manager.update(
                    np.vstack(point_batches), np.concatenate(radius_batches)
                )
                coverage = proxy_manager.coverage_audit()
                if not coverage.all_raw_sample_balls_certified:
                    raise AssertionError("CenterVox certificate coverage failed")
                proxies = proxy_manager.snapshot
                next_mvt = None
                mvt_missing = 0
                mvt_started = time.perf_counter()
                if index_mode != "full_scan":
                    next_mvt, mvt_missing = _build_mvt_and_audit(
                        proxies,
                        representation,
                        robot_before,
                        robot_radii,
                        simd=index_mode == "mvt_simd",
                    )
                mvt_update_ms = (time.perf_counter() - mvt_started) * 1000.0
                if controller is None:
                    controller = _new_controller(
                        representation,
                        model,
                        data,
                        scene,
                        robot,
                        proxies,
                        next_mvt,
                    )
                else:
                    kwargs = dict(
                        obstacle_offsets=proxies.proxy_offset_radii,
                        proxy_ids=proxies.proxy_ids,
                        obstacle_index=next_mvt,
                    )
                    if representation == "sphere":
                        kwargs["obstacle_radii"] = proxies.sphere_radii
                    else:
                        kwargs["obstacle_shapes"] = proxies.base_ellipsoid_shapes
                        kwargs["obstacle_uncertainty_shapes"] = (
                            proxies.proxy_uncertainty_shapes
                        )
                    controller.update_obstacles(proxies.centers, **kwargs)
                old_mvt = active_mvt
                active_mvt = next_mvt
                if old_mvt is not None:
                    old_mvt.close()
                last_camera_tick = camera_tick
                frame_update_ms = (time.perf_counter() - frame_started) * 1000.0
                frames.append(
                    {
                        "frame": len(frames),
                        "cycle": cycle,
                        "time_s": cycle * DT,
                        "raw_depth_points": int(
                            sum(len(item.points) for item in observations)
                        ),
                        "workspace_depth_points": int(
                            sum(len(item) for item in point_batches)
                        ),
                        "raw_points_by_camera": json.dumps(
                            raw_points_by_camera, sort_keys=True
                        ),
                        "environment_points_by_camera": json.dumps(
                            environment_points_by_camera, sort_keys=True
                        ),
                        "self_returns_by_camera": json.dumps(
                            self_returns_by_camera, sort_keys=True
                        ),
                        "workspace_points_by_camera": json.dumps(
                            workspace_points_by_camera, sort_keys=True
                        ),
                        "free_voxels": map_stats.free_voxels,
                        "occupied_voxels": map_stats.occupied_voxels,
                        "center_voxels": update_stats.center_voxels,
                        "proxy_count": update_stats.proxy_count,
                        "proxy_generation": update_stats.generation,
                        "proxy_snapshot_sha256": update_stats.snapshot_sha256,
                        "proxy_update_ms": update_stats.update_ms,
                        "capture_ms": capture_ms,
                        "map_update_ms": map_update_ms,
                        "mvt_update_ms": mvt_update_ms,
                        "frame_pipeline_ms": frame_update_ms,
                        "coverage_ok": coverage.all_raw_sample_balls_certified,
                        "minimum_centervox_cover_slack": coverage.minimum_centervox_cover_slack,
                        "minimum_proxy_offset_slack": coverage.minimum_proxy_offset_slack,
                        "maximum_base_ellipsoid_value": coverage.maximum_base_ellipsoid_value,
                        "mvt_missing_candidates": mvt_missing,
                        "mvt_levels": 0 if active_mvt is None else active_mvt.stats.level_count,
                        "mvt_index_references": 0 if active_mvt is None else active_mvt.stats.index_references,
                        "mvt_cell_lookups_per_robot_query": 0 if active_mvt is None else active_mvt.stats.cell_lookups_per_query,
                    }
                )

            if controller is None:
                raise RuntimeError("controller was not initialized from a causal frame")
            qdot_proposed, metrics = controller.solve(target)
            sweep_scale = 1.0
            sweep_trials = 0
            while True:
                sweep_trials += 1
                proposed_q = q_before + sweep_scale * qdot_proposed * DT
                set_configuration(model, data, proposed_q)
                robot_after = certificate_world_positions(data, robot)
                set_configuration(model, data, q_before)
                sweep = occupancy.certify_swept_spheres(
                    robot_before,
                    robot_after,
                    robot_radii,
                    # Unknown is checked against the physical swept robot
                    # certificate.  The 6 mm obstacle safety margin is already
                    # a hard QP clearance, not a second physical body shell.
                    margin=0.0,
                )
                sweep_blocked = sweep.occupied_voxels > 0 or (
                    unknown_policy == "strict" and sweep.unknown_voxels > 0
                )
                if not sweep_blocked or sweep_scale <= 1.0 / 128.0:
                    break
                # Causal safety scaling is allowed by Eq. (41): it queries only
                # the current map snapshot.  It never reads MuJoCo contact or
                # future geometry and it does not change the target/direction.
                sweep_scale *= 0.5
            guard_safe = not sweep_blocked
            if (
                unknown_policy == "strict"
                and not guard_safe
                and first_unknown_blocker is None
            ):
                # One-time causal diagnostic only: identify which physical
                # robot certificate sphere touches the first still-unknown
                # voxel.  These read-only map queries do not alter qdot or the
                # map and are excluded from controller_ms.
                for sphere_index, sphere_meta in enumerate(robot):
                    sphere_sweep = occupancy.certify_swept_spheres(
                        robot_before[sphere_index : sphere_index + 1],
                        robot_after[sphere_index : sphere_index + 1],
                        robot_radii[sphere_index : sphere_index + 1],
                        margin=0.0,
                    )
                    if not sphere_sweep.safe:
                        first_unknown_blocker = {
                            "cycle": cycle,
                            "sphere_index": sphere_index,
                            "body_name": sphere_meta.body_name,
                            "source_geom_name": model.geom(
                                sphere_meta.source_geom_id
                            ).name,
                            "radius_m": float(robot_radii[sphere_index]),
                            "start_center_m": robot_before[sphere_index].tolist(),
                            "proposed_center_m": robot_after[
                                sphere_index
                            ].tolist(),
                            "occupied_voxels": sphere_sweep.occupied_voxels,
                            "unknown_voxels": sphere_sweep.unknown_voxels,
                            "first_nonfree_voxel": _first_nonfree_swept_voxel(
                                occupancy,
                                robot_before[sphere_index],
                                robot_after[sphere_index],
                                robot_radii[sphere_index],
                            ),
                        }
                        break
            qdot_executed = (
                sweep_scale * qdot_proposed
                if guard_safe
                else np.zeros_like(qdot_proposed)
            )
            q_after = q_before + qdot_executed * DT
            set_configuration(model, data, q_after)
            ee = attachment_position(model, data)
            error = float(np.linalg.norm(target - ee))
            penetrating = [
                index
                for index in range(data.ncon)
                if float(data.contact[index].dist) < -1.0e-8
            ]
            minimum_contact_distance = (
                min(float(data.contact[index].dist) for index in range(data.ncon))
                if data.ncon
                else None
            )
            if minimum_contact_distance is not None:
                maximum_penetration = min(
                    maximum_penetration, minimum_contact_distance
                )
            if error <= SUCCESS_TOLERANCE and not penetrating:
                success_hold += 1
                if first_success_cycle is None:
                    first_success_cycle = cycle
            else:
                success_hold = 0
                if error > SUCCESS_TOLERANCE:
                    first_success_cycle = None
            cycles.append(
                {
                    "cycle": cycle,
                    "time_s": cycle * DT,
                    "representation": representation,
                    "index_mode": index_mode,
                    "camera_updated": camera_updated,
                    "status": metrics.status,
                    "ee_x": float(ee[0]),
                    "ee_y": float(ee[1]),
                    "ee_z": float(ee[2]),
                    "error_m": error,
                    "proposed_qdot_norm": float(np.linalg.norm(qdot_proposed)),
                    "executed_qdot_norm": float(np.linalg.norm(qdot_executed)),
                    "causal_sweep_scale": sweep_scale if guard_safe else 0.0,
                    "causal_sweep_trials": sweep_trials,
                    "occupancy_guard_safe": guard_safe,
                    "unknown_policy": unknown_policy,
                    "sweep_occupied_voxels": sweep.occupied_voxels,
                    "sweep_unknown_voxels": sweep.unknown_voxels,
                    "sweep_checked_voxels": sweep.checked_voxels,
                    "exact_contact_count": int(data.ncon),
                    "exact_penetrating_contact_count": len(penetrating),
                    "minimum_contact_distance_m": minimum_contact_distance,
                    "frame_pipeline_ms": frame_update_ms,
                    "wall_cycle_ms": (time.perf_counter() - wall_started) * 1000.0,
                    **metrics.as_dict(),
                }
            )
            q_history.append(q_after.copy())
            ee_history.append(ee.copy())
            if success_hold >= SUCCESS_HOLD_CYCLES:
                break
    finally:
        camera.close()
        occupancy.close()
        if active_mvt is not None:
            active_mvt.close()

    elapsed = time.perf_counter() - run_started
    errors = np.asarray([row["error_m"] for row in cycles])
    controller_ms = np.asarray([row["total_controller_ms"] for row in cycles])
    wall_ms = np.asarray([row["wall_cycle_ms"] for row in cycles])
    q_array = np.asarray(q_history)
    ee_array = np.asarray(ee_history)
    final_proxies = proxy_manager.snapshot
    np.save(run_dir / "q_history.npy", q_array)
    np.save(run_dir / "ee_history.npy", ee_array)
    np.savez_compressed(
        run_dir / "final_causal_proxies.npz",
        proxy_ids=final_proxies.proxy_ids,
        centers=final_proxies.centers,
        sphere_radii=final_proxies.sphere_radii,
        base_sphere_radii=final_proxies.base_sphere_radii,
        ellipsoid_shapes=final_proxies.base_ellipsoid_shapes,
        ellipsoid_outer_shapes=final_proxies.ellipsoid_outer_shapes,
        proxy_uncertainty_shapes=final_proxies.proxy_uncertainty_shapes,
        uncertainty_offsets=final_proxies.proxy_offset_radii,
        filtered_points=final_proxies.filtered_points,
        filtered_cluster_indices=final_proxies.filtered_cluster_indices,
        filtered_uncertainty_shapes=final_proxies.filtered_uncertainty_shapes,
    )
    _write_csv(run_dir / "cycles.csv", cycles)
    _write_csv(run_dir / "camera_frames.csv", frames)
    summary = {
        "experiment": "protocol_v3_causal_online_liuqp",
        "run_name": run_name,
        "representation": representation,
        "index_mode": index_mode,
        "success": success_hold >= SUCCESS_HOLD_CYCLES,
        "cycles": len(cycles),
        "simulated_duration_s": len(cycles) * DT,
        "wall_duration_s": elapsed,
        "first_success_time_s": (
            None if first_success_cycle is None else first_success_cycle * DT
        ),
        "final_error_m": float(errors[-1]),
        "minimum_error_m": float(np.min(errors)),
        "maximum_ee_x_m": float(np.max(ee_array[:, 0])),
        "occupancy_guard_rejections": int(
            sum(not row["occupancy_guard_safe"] for row in cycles)
        ),
        "unknown_monitor_cycles": int(
            sum(row["sweep_unknown_voxels"] > 0 for row in cycles)
        ),
        "causal_sweep_scaled_cycles": int(
            sum(0.0 < row["causal_sweep_scale"] < 1.0 for row in cycles)
        ),
        "causal_sweep_safe_stop_cycles": int(
            sum(row["causal_sweep_scale"] == 0.0 for row in cycles)
        ),
        "maximum_sweep_unknown_voxels": int(
            max(row["sweep_unknown_voxels"] for row in cycles)
        ),
        "maximum_sweep_occupied_voxels": int(
            max(row["sweep_occupied_voxels"] for row in cycles)
        ),
        "first_unknown_blocker": first_unknown_blocker,
        "qp_solved_fraction": float(
            np.mean([row["status"].startswith("solved") for row in cycles])
        ),
        "exact_penetrating_cycles": int(
            sum(row["exact_penetrating_contact_count"] > 0 for row in cycles)
        ),
        "maximum_penetration_m": float(maximum_penetration),
        "controller_ms_p50": float(np.percentile(controller_ms, 50)),
        "controller_ms_p95": float(np.percentile(controller_ms, 95)),
        "controller_ms_p99": float(np.percentile(controller_ms, 99)),
        "wall_cycle_ms_p50": float(np.percentile(wall_ms, 50)),
        "wall_cycle_ms_p95": float(np.percentile(wall_ms, 95)),
        "wall_cycle_ms_p99": float(np.percentile(wall_ms, 99)),
        "camera_frames": len(frames),
        "initial_depth_points": frames[0]["raw_depth_points"],
        "final_center_voxels": frames[-1]["center_voxels"],
        "final_proxy_count": frames[-1]["proxy_count"],
        "all_centervox_coverage_checks_passed": all(
            row["coverage_ok"] for row in frames
        ),
        "all_mvt_oracle_checks_passed": all(
            row["mvt_missing_candidates"] == 0 for row in frames
        ),
        "local_moving_cameras": [
            "ur5_depth_wrist",
            "ur5_depth_wrist_right",
            "ur5_depth_shoulder",
        ],
        "camera_count": 3,
        "incremental_map": True,
        "online_protocol_revision": "v3.2-local-multiview",
        "map_voxel_size_m": MAP_VOXEL_SIZE,
        "calibrated_staging_aabb": None,
        "unknown_map_state_preserved": True,
        "unknown_motion_policy": unknown_policy,
        "initial_calibrated_free_padding_m": INITIAL_CALIBRATED_FREE_PADDING,
        "unknown_sweep_extra_margin_m": 0.0,
        "target_count": 1,
        "path_planner": None,
        "intermediate_targets": None,
        "random_dither": False,
        "truth_collision_backtracking": False,
        "robot_certificate_spheres": len(robot),
        "source_hashes": _source_hashes(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representation", choices=("sphere", "ellipsoid", "both"), default="both"
    )
    parser.add_argument(
        "--index-mode",
        choices=("full_scan", "mvt_scalar", "mvt_simd"),
        default="mvt_simd",
    )
    parser.add_argument("--maximum-cycles", type=int)
    parser.add_argument(
        "--unknown-policy", choices=("monitor", "strict"), default="monitor"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    representations = (
        ("sphere", "ellipsoid")
        if args.representation == "both"
        else (args.representation,)
    )
    summaries = [
        run_one(
            representation,
            args.index_mode,
            args.output,
            maximum_cycles=args.maximum_cycles,
            unknown_policy=args.unknown_policy,
        )
        for representation in representations
    ]
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
