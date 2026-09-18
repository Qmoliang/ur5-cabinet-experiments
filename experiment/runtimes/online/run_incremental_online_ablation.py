"""Formal causal single-wrist-camera sphere/ellipsoid LiuQP comparison."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import multiprocessing
import queue
from pathlib import Path
import threading
import time

import mujoco
import numpy as np

from depth_camera_perception import UR5MountedDepthCamera
from configuration_corridor_planner import plan_configuration_corridor
from ellipsoid_qp_controller import EllipsoidLiuQPController
from incremental_drawer_scene import incremental_drawer_v12_certified_candidate_scene
from incremental_proxy_manager import IncrementalMatchedProxyManager
from iris_taskspace_planner import plan_iris_taskspace_corridor
from liuqp_controller import LiuQPController
from model import (
    DT,
    attachment_position,
    build_model,
    build_robot_certificate,
    build_xml,
    certificate_world_positions,
    certificate_world_state,
    set_configuration,
)
from native_mvt import NativeMVT
from native_occupancy import NativeIncrementalOccupancyMap, NativeOccupancySnapshot
from run_simulation import (
    minimum_mujoco_self_contact,
    minimum_mujoco_world_contact,
)
from ellipsoid_model import (
    build_robot_ellipsoid_certificate,
    ellipsoid_world_state,
)
from pointcloud_proxy import loewner_union_outer_shape, minkowski_outer_shape


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_incremental_v2"

MAP_VOXEL = 0.015
CENTER_VOXEL = 0.006
CLUSTER_SIZE = 0.300
NORMAL_CONNECTION = 0.027
MAXIMUM_AABB_OVERSHOOT = 0.025
CALIBRATED_DEPTH_BOUND = 0.003
SAFETY_MARGIN = 0.001
COMMON_NEAR_WEIGHT = 22.0
COMMON_NEAR_DISTANCE = 0.040
MVT_QUERY_PADDING = COMMON_NEAR_DISTANCE + SAFETY_MARGIN
SENSOR_PERIOD_STEPS = 10
SUCCESS_TOLERANCE = 0.018
SUCCESS_HOLD_STEPS = 25
UNKNOWN_EXPLORATION_HORIZON = 0.5 * MAP_VOXEL
PATH_LOOKAHEAD = 0.040
REPLAN_STALL_STEPS = 1000
REPLAN_BACKOFF_STEPS = 1500
BACKTRACK_SCALES = (
    1.0,
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
    0.0078125,
    0.00390625,
    0.001953125,
    0.0009765625,
    0.0,
)
ROBOT_SWEEP_RADIUS_BOUND = 1.5
ELLIPSOID_SWEEP_SAMPLES = 5


@dataclass(frozen=True)
class PerceptionSnapshot:
    generation: int
    source_step: int
    source_sim_time: float
    published_wall_time: float
    proxies: object
    obstacle_index: NativeMVT
    occupancy: NativeOccupancySnapshot
    map_keys: np.ndarray
    map_states: np.ndarray
    observed_points: np.ndarray
    robot_centers: np.ndarray
    map_stats: dict
    proxy_stats: dict
    capture_ms: float
    total_update_ms: float


def _perception_process(certificate: str, requests, results) -> None:
    """Process-isolated renderer/map/proxy loop; publishes plain arrays only."""

    camera = None
    try:
        scene = incremental_drawer_v12_certified_candidate_scene()
        model = build_model(scene)
        data = mujoco.MjData(model)
        robot = build_robot_certificate(model)
        occupancy = NativeIncrementalOccupancyMap(MAP_VOXEL)
        manager = IncrementalMatchedProxyManager(
            filter_size=CENTER_VOXEL,
            cluster_size=CLUSTER_SIZE,
            maximum_aabb_overshoot=MAXIMUM_AABB_OVERSHOOT,
            normal_connection_distance=NORMAL_CONNECTION,
        )
        camera = UR5MountedDepthCamera(
            model,
            camera_names=("ur5_depth_wrist", "ur5_depth_shoulder"),
            width=320,
            height=240,
            pixel_stride=2,
            occluding_self_filter=True,
            optical_depth_error_bound=CALIBRATED_DEPTH_BOUND,
        )
        initialized = False
        while True:
            item = requests.get()
            if item is None:
                break
            source_step, source_sim_time, q = item
            update_started = time.perf_counter()
            set_configuration(model, data, q)
            centers, _, radii = certificate_world_state(model, data, robot)
            if not initialized:
                occupancy.mark_current_robot_free(centers, radii)
                initialized = True
            capture_started = time.perf_counter()
            observations = camera.capture(data)
            capture_ms = (time.perf_counter() - capture_started) * 1000.0
            map_stats = occupancy.integrate(observations)
            occupancy.mark_current_robot_free(centers, radii)
            points = np.vstack(
                [observation.points for observation in observations]
            )
            point_radii = np.concatenate(
                [observation.sample_radii for observation in observations]
            )
            point_uncertainty_shapes = np.concatenate(
                [
                    observation.sample_uncertainty_shapes
                    for observation in observations
                ]
            )
            proxy_stats = manager.update(
                points, point_radii, point_uncertainty_shapes
            )
            keys, states = occupancy.snapshot_arrays()
            results.put(
                {
                    "certificate": certificate,
                    "source_step": source_step,
                    "source_sim_time": source_sim_time,
                    "proxies": manager.snapshot,
                    "map_keys": keys,
                    "map_states": states,
                    "observed_points": points,
                    "robot_centers": centers,
                    "map_stats": asdict(map_stats),
                    "proxy_stats": asdict(proxy_stats),
                    "capture_ms": capture_ms,
                    "total_update_ms": (
                        time.perf_counter() - update_started
                    )
                    * 1000.0,
                }
            )
    except BaseException as error:
        results.put({"error": repr(error)})
    finally:
        if camera is not None:
            camera.close()
        results.put(None)


class OnlinePerceptionWorker:
    """Own the renderer/map/proxy builders and atomically publish snapshots."""

    def __init__(self, certificate: str):
        self.certificate = certificate
        self.scene = incremental_drawer_v12_certified_candidate_scene()
        context = multiprocessing.get_context("spawn")
        self._queue = context.Queue(maxsize=1)
        self._results = context.Queue(maxsize=2)
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._latest: PerceptionSnapshot | None = None
        self._error: BaseException | None = None
        self._process = context.Process(
            target=_perception_process,
            args=(certificate, self._queue, self._results),
            daemon=True,
        )
        self._receiver = threading.Thread(
            target=self._receive, daemon=True
        )
        worker_model = build_model(self.scene)
        self._robot_rmax = max(
            item.radius for item in build_robot_certificate(worker_model)
        )

    def start(self, q: np.ndarray) -> PerceptionSnapshot:
        self._process.start()
        self._receiver.start()
        self.submit(0, 0.0, q)
        if not self._event.wait(timeout=30.0):
            raise TimeoutError("initial causal perception snapshot timed out")
        if self._error is not None:
            raise RuntimeError("perception worker failed") from self._error
        assert self._latest is not None
        return self._latest

    def submit(self, step: int, sim_time: float, q: np.ndarray) -> None:
        item = (int(step), float(sim_time), np.asarray(q, dtype=float).copy())
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                # The feeder thread may still own the single slot. Dropping a
                # stale sensor request is preferable to blocking control.
                pass

    def latest(self) -> PerceptionSnapshot:
        if self._error is not None:
            raise RuntimeError("perception worker failed") from self._error
        with self._lock:
            if self._latest is None:
                raise RuntimeError("no perception snapshot")
            return self._latest

    def close(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(None)
        self._process.join(timeout=10.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
        self._receiver.join(timeout=5.0)

    def _receive(self) -> None:
        try:
            while True:
                payload = self._results.get()
                if payload is None:
                    break
                if "error" in payload:
                    raise RuntimeError(payload["error"])
                proxies = payload["proxies"]
                offsets = np.asarray(
                    proxies.proxy_offset_radii, dtype=float
                )
                if self.certificate == "sphere":
                    obstacle_index = NativeMVT.from_spheres(
                        proxies.centers,
                        proxies.sphere_radii + offsets,
                        voxel_size=self._robot_rmax + 0.003,
                        query_padding=MVT_QUERY_PADDING,
                        simd=True,
                    )
                else:
                    obstacle_index = NativeMVT.from_ellipsoids(
                        proxies.centers,
                        proxies.ellipsoid_outer_shapes,
                        offsets,
                        voxel_size=self._robot_rmax + 0.003,
                        query_padding=MVT_QUERY_PADDING,
                        simd=True,
                    )
                native_occupancy = NativeOccupancySnapshot(
                    payload["map_keys"],
                    payload["map_states"],
                    MAP_VOXEL,
                )
                snapshot = PerceptionSnapshot(
                    generation=payload["proxy_stats"]["generation"],
                    source_step=payload["source_step"],
                    source_sim_time=payload["source_sim_time"],
                    published_wall_time=time.perf_counter(),
                    proxies=proxies,
                    obstacle_index=obstacle_index,
                    occupancy=native_occupancy,
                    map_keys=payload["map_keys"],
                    map_states=payload["map_states"],
                    observed_points=payload["observed_points"],
                    robot_centers=payload["robot_centers"],
                    map_stats=payload["map_stats"],
                    proxy_stats=payload["proxy_stats"],
                    capture_ms=payload["capture_ms"],
                    total_update_ms=payload["total_update_ms"],
                )
                with self._lock:
                    self._latest = snapshot
                self._event.set()
        except BaseException as error:
            self._error = error
            self._event.set()

    def _run(self) -> None:
        camera = None
        try:
            model = build_model(self.scene)
            data = mujoco.MjData(model)
            robot = build_robot_certificate(model)
            robot_rmax = max(item.radius for item in robot)
            occupancy = NativeIncrementalOccupancyMap(MAP_VOXEL)
            manager = IncrementalMatchedProxyManager(
                filter_size=CENTER_VOXEL,
                cluster_size=CLUSTER_SIZE,
                maximum_aabb_overshoot=MAXIMUM_AABB_OVERSHOOT,
                normal_connection_distance=NORMAL_CONNECTION,
            )
            camera = UR5MountedDepthCamera(
                model,
                camera_names=(
                    "ur5_depth_wrist",
                    "ur5_depth_shoulder",
                ),
                width=320,
                height=240,
                pixel_stride=2,
                occluding_self_filter=True,
                optical_depth_error_bound=CALIBRATED_DEPTH_BOUND,
            )
            initialized = False
            while True:
                item = self._queue.get()
                if item is None:
                    break
                source_step, source_sim_time, q = item
                update_started = time.perf_counter()
                set_configuration(model, data, q)
                centers, _, radii = certificate_world_state(
                    model, data, robot
                )
                if not initialized:
                    occupancy.mark_current_robot_free(centers, radii)
                    initialized = True
                capture_started = time.perf_counter()
                observations = camera.capture(data)
                capture_ms = (time.perf_counter() - capture_started) * 1000.0
                map_stats = occupancy.integrate(observations)
                occupancy.mark_current_robot_free(centers, radii)
                points = np.vstack(
                    [observation.points for observation in observations]
                )
                point_radii = np.concatenate(
                    [observation.sample_radii for observation in observations]
                )
                point_uncertainty_shapes = np.concatenate(
                    [
                        observation.sample_uncertainty_shapes
                        for observation in observations
                    ]
                )
                proxy_stats = manager.update(
                    points, point_radii, point_uncertainty_shapes
                )
                proxies = manager.snapshot
                offsets = np.asarray(
                    proxies.proxy_offset_radii, dtype=float
                )
                if self.certificate == "sphere":
                    obstacle_index = NativeMVT.from_spheres(
                        proxies.centers,
                        proxies.sphere_radii + offsets,
                        voxel_size=robot_rmax + 0.003,
                        query_padding=MVT_QUERY_PADDING,
                        simd=True,
                    )
                else:
                    obstacle_index = NativeMVT.from_ellipsoids(
                        proxies.centers,
                        proxies.ellipsoid_outer_shapes,
                        offsets,
                        voxel_size=robot_rmax + 0.003,
                        query_padding=MVT_QUERY_PADDING,
                        simd=True,
                    )
                keys, states = occupancy.snapshot_arrays()
                native_occupancy = NativeOccupancySnapshot(
                    keys, states, MAP_VOXEL
                )
                snapshot = PerceptionSnapshot(
                    generation=proxy_stats.generation,
                    source_step=source_step,
                    source_sim_time=source_sim_time,
                    published_wall_time=time.perf_counter(),
                    proxies=proxies,
                    obstacle_index=obstacle_index,
                    occupancy=native_occupancy,
                    map_keys=keys,
                    map_states=states,
                    observed_points=points,
                    robot_centers=centers.copy(),
                    map_stats=asdict(map_stats),
                    proxy_stats=asdict(proxy_stats),
                    capture_ms=capture_ms,
                    total_update_ms=(time.perf_counter() - update_started)
                    * 1000.0,
                )
                with self._lock:
                    self._latest = snapshot
                self._event.set()
        except BaseException as error:
            self._error = error
            self._event.set()
        finally:
            if camera is not None:
                camera.close()


def _make_controller(certificate, model, data, scene, robot, snapshot):
    proxies = snapshot.proxies
    offsets = np.asarray(proxies.proxy_offset_radii, dtype=float)
    if certificate == "sphere":
        controller = LiuQPController(
            model,
            data,
            scene,
            robot,
            proxies.centers,
            proxies.sphere_radii + offsets,
            safety_margin=SAFETY_MARGIN,
            obstacle_index=snapshot.obstacle_index,
        )
    else:
        controller = EllipsoidLiuQPController(
            model,
            data,
            scene,
            build_robot_ellipsoid_certificate(model),
            proxies.centers,
            proxies.ellipsoid_shapes,
            obstacle_offsets=offsets,
            obstacle_uncertainty_shapes=proxies.proxy_uncertainty_shapes,
            obstacle_pruning_shapes=proxies.ellipsoid_outer_shapes,
            safety_margin=SAFETY_MARGIN,
            obstacle_index=snapshot.obstacle_index,
        )
    controller.near_weight = COMMON_NEAR_WEIGHT
    controller.near_distance = COMMON_NEAR_DISTANCE
    return controller


def _update_controller(controller, certificate, snapshot) -> None:
    proxies = snapshot.proxies
    offsets = np.asarray(proxies.proxy_offset_radii, dtype=float)
    controller.obstacle_centers = np.asarray(proxies.centers, dtype=float)
    controller.obstacle_index = snapshot.obstacle_index
    if certificate == "sphere":
        controller.obstacle_radii = (
            np.asarray(proxies.sphere_radii, dtype=float) + offsets
        )
    else:
        controller.obstacle_shapes = np.asarray(
            proxies.ellipsoid_shapes, dtype=float
        )
        controller.obstacle_uncertainty_shapes = np.asarray(
            proxies.proxy_uncertainty_shapes, dtype=float
        )
        controller.obstacle_pruning_shapes = np.asarray(
            proxies.ellipsoid_outer_shapes, dtype=float
        )
        (
            controller.obstacle_eigenvalues,
            controller.obstacle_rotations,
        ) = np.linalg.eigh(controller.obstacle_shapes)
        controller.obstacle_offsets = offsets.copy()
        controller._closest_multiplier_cache.clear()


def _path_cumulative_lengths(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return np.zeros(len(points), dtype=float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def _point_at_path_distance(
    points: np.ndarray, cumulative: np.ndarray, distance: float
) -> tuple[np.ndarray, int]:
    distance = float(np.clip(distance, cumulative[0], cumulative[-1]))
    upper = int(np.searchsorted(cumulative, distance, side="right"))
    upper = min(max(upper, 1), len(points) - 1)
    lower = upper - 1
    span = float(cumulative[upper] - cumulative[lower])
    fraction = 0.0 if span <= 1.0e-12 else (distance - cumulative[lower]) / span
    return points[lower] + fraction * (points[upper] - points[lower]), upper


def _pure_pursuit_target(
    points: np.ndarray,
    cumulative: np.ndarray,
    position: np.ndarray,
    previous_progress: float,
    lookahead: float,
) -> tuple[np.ndarray, int, float]:
    """Project onto the autonomous polyline and advance monotonically."""

    if len(points) == 1:
        return points[0].copy(), 0, 0.0
    best_distance = np.inf
    projected_progress = previous_progress
    for index, (first, second) in enumerate(zip(points[:-1], points[1:])):
        if cumulative[index] > previous_progress + lookahead:
            break
        segment = second - first
        squared = float(segment @ segment)
        fraction = 0.0 if squared <= 1.0e-15 else float(
            np.clip((position - first) @ segment / squared, 0.0, 1.0)
        )
        candidate_progress = float(
            cumulative[index]
            + fraction * (cumulative[index + 1] - cumulative[index])
        )
        if candidate_progress + 1.0e-10 < previous_progress:
            continue
        candidate = first + fraction * segment
        distance = float(np.linalg.norm(position - candidate))
        if distance < best_distance:
            best_distance = distance
            projected_progress = candidate_progress
    progress = max(previous_progress, projected_progress)
    return (*_point_at_path_distance(points, cumulative, progress + lookahead), progress)


def _planner_geometry(certificate, model, data, robot, snapshot):
    proxies = snapshot.proxies
    proxy_offsets = np.asarray(proxies.proxy_offset_radii, dtype=float)
    attachment = attachment_position(model, data)
    attachment_body = int(model.site_bodyid[model.site("attachment_site").id])
    if certificate == "sphere":
        obstacle_radii = np.asarray(proxies.sphere_radii, dtype=float) + proxy_offsets
        obstacle_shapes = np.eye(3)[None, :, :] * obstacle_radii[:, None, None] ** 2
        obstacle_support_offsets = np.zeros(len(obstacle_radii))
        obstacle_uncertainty_shapes = np.zeros_like(obstacle_shapes)
        obstacle_pruning_shapes = obstacle_shapes.copy()
        positions, _, radii = certificate_world_state(model, data, robot)
        mask = np.asarray([item.body_id == attachment_body for item in robot], dtype=bool)
        tool_shapes = np.eye(3)[None, :, :] * radii[mask, None, None] ** 2
        tool_offsets = positions[mask] - attachment
    else:
        obstacle_shapes = np.asarray(proxies.ellipsoid_shapes, dtype=float)
        obstacle_support_offsets = proxy_offsets
        obstacle_uncertainty_shapes = np.asarray(
            proxies.proxy_uncertainty_shapes, dtype=float
        )
        obstacle_pruning_shapes = np.asarray(
            proxies.ellipsoid_outer_shapes, dtype=float
        )
        certificate = build_robot_ellipsoid_certificate(model)
        positions, _, _, shapes = ellipsoid_world_state(model, data, certificate)
        mask = np.asarray(
            [item.body_id == attachment_body for item in certificate], dtype=bool
        )
        tool_shapes = shapes[mask]
        tool_offsets = positions[mask] - attachment
    return (
        np.asarray(proxies.centers, dtype=float).copy(),
        obstacle_shapes.copy(),
        obstacle_support_offsets.copy(),
        tool_shapes.copy(),
        tool_offsets.copy(),
        obstacle_uncertainty_shapes.copy(),
        obstacle_pruning_shapes.copy(),
    )


def _solve_planner_problem(start, target, geometry):
    (
        centers,
        shapes,
        offsets,
        tool_shapes,
        tool_offsets,
        uncertainty_shapes,
        pruning_shapes,
    ) = geometry
    started = time.perf_counter()
    try:
        plan = plan_iris_taskspace_corridor(
            start,
            target,
            centers,
            shapes,
            offsets,
            tool_radius=0.0,
            safety_margin=SAFETY_MARGIN,
            resolution=0.015,
            maximum_control_step=0.025,
            tool_shapes=tool_shapes,
            tool_offsets=tool_offsets,
            obstacle_uncertainty_shapes=uncertainty_shapes,
            obstacle_pruning_shapes=pruning_shapes,
        )
        error = None
    except RuntimeError as caught:
        plan = None
        error = str(caught)
    return plan, error, (time.perf_counter() - started) * 1000.0


def _snapshot_sha256(snapshot: PerceptionSnapshot) -> str:
    """Hash every causal planning input, not only the proxy parameters."""

    digest = hashlib.sha256()
    arrays = (
        snapshot.map_keys,
        snapshot.map_states,
        snapshot.observed_points,
        snapshot.proxies.centers,
        snapshot.proxies.sphere_radii,
        snapshot.proxies.ellipsoid_shapes,
        snapshot.proxies.proxy_offset_radii,
        snapshot.proxies.proxy_uncertainty_shapes,
        snapshot.proxies.ellipsoid_outer_shapes,
    )
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _snapshot_arrays(snapshot: PerceptionSnapshot) -> dict[str, np.ndarray]:
    """Persist both the effective proxies and their coverage decomposition."""

    arrays = {
        "map_keys": snapshot.map_keys,
        "map_states": snapshot.map_states,
        "observed_points": snapshot.observed_points,
        "centers": snapshot.proxies.centers,
        "sphere_radii": snapshot.proxies.sphere_radii,
        "ellipsoid_shapes": snapshot.proxies.ellipsoid_shapes,
        "proxy_offsets": snapshot.proxies.proxy_offset_radii,
    }
    optional = {
        "base_sphere_radii": snapshot.proxies.base_sphere_radii,
        "base_ellipsoid_shapes": snapshot.proxies.base_ellipsoid_shapes,
        "proxy_uncertainty_shapes": snapshot.proxies.proxy_uncertainty_shapes,
        "ellipsoid_outer_shapes": snapshot.proxies.ellipsoid_outer_shapes,
        "filtered_uncertainty_shapes": snapshot.proxies.filtered_uncertainty_shapes,
        "filtered_points": snapshot.proxies.filtered_points,
        "filtered_cluster_indices": snapshot.proxies.filtered_cluster_indices,
    }
    arrays.update({key: value for key, value in optional.items() if value is not None})
    return arrays


def _conservative_ellipsoid_sweep_outer_shapes(
    model,
    data,
    robot_ellipsoids,
    q_start: np.ndarray,
    q_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose the full joint-interpolated certificate sweep.

    Every sampled translated ellipsoid is recentered at its start center using
    a Minkowski outer shape.  A final isotropic term bounds motion between the
    nearest joint samples by ``R * ||dq||_1`` with R=1.5 m, larger than the
    UR5 base-to-tool reach including the retrieval tool.
    """

    alphas = np.linspace(0.0, 1.0, ELLIPSOID_SWEEP_SAMPLES)
    sampled_centers = []
    sampled_shapes = []
    for alpha in alphas:
        q = (1.0 - alpha) * q_start + alpha * q_end
        set_configuration(model, data, q)
        centers, _, _, shapes = ellipsoid_world_state(
            model, data, robot_ellipsoids
        )
        sampled_centers.append(centers)
        sampled_shapes.append(shapes)
    sampled_centers = np.asarray(sampled_centers)
    sampled_shapes = np.asarray(sampled_shapes)
    reference_centers = sampled_centers[0]
    half_interval_l1 = float(
        np.sum(np.abs(q_end[:6] - q_start[:6]))
        / (2.0 * (ELLIPSOID_SWEEP_SAMPLES - 1))
    )
    between_sample_radius = ROBOT_SWEEP_RADIUS_BOUND * half_interval_l1
    outer_shapes = []
    for item in range(sampled_centers.shape[1]):
        candidates = []
        for sample in range(len(alphas)):
            displacement = sampled_centers[sample, item] - reference_centers[item]
            shape = sampled_shapes[sample, item]
            if float(np.linalg.norm(displacement)) > 1.0e-15:
                shape = minkowski_outer_shape(
                    shape, np.outer(displacement, displacement)
                )
            candidates.append(shape)
        outer = loewner_union_outer_shape(np.asarray(candidates))
        if between_sample_radius > 0.0:
            outer = minkowski_outer_shape(
                outer, np.eye(3) * between_sample_radius**2
            )
        outer_shapes.append(outer)
    set_configuration(model, data, q_end)
    return reference_centers, np.asarray(outer_shapes)


def run(
    certificate: str,
    duration: float,
    realtime: bool = True,
    output_tag: str | None = None,
    use_configuration_corridor: bool = True,
) -> dict:
    scene = incremental_drawer_v12_certified_candidate_scene()
    result_root = RESULTS if output_tag is None else RESULTS / output_tag
    output = result_root / f"{scene.name}_{certificate}_causal_online"
    output.mkdir(parents=True, exist_ok=True)
    (output / "scene.xml").write_text(build_xml(scene), encoding="utf-8")
    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(scene.q0))
    robot = build_robot_certificate(model)
    robot_ellipsoids = (
        build_robot_ellipsoid_certificate(model)
        if certificate == "ellipsoid"
        else None
    )
    worker = OnlinePerceptionWorker(certificate)
    snapshot = worker.start(data.qpos[:6].copy())
    planning_snapshot_sha256 = _snapshot_sha256(snapshot)
    planning_proxy_snapshot_sha256 = snapshot.proxy_stats[
        "snapshot_sha256"
    ]
    planning_map_generation = snapshot.generation
    planning_map_source_step = snapshot.source_step
    np.savez_compressed(
        output / "planning_causal_map_and_proxies.npz",
        **_snapshot_arrays(snapshot),
    )
    target = np.asarray(scene.waypoints[-1], dtype=float)
    initial_geometry = _planner_geometry(
        certificate, model, data, robot, snapshot
    )
    plan, planner_error, planner_ms = _solve_planner_problem(
        attachment_position(model, data), target, initial_geometry
    )
    planner_tool_shapes = initial_geometry[3]
    guidance_targets = (
        np.asarray(plan.control_targets, dtype=float)
        if plan is not None
        else target[None, :]
    )
    guidance_cumulative = _path_cumulative_lengths(guidance_targets)
    guidance_progress = 0.0
    guidance_index = 0
    configuration_progress_index = 0
    configuration_corridor = None
    configuration_corridor_error = None
    if plan is not None and use_configuration_corridor:
        try:
            configuration_corridor = plan_configuration_corridor(
                guidance_targets,
                np.asarray(scene.q0),
                initial_geometry[0],
                initial_geometry[1],
                initial_geometry[2],
                safety_margin=SAFETY_MARGIN,
                robot_certificate=certificate,
                obstacle_uncertainty_shapes=initial_geometry[5],
                obstacle_pruning_shapes=initial_geometry[6],
            )
        except RuntimeError as caught:
            configuration_corridor_error = str(caught)
        if configuration_corridor is not None:
            guidance_targets = configuration_corridor.task_targets.copy()
            guidance_cumulative = _path_cumulative_lengths(guidance_targets)
            np.savez_compressed(
                output / "autonomous_configuration_corridor.npz",
                q_targets=configuration_corridor.q_targets,
                task_targets=configuration_corridor.task_targets,
                task_rotations=configuration_corridor.task_rotations,
            )
    if plan is not None:
        np.savez_compressed(
            output / "autonomous_iris_plan.npz",
            grid_path=plan.grid_path,
            smoothed_path=plan.smoothed_path,
            control_targets=plan.control_targets,
        )
    controller = _make_controller(
        certificate, model, data, scene, robot, snapshot
    )
    rows = []
    trajectories = []
    success_hold = 0
    exact_violations = 0
    unknown_rejections = 0
    unknown_exposure_steps = 0
    occupied_rejections = 0
    latest_generation = snapshot.generation
    replan_executor = ThreadPoolExecutor(max_workers=1)
    replan_future: Future | None = None
    replan_event_index: int | None = None
    replan_events: list[dict] = []
    plan_version = 0
    best_final_error = float(
        np.linalg.norm(attachment_position(model, data) - target)
    )
    last_progress_step = 0
    last_replan_step = -REPLAN_BACKOFF_STEPS
    wall_start = time.perf_counter()
    maximum_steps = int(round(duration / DT))
    try:
        for step in range(maximum_steps):
            deadline = wall_start + (step + 1) * DT
            cycle_started = time.perf_counter()
            newest = worker.latest()
            if newest.generation != latest_generation:
                snapshot = newest
                latest_generation = newest.generation
                _update_controller(controller, certificate, snapshot)
            if replan_future is not None and replan_future.done():
                new_plan, replan_error, replan_ms = replan_future.result()
                assert replan_event_index is not None
                event = replan_events[replan_event_index]
                event.update(
                    {
                        "completed_step": step,
                        "planner_ms": replan_ms,
                        "error": replan_error,
                        "success": new_plan is not None,
                    }
                )
                if new_plan is not None:
                    plan = new_plan
                    configuration_corridor = None
                    plan_version += 1
                    guidance_targets = np.asarray(
                        plan.control_targets, dtype=float
                    )
                    guidance_cumulative = _path_cumulative_lengths(
                        guidance_targets
                    )
                    guidance_progress = 0.0
                    guidance_index = 0
                    np.savez_compressed(
                        output / f"autonomous_iris_replan_{plan_version}.npz",
                        grid_path=plan.grid_path,
                        smoothed_path=plan.smoothed_path,
                        control_targets=plan.control_targets,
                    )
                    event.update(
                        {
                            "target_count": len(plan.control_targets),
                            "region_count": len(plan.regions),
                            "minimum_clearance_m": plan.minimum_path_clearance,
                        }
                    )
                replan_future = None
                replan_event_index = None
            ee_before = attachment_position(model, data)
            if configuration_corridor is not None:
                search_end = min(
                    len(configuration_corridor.q_targets),
                    configuration_progress_index + 25,
                )
                search = configuration_corridor.q_targets[
                    configuration_progress_index:search_end
                ]
                nearest = configuration_progress_index + int(
                    np.argmin(
                        np.linalg.norm(
                            search - data.qpos[:6][None, :], axis=1
                        )
                    )
                )
                configuration_progress_index = max(
                    configuration_progress_index, nearest
                )
                guidance_index = min(
                    configuration_progress_index + 12,
                    len(guidance_targets) - 1,
                )
                active_target = guidance_targets[guidance_index].copy()
                guidance_progress = float(guidance_cumulative[guidance_index])
            else:
                active_target, guidance_index, guidance_progress = (
                    _pure_pursuit_target(
                        guidance_targets,
                        guidance_cumulative,
                        ee_before,
                        guidance_progress,
                        PATH_LOOKAHEAD,
                    )
                )
            controller.task_region_A = None
            controller.task_region_b = None
            if plan is not None and configuration_corridor is None:
                containing_regions = [
                    region
                    for region in plan.regions
                    if region.covered_path_start <= guidance_index
                    and bool(np.all(region.A @ ee_before <= region.b + 1.0e-6))
                ]
                for region in reversed(containing_regions):
                    capped_distance = min(
                        guidance_progress + PATH_LOOKAHEAD,
                        guidance_cumulative[region.covered_path_end],
                    )
                    active_target, guidance_index = _point_at_path_distance(
                        guidance_targets, guidance_cumulative, capped_distance
                    )
                    if (
                        region.covered_path_start <= guidance_index
                        <= region.covered_path_end + 1
                    ):
                        controller.task_region_A = region.A
                        controller.task_region_b = region.b
                        break
            controller.posture_target = (
                None
                if configuration_corridor is None
                else configuration_corridor.q_targets[
                    min(guidance_index, len(configuration_corridor.q_targets) - 1)
                ]
            )
            if configuration_corridor is not None:
                controller.orientation_target = (
                    configuration_corridor.task_rotations[guidance_index]
                )
            q_before = data.qpos.copy()
            starts = certificate_world_positions(data, robot)
            radii = np.fromiter(
                (item.radius for item in robot), dtype=float
            )
            qdot, metrics = controller.solve(active_target)
            selected_scale = 0.0
            sweep = None
            accepted = False
            q_full = q_before.copy()
            mujoco.mj_integratePos(model, q_full, qdot, DT)
            set_configuration(model, data, q_full)
            full_ends = certificate_world_positions(data, robot)
            current_horizon = float(
                np.max(
                    np.linalg.norm(
                        starts - starts, axis=1
                    )
                )
            )
            full_horizon = float(
                np.max(
                    np.linalg.norm(
                        full_ends - starts, axis=1
                    )
                )
            )
            if full_horizon <= UNKNOWN_EXPLORATION_HORIZON:
                first_scale = 1.0
            elif full_horizon <= current_horizon + 1.0e-12:
                first_scale = 0.0
            else:
                first_scale = float(
                    np.clip(
                        (UNKNOWN_EXPLORATION_HORIZON - current_horizon)
                        / (full_horizon - current_horizon),
                        0.0,
                        1.0,
                    )
                )
            trial_scales = (
                first_scale,
                0.5 * first_scale,
                0.25 * first_scale,
                0.125 * first_scale,
                0.0,
            )
            for scale in trial_scales:
                q_trial = q_before.copy()
                if scale:
                    mujoco.mj_integratePos(
                        model, q_trial, qdot, DT * scale
                    )
                set_configuration(model, data, q_trial)
                ends = certificate_world_positions(data, robot)
                if certificate == "ellipsoid":
                    sweep_centers, sweep_shapes = (
                        _conservative_ellipsoid_sweep_outer_shapes(
                            model,
                            data,
                            robot_ellipsoids,
                            q_before,
                            q_trial,
                        )
                    )
                    sweep = snapshot.occupancy.certify_ellipsoids(
                        sweep_centers, sweep_shapes
                    )
                else:
                    sweep = snapshot.occupancy.certify_swept_spheres(
                        starts, ends, radii, margin=0.0
                    )
                unknown_horizon = float(
                    np.max(
                        np.linalg.norm(
                            ends - starts, axis=1
                        )
                    )
                )
                unknown_frontier_safe = (
                    sweep.unknown_voxels > 0
                    and unknown_horizon
                    <= UNKNOWN_EXPLORATION_HORIZON + 1.0e-12
                )
                # Occupied depth returns are already conservatively enclosed
                # by the matched online proxies and enter LiuQP as hard CBF
                # rows.  The voxel map is the independent unknown-space gate;
                # rejecting occupied voxel boxes here would inflate the same
                # measurement a second time by the map voxel half diagonal.
                known_space_safe = sweep.unknown_voxels == 0
                if known_space_safe or unknown_frontier_safe:
                    selected_scale = scale
                    accepted = True
                    unknown_exposure_steps += int(unknown_frontier_safe)
                    break
                unknown_rejections += int(sweep.unknown_voxels > 0)
                occupied_rejections += int(sweep.occupied_voxels > 0)
            assert sweep is not None
            if not accepted:
                set_configuration(model, data, q_before)
            ee = attachment_position(model, data)
            error = float(np.linalg.norm(ee - target))
            world_distance = minimum_mujoco_world_contact(data, model)
            self_distance = minimum_mujoco_self_contact(data, model)
            exact_bad = (
                (np.isfinite(world_distance) and world_distance < -1.0e-7)
                or (np.isfinite(self_distance) and self_distance < -1.0e-7)
            )
            exact_violations += int(exact_bad)
            success_hold = success_hold + 1 if error <= SUCCESS_TOLERANCE else 0
            cycle_ms = (time.perf_counter() - cycle_started) * 1000.0
            metrics_row = metrics.as_dict()
            row = {
                "step": step,
                "sim_time_s": step * DT,
                "certificate": certificate,
                "ee_x": float(ee[0]),
                "ee_y": float(ee[1]),
                "ee_z": float(ee[2]),
                "ee_error_m": error,
                "guidance_target_index": guidance_index,
                "guidance_target_x": float(active_target[0]),
                "guidance_target_y": float(active_target[1]),
                "guidance_target_z": float(active_target[2]),
                "guidance_progress_m": guidance_progress,
                "guidance_total_length_m": float(guidance_cumulative[-1]),
                "path_lookahead_m": PATH_LOOKAHEAD,
                "plan_version": plan_version,
                "replan_inflight": replan_future is not None,
                "selected_scale": selected_scale,
                "cycle_ms": cycle_ms,
                "map_generation": snapshot.generation,
                "map_source_step": snapshot.source_step,
                "map_age_steps": step - snapshot.source_step,
                "proxy_count": len(snapshot.proxies.centers),
                "snapshot_sha256": snapshot.proxy_stats[
                    "snapshot_sha256"
                ],
                "unknown_voxels": sweep.unknown_voxels,
                "occupied_voxels": sweep.occupied_voxels,
                "checked_voxels": sweep.checked_voxels,
                "unknown_horizon_m": unknown_horizon,
                "unknown_frontier_accepted": bool(
                    accepted and not sweep.safe
                ),
                "exact_world_contact_distance_m": (
                    None if not np.isfinite(world_distance) else world_distance
                ),
                "exact_self_contact_distance_m": (
                    None if not np.isfinite(self_distance) else self_distance
                ),
                **metrics_row,
            }
            rows.append(row)
            trajectories.append(data.qpos[:6].copy())
            if step > 0 and step % 100 == 0:
                np.savez_compressed(
                    output / "partial_control_checkpoint.npz",
                    q=np.asarray(trajectories),
                    ee=np.asarray(
                        [[item["ee_x"], item["ee_y"], item["ee_z"]] for item in rows]
                    ),
                    final_error=np.asarray(
                        [item["ee_error_m"] for item in rows]
                    ),
                    selected_scale=np.asarray(
                        [item["selected_scale"] for item in rows]
                    ),
                    last_step=np.asarray(step, dtype=np.int64),
                )
            if error < best_final_error - 0.003:
                best_final_error = error
                last_progress_step = step
            if (
                replan_future is None
                and step - last_progress_step >= REPLAN_STALL_STEPS
                and step - last_replan_step >= REPLAN_BACKOFF_STEPS
                and snapshot.generation > planning_map_generation
            ):
                geometry = _planner_geometry(
                    certificate, model, data, robot, snapshot
                )
                event = {
                    "trigger_step": step,
                    "source_map_generation": snapshot.generation,
                    "source_map_step": snapshot.source_step,
                    "source_snapshot_sha256": _snapshot_sha256(snapshot),
                    "start_m": ee.tolist(),
                    "best_final_error_m": best_final_error,
                    "trigger_reason": "no_3mm_final_error_improvement_for_20s",
                }
                replan_events.append(event)
                replan_event_index = len(replan_events) - 1
                replan_future = replan_executor.submit(
                    _solve_planner_problem,
                    ee.copy(),
                    target.copy(),
                    geometry,
                )
                last_replan_step = step
                snapshot_file = output / f"replan_snapshot_{len(replan_events)}.npz"
                np.savez_compressed(
                    snapshot_file,
                    **_snapshot_arrays(snapshot),
                )
            if step % SENSOR_PERIOD_STEPS == 0:
                worker.submit(step, step * DT, data.qpos[:6].copy())
            if exact_bad or success_hold >= SUCCESS_HOLD_STEPS:
                break
            if realtime:
                remaining = deadline - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        worker.close()
        replan_executor.shutdown(wait=True, cancel_futures=False)

    if replan_future is not None:
        final_plan, final_replan_error, final_replan_ms = replan_future.result()
        assert replan_event_index is not None
        replan_events[replan_event_index].update(
            {
                "completed_after_control_stop": True,
                "planner_ms": final_replan_ms,
                "error": final_replan_error,
                "success": final_plan is not None,
                "target_count": 0 if final_plan is None else len(final_plan.control_targets),
                "region_count": 0 if final_plan is None else len(final_plan.regions),
                "minimum_clearance_m": (
                    None if final_plan is None else final_plan.minimum_path_clearance
                ),
            }
        )
    (output / "online_replan_events.json").write_text(
        json.dumps(replan_events, indent=2), encoding="utf-8"
    )

    fields = sorted({key for row in rows for key in row})
    with (output / "cycles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output / "trajectory.npz",
        q=np.asarray(trajectories),
        target=target,
    )
    final_snapshot = snapshot
    np.savez_compressed(
        output / "final_causal_map_and_proxies.npz",
        **_snapshot_arrays(final_snapshot),
    )
    cycle_times = np.asarray([row["cycle_ms"] for row in rows])
    solve_times = np.asarray([row["solve_ms"] for row in rows])
    result = {
        "scene": scene.name,
        "certificate": certificate,
        "causal_online": True,
        "mounted_depth_cameras": [
            "ur5_depth_wrist",
            "ur5_depth_shoulder",
        ],
        "uses_future_or_other_trajectory_frames": False,
        "uses_exact_collision_for_control": False,
        "uses_human_waypoints": False,
        "uses_autonomous_iris_region_guidance": plan is not None,
        "robot_qp_certificate": (
            "conservative_sphere_chain"
            if certificate == "sphere"
            else "pose_rotated_conservative_ellipsoid_chain"
        ),
        "occupancy_sweep_certificate": (
            "conservative_sphere_chain"
            if certificate == "sphere"
            else "continuous_outer_ellipsoid_chain_exact_voxel_box"
        ),
        "occupied_collision_policy": "matched_proxy_hard_cbf_constraints",
        "voxel_map_policy": "unknown_space_gate_with_bounded_frontier_step",
        "planning_snapshot_sha256": planning_snapshot_sha256,
        "planning_proxy_snapshot_sha256": planning_proxy_snapshot_sha256,
        "planning_map_generation": planning_map_generation,
        "planning_map_source_step": planning_map_source_step,
        "output_tag": output_tag,
        "planner_error": planner_error,
        "planner_ms": planner_ms,
        "planner_target_count": (
            0 if plan is None else len(plan.control_targets)
        ),
        "planner_region_count": 0 if plan is None else len(plan.regions),
        "planner_tool_proxy_count": len(planner_tool_shapes),
        "configuration_corridor_error": configuration_corridor_error,
        "configuration_corridor_target_count": (
            0
            if configuration_corridor is None
            else len(configuration_corridor.q_targets)
        ),
        "configuration_corridor_minimum_proxy_clearance_m": (
            None
            if configuration_corridor is None
            else configuration_corridor.minimum_proxy_clearance
        ),
        "configuration_corridor_attempted_goal_ik": (
            0
            if configuration_corridor is None
            else configuration_corridor.attempted_goal_ik
        ),
        "configuration_corridor_elapsed_ms": (
            None
            if configuration_corridor is None
            else configuration_corridor.elapsed_ms
        ),
        "configuration_corridor_planning_method": (
            None
            if configuration_corridor is None
            else configuration_corridor.planning_method
        ),
        "configuration_corridor_enabled": use_configuration_corridor,
        "online_replan_trigger_stall_steps": REPLAN_STALL_STEPS,
        "online_replan_backoff_steps": REPLAN_BACKOFF_STEPS,
        "online_replan_events": replan_events,
        "online_replan_success_count": sum(
            int(event.get("success", False)) for event in replan_events
        ),
        "final_plan_version": plan_version,
        "planner_minimum_clearance_m": (
            None if plan is None else plan.minimum_path_clearance
        ),
        "steps": len(rows),
        "success": success_hold >= SUCCESS_HOLD_STEPS,
        "final_error_m": rows[-1]["ee_error_m"],
        "maximum_ee_x_m": max(row["ee_x"] for row in rows),
        "exact_collision_violations": exact_violations,
        "unknown_rejection_events": unknown_rejections,
        "unknown_frontier_exposure_steps": unknown_exposure_steps,
        "unknown_exploration_horizon_m": UNKNOWN_EXPLORATION_HORIZON,
        "common_near_weight": COMMON_NEAR_WEIGHT,
        "common_near_distance_m": COMMON_NEAR_DISTANCE,
        "mvt_query_padding_m": MVT_QUERY_PADDING,
        "occupied_rejection_events": occupied_rejections,
        "final_proxy_count": len(final_snapshot.proxies.centers),
        "final_snapshot_sha256": final_snapshot.proxy_stats[
            "snapshot_sha256"
        ],
        "control_cycle_median_ms": float(np.median(cycle_times)),
        "control_cycle_p95_ms": float(np.percentile(cycle_times, 95)),
        "control_cycle_p99_ms": float(np.percentile(cycle_times, 99)),
        "qp_solve_median_ms": float(np.median(solve_times)),
        "qp_solve_p99_ms": float(np.percentile(solve_times, 99)),
        "final_map_generation": final_snapshot.generation,
        "final_map_stats": final_snapshot.map_stats,
        "final_proxy_update_stats": final_snapshot.proxy_stats,
        "final_perception_capture_ms": final_snapshot.capture_ms,
        "final_perception_total_update_ms": final_snapshot.total_update_ms,
        "native_mvt_stats": asdict(final_snapshot.obstacle_index.stats),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--certificate", choices=("sphere", "ellipsoid", "both"), default="both"
    )
    parser.add_argument("--duration", type=float, default=35.0)
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument(
        "--taskspace-only",
        action="store_true",
        help="track the autonomous IRIS region path directly with LiuQP",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="optional results_incremental_v2 subdirectory (for example formal_v2)",
    )
    args = parser.parse_args()
    certificates = (
        ("sphere", "ellipsoid")
        if args.certificate == "both"
        else (args.certificate,)
    )
    results = [
        run(
            value,
            args.duration,
            realtime=not args.no_realtime,
            output_tag=args.output_tag,
            use_configuration_corridor=not args.taskspace_only,
        )
        for value in certificates
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
