"""Extracted original v4.3 ellipsoid dependency; see docs/source_manifest.json."""
from __future__ import annotations
import atexit
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
import csv
import ctypes
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import mujoco
import numpy as np
from camera import UR5MountedDepthCamera, environment_endpoint_mask
from geometry import closest_point_on_ellipsoid
from proxy_manager import IncrementalMatchedProxyManager
from robot import DT, JOINT_NAMES, attachment_position, build_model, build_robot_certificate, build_xml, certificate_world_positions, set_configuration
from voxel_index import NativeMultilevelMVT
from native_support import NativeEllipsoidSupport
from occupancy import NativeIncrementalOccupancyMap, NativeOccupancySnapshot
from observability import ObservedPointSnapshot, evaluate_observed_before_risk, snapshot_from_proxies
from scene import CAMERA_HZ, INITIAL_CALIBRATED_FREE_PADDING, SUCCESS_TOLERANCE, formal_drawer_camera_quarter_scene
from controller import ProtocolLiuQPController
from proxy_geometry import NORMAL_ESTIMATION_WORKERS
from online_helpers import MAP_VOXEL_SIZE, MAXIMUM_AABB_OVERSHOOT, PROXY_CLUSTER_SIZE, PROXY_WORKSPACE_LOWER, PROXY_WORKSPACE_UPPER, _build_mvt_and_audit, _build_mvt_only, _crop_proxy_observation, _controller_uncertainty_shapes, _environment_half_extents, _new_controller
CAMERA_NAMES_WRIST_PAIR_SHOULDER = ('ur5_depth_wrist', 'ur5_depth_wrist_right', 'ur5_depth_shoulder')
CAMERA_NAMES_WRIST_SHOULDER = ('ur5_depth_wrist', 'ur5_depth_shoulder')
CAMERA_NAMES = ('ur5_depth_wrist', 'ur5_depth_forearm')
CAMERA_NAMES_DUAL_SHOULDER = CAMERA_NAMES_WRIST_PAIR_SHOULDER + ('ur5_depth_shoulder_right',)
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 180
CAMERA_PIXEL_STRIDE = 1
CAMERA_MINIMUM_RANGE = 0.07
CENTERVOX_SIZE = 0.0075
SUCCESS_HOLD_CYCLES = 10
OBSERVABILITY_DISTANCE = 0.1
OBSERVABILITY_GUARD_TIME = 0.1
OBSERVABILITY_TRUTH_SPACING = 0.012
_PROCESS_WORKER: PerceptionWorker | None = None
_PROCESS_DENSE_MAP = True
_PROCESS_MAP_DELTA = False
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'runs'

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'refusing to write empty CSV: {path}')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _set_current_thread_affinity(cpus: tuple[int, ...] | None) -> int | set[int] | None:
    if cpus is None:
        return None
    if not cpus:
        raise ValueError('thread CPU affinity cannot be empty')
    if platform.system() == 'Windows':
        mask = sum((1 << int(cpu) for cpu in cpus))
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.GetCurrentThread.restype = ctypes.c_void_p
        kernel32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetThreadAffinityMask.restype = ctypes.c_size_t
        previous = int(kernel32.SetThreadAffinityMask(kernel32.GetCurrentThread(), ctypes.c_size_t(mask)))
        if previous == 0:
            raise OSError(ctypes.get_last_error(), 'thread affinity failed')
        return previous
    previous = set(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(cpus))
    return previous

def _restore_current_thread_affinity(previous: int | set[int] | None) -> None:
    if previous is None:
        return
    if platform.system() == 'Windows':
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.GetCurrentThread.restype = ctypes.c_void_p
        kernel32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetThreadAffinityMask.restype = ctypes.c_size_t
        if kernel32.SetThreadAffinityMask(kernel32.GetCurrentThread(), ctypes.c_size_t(int(previous))) == 0:
            raise OSError(ctypes.get_last_error(), 'thread affinity restore failed')
    else:
        os.sched_setaffinity(0, set(previous))

def _initialize_thread_affinity(cpus: tuple[int, ...] | None) -> None:
    _set_current_thread_affinity(cpus)

def _save_observability_snapshots(path: Path, snapshots: list[ObservedPointSnapshot]) -> None:
    counts = np.asarray([len(item.points) for item in snapshots], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    if int(offsets[-1]) > 0:
        points = np.concatenate([item.points for item in snapshots], axis=0)
        cover_radii = np.concatenate([item.cover_radii for item in snapshots], axis=0)
    else:
        points = np.empty((0, 3), dtype=float)
        cover_radii = np.empty(0, dtype=float)
    np.savez_compressed(path, publish_cycles=np.asarray([item.publish_cycle for item in snapshots], dtype=np.int64), source_cycles=np.asarray([item.source_cycle for item in snapshots], dtype=np.int64), offsets=offsets, points=points, cover_radii=cover_radii)

def _save_proxy_snapshots(path: Path, snapshots: list[tuple[int, int, object]]) -> None:
    counts = np.asarray([len(item[2].centers) for item in snapshots], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))

    def concatenate(field: str, empty_shape: tuple[int, ...], dtype=float):
        if int(offsets[-1]) == 0:
            return np.empty(empty_shape, dtype=dtype)
        return np.concatenate([np.asarray(getattr(item[2], field), dtype=dtype) for item in snapshots], axis=0)
    np.savez_compressed(path, publish_cycles=np.asarray([item[0] for item in snapshots], dtype=np.int64), source_cycles=np.asarray([item[1] for item in snapshots], dtype=np.int64), offsets=offsets, proxy_ids=concatenate('proxy_ids', (0,), np.int64), centers=concatenate('centers', (0, 3)), sphere_radii=concatenate('sphere_radii', (0,)), base_sphere_radii=concatenate('base_sphere_radii', (0,)), ellipsoid_shapes=concatenate('base_ellipsoid_shapes', (0, 3, 3)), ellipsoid_outer_shapes=concatenate('ellipsoid_outer_shapes', (0, 3, 3)), proxy_uncertainty_shapes=concatenate('proxy_uncertainty_shapes', (0, 3, 3)), uncertainty_offsets=concatenate('proxy_offset_radii', (0,)))

def _save_occupancy_snapshots(path: Path, snapshots: list[tuple[int, int, np.ndarray, np.ndarray]]) -> None:
    counts = np.asarray([len(item[2]) for item in snapshots], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    keys = np.concatenate([np.asarray(item[2], dtype=np.int32) for item in snapshots]) if int(offsets[-1]) else np.empty((0, 3), dtype=np.int32)
    states = np.concatenate([np.asarray(item[3], dtype=np.int8) for item in snapshots]) if int(offsets[-1]) else np.empty(0, dtype=np.int8)
    np.savez_compressed(path, publish_cycles=np.asarray([item[0] for item in snapshots], dtype=np.int64), source_cycles=np.asarray([item[1] for item in snapshots], dtype=np.int64), offsets=offsets, voxel_keys=keys, voxel_states=states, voxel_size_m=np.asarray(MAP_VOXEL_SIZE, dtype=float), storage=np.asarray('discrete_state_change_deltas'))

def _save_source_configurations(path: Path, snapshots: list[tuple[int, int, np.ndarray]]) -> None:
    np.savez_compressed(path, publish_cycles=np.asarray([item[0] for item in snapshots], dtype=np.int64), source_cycles=np.asarray([item[1] for item in snapshots], dtype=np.int64), q=np.asarray([item[2] for item in snapshots], dtype=float))

@dataclass
class PerceptionPacket:
    generation: int
    requested_frame: int
    source_cycle: int
    source_time_s: float
    source_q: np.ndarray
    proxies: object
    occupancy: NativeOccupancySnapshot | None
    mvt: NativeMultilevelMVT | None
    frame_row: dict
    completed_wall_time: float
    occupancy_keys: np.ndarray | None = None
    occupancy_states: np.ndarray | None = None
    occupancy_is_delta: bool = False
    ellipsoid_eigenvalues: np.ndarray | None = None
    ellipsoid_rotations: np.ndarray | None = None
    uncertainty_eigenvalues: np.ndarray | None = None

    def close(self) -> None:
        if self.occupancy is not None:
            self.occupancy.close()
        if self.mvt is not None:
            self.mvt.close()

@dataclass
class SerializedPerceptionPacket:
    generation: int
    requested_frame: int
    source_cycle: int
    source_time_s: float
    source_q: np.ndarray
    source_robot_positions: np.ndarray
    proxies: object
    occupancy_keys: np.ndarray | None
    occupancy_states: np.ndarray | None
    dense_map_published: bool
    map_delta_published: bool
    frame_row: dict
    completed_wall_time: float
    ellipsoid_eigenvalues: np.ndarray | None = None
    ellipsoid_rotations: np.ndarray | None = None
    uncertainty_eigenvalues: np.ndarray | None = None

class PerceptionWorker:

    def __init__(self, representation: str, index_mode: str, centervox_size: float=CENTERVOX_SIZE, maximum_uncertainty_union_inflation: float | None=None, certificate_radius_limit: float | None=None, uncertainty_fusion_mode: str='separate_uncertainty', direct_thin_axis_inflation: float=np.sqrt(2.0), direct_tangent_subdivisions: int=1, direct_partition_mode: str='grid', camera_width: int=CAMERA_WIDTH, camera_height: int=CAMERA_HEIGHT, camera_pixel_stride: int=CAMERA_PIXEL_STRIDE, camera_names: tuple[str, ...]=CAMERA_NAMES, scene_version: str='formal', track_map_deltas: bool=False) -> None:
        self.representation = representation
        self.index_mode = index_mode
        self.centervox_size = float(centervox_size)
        if self.centervox_size <= 0.0:
            raise ValueError('centervox_size must be positive')
        if certificate_radius_limit is not None and certificate_radius_limit <= 0.0:
            raise ValueError('certificate radius limit must be positive')
        self.certificate_radius_limit = None if certificate_radius_limit is None else float(certificate_radius_limit)
        if uncertainty_fusion_mode not in {'separate_uncertainty', 'fused_certified_ellipsoid', 'fused_certified_centervox'}:
            raise ValueError('unknown uncertainty_fusion_mode')
        self.uncertainty_fusion_mode = uncertainty_fusion_mode
        if camera_width <= 0 or camera_height <= 0 or camera_pixel_stride <= 0:
            raise ValueError('camera dimensions and pixel stride must be positive')
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.camera_pixel_stride = int(camera_pixel_stride)
        self.camera_names = tuple(camera_names)
        if not self.camera_names:
            raise ValueError('at least one local camera is required')
        self.scene = _protocol_scene(scene_version)
        self.model = build_model(self.scene)
        self.data = mujoco.MjData(self.model)
        set_configuration(self.model, self.data, np.asarray(self.scene.q0, dtype=float))
        self.robot = build_robot_certificate(self.model)
        self.robot_radii = np.asarray([sphere.radius for sphere in self.robot], dtype=float)
        self.occupancy = NativeIncrementalOccupancyMap(MAP_VOXEL_SIZE, track_state_deltas=track_map_deltas)
        self.proxy_manager = IncrementalMatchedProxyManager(filter_size=self.centervox_size, cluster_size=PROXY_CLUSTER_SIZE, maximum_aabb_overshoot=MAXIMUM_AABB_OVERSHOOT, maximum_uncertainty_union_inflation=maximum_uncertainty_union_inflation, certificate_radius_limit=self.certificate_radius_limit, uncertainty_fusion_mode=self.uncertainty_fusion_mode, direct_thin_axis_inflation=direct_thin_axis_inflation, direct_tangent_subdivisions=direct_tangent_subdivisions, direct_partition_mode=direct_partition_mode)
        self.camera = UR5MountedDepthCamera(self.model, camera_names=self.camera_names, width=self.camera_width, height=self.camera_height, pixel_stride=self.camera_pixel_stride, minimum_range=CAMERA_MINIMUM_RANGE, optical_depth_error_bound=0.003, occluding_self_filter=True)
        self._map_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='causal-occupancy')
        self.generation = -1

    def process(self, q: np.ndarray, requested_frame: int, source_cycle: int, source_time_s: float, transferable: bool=False, include_dense_snapshot: bool=True, include_map_delta: bool=False) -> PerceptionPacket | SerializedPerceptionPacket:
        started = time.perf_counter()
        q = np.asarray(q, dtype=float).copy()
        set_configuration(self.model, self.data, q)
        robot_positions = certificate_world_positions(self.data, self.robot)
        capture_started = time.perf_counter()
        observations = self.camera.capture(self.data)
        capture_ms = (time.perf_counter() - capture_started) * 1000.0

        def update_map():
            map_started = time.perf_counter()
            stats = self.occupancy.integrate(observations)
            self.occupancy.mark_current_robot_free(robot_positions, self.robot_radii, padding=INITIAL_CALIBRATED_FREE_PADDING if self.generation < 0 else 0.0)
            return (stats, (time.perf_counter() - map_started) * 1000.0)
        map_future = self._map_executor.submit(update_map)
        point_batches: list[np.ndarray] = []
        radius_batches: list[np.ndarray] = []
        uncertainty_batches: list[np.ndarray] = []
        raw_points_by_camera: dict[str, int] = {}
        environment_points_by_camera: dict[str, int] = {}
        self_returns_by_camera: dict[str, int] = {}
        workspace_points_by_camera: dict[str, int] = {}
        for observation in observations:
            endpoint_mask = environment_endpoint_mask(observation)
            raw_points_by_camera[observation.camera_name] = int(len(observation.points))
            environment_points_by_camera[observation.camera_name] = int(np.count_nonzero(endpoint_mask))
            self_returns_by_camera[observation.camera_name] = int(len(observation.points) - np.count_nonzero(endpoint_mask))
            points, radii = _crop_proxy_observation(observation)
            workspace_points_by_camera[observation.camera_name] = int(len(points))
            if len(points):
                if observation.sample_uncertainty_shapes is None:
                    raise RuntimeError('depth frame omitted directional uncertainty')
                raw_points = np.asarray(observation.points, dtype=float)
                keep = endpoint_mask.copy()
                keep &= np.all(raw_points >= PROXY_WORKSPACE_LOWER, axis=1)
                keep &= np.all(raw_points <= PROXY_WORKSPACE_UPPER, axis=1)
                point_batches.append(points)
                radius_batches.append(radii)
                uncertainty_batches.append(np.asarray(observation.sample_uncertainty_shapes, dtype=float)[keep])
        if not point_batches:
            raise RuntimeError('causal camera frame contained no workspace points')
        proxy_started = time.perf_counter()
        update_stats = self.proxy_manager.update(np.vstack(point_batches), np.concatenate(radius_batches), np.vstack(uncertainty_batches))
        manager_update_ms = (time.perf_counter() - proxy_started) * 1000.0
        coverage_started = time.perf_counter()
        coverage = self.proxy_manager.coverage_audit()
        coverage_audit_ms = (time.perf_counter() - coverage_started) * 1000.0
        proxy_update_ms = (time.perf_counter() - proxy_started) * 1000.0
        map_stats, map_update_ms = map_future.result()
        if not coverage.all_raw_sample_balls_certified:
            raise AssertionError('CenterVox certificate coverage failed')
        proxies = self.proxy_manager.snapshot
        effective_sphere_radii = np.asarray(proxies.sphere_radii, dtype=float) + np.asarray(proxies.proxy_offset_radii, dtype=float)
        outer_eigenvalues = np.linalg.eigvalsh(np.asarray(proxies.ellipsoid_outer_shapes, dtype=float))
        ellipsoid_outer_longest_radii = np.sqrt(np.maximum(outer_eigenvalues[:, -1], 0.0)) + np.asarray(proxies.proxy_offset_radii, dtype=float)
        radius_limit_ok = bool(self.certificate_radius_limit is None or (np.all(np.abs(effective_sphere_radii - self.certificate_radius_limit) <= 1e-09) and np.all(ellipsoid_outer_longest_radii <= self.certificate_radius_limit + 1e-09)))
        if not radius_limit_ok:
            raise AssertionError('published proxy snapshot violated radius protocol')
        if self.representation == 'ellipsoid':
            ellipsoid_eigenvalues, ellipsoid_rotations = np.linalg.eigh(np.asarray(proxies.base_ellipsoid_shapes, dtype=float))
            if np.array_equal(np.asarray(proxies.ellipsoid_outer_shapes), np.asarray(proxies.base_ellipsoid_shapes)):
                outer_eigenvalues = ellipsoid_eigenvalues
            uncertainty_eigenvalues = None if proxies.proxy_uncertainty_shapes is None else np.linalg.eigvalsh(np.asarray(proxies.proxy_uncertainty_shapes, dtype=float))
        else:
            ellipsoid_eigenvalues = None
            ellipsoid_rotations = None
            uncertainty_eigenvalues = None
        snapshot_started = time.perf_counter()
        immutable_occupancy = None
        occupancy_keys = None
        occupancy_states = None
        if include_dense_snapshot:
            occupancy_keys, occupancy_states = self.occupancy.snapshot_arrays()
        elif include_map_delta:
            occupancy_keys, occupancy_states = self.occupancy.snapshot_delta_arrays()
        if not transferable:
            immutable_occupancy = self.occupancy.clone_snapshot()
        snapshot_clone_ms = (time.perf_counter() - snapshot_started) * 1000.0
        mvt = None
        mvt_missing = 0
        mvt_levels = 0
        mvt_index_references = 0
        mvt_cell_lookups = 0
        mvt_started = time.perf_counter()
        try:
            if self.index_mode != 'full_scan':
                built_mvt, mvt_missing = _build_mvt_and_audit(proxies, self.representation, robot_positions, self.robot_radii, simd=self.index_mode == 'mvt_simd')
                mvt_levels = built_mvt.stats.level_count
                mvt_index_references = built_mvt.stats.index_references
                mvt_cell_lookups = built_mvt.stats.cell_lookups_per_query
                if transferable:
                    built_mvt.close()
                else:
                    mvt = built_mvt
        except Exception:
            if immutable_occupancy is not None:
                immutable_occupancy.close()
            raise
        mvt_update_ms = (time.perf_counter() - mvt_started) * 1000.0
        self.generation += 1
        frame_pipeline_ms = (time.perf_counter() - started) * 1000.0
        frame_row = {'generation': self.generation, 'requested_frame': requested_frame, 'source_cycle': source_cycle, 'source_time_s': source_time_s, 'raw_depth_points': int(sum((len(item.points) for item in observations))), 'workspace_depth_points': int(sum((len(item) for item in point_batches))), 'raw_points_by_camera': json.dumps(raw_points_by_camera, sort_keys=True), 'environment_points_by_camera': json.dumps(environment_points_by_camera, sort_keys=True), 'self_returns_by_camera': json.dumps(self_returns_by_camera, sort_keys=True), 'workspace_points_by_camera': json.dumps(workspace_points_by_camera, sort_keys=True), 'free_voxels': map_stats.free_voxels, 'occupied_voxels': map_stats.occupied_voxels, 'center_voxels': update_stats.center_voxels, 'proxy_count': update_stats.proxy_count, 'proxy_snapshot_sha256': update_stats.snapshot_sha256, 'capture_ms': capture_ms, 'map_update_ms': map_update_ms, 'proxy_update_ms': proxy_update_ms, 'proxy_manager_update_ms': manager_update_ms, 'proxy_coverage_audit_ms': coverage_audit_ms, 'proxy_frame_centervox_ms': update_stats.frame_centervox_ms, 'proxy_history_union_ms': update_stats.history_union_ms, 'proxy_cell_merge_ms': update_stats.cell_merge_ms, 'proxy_bucket_fit_ms': update_stats.bucket_fit_ms, 'proxy_publish_ids_hash_ms': update_stats.publish_ids_hash_ms, 'snapshot_clone_ms': snapshot_clone_ms, 'mvt_update_ms': mvt_update_ms, 'frame_pipeline_ms': frame_pipeline_ms, 'coverage_ok': coverage.all_raw_sample_balls_certified, 'certificate_radius_limit_m': self.certificate_radius_limit, 'certificate_radius_limit_ok': radius_limit_ok, 'sphere_effective_radius_min_m': float(np.min(effective_sphere_radii)), 'sphere_effective_radius_max_m': float(np.max(effective_sphere_radii)), 'ellipsoid_outer_longest_radius_max_m': float(np.max(ellipsoid_outer_longest_radii)), 'directional_measurement_uncertainty': True, 'map_proxy_updates_overlapped': True, 'minimum_centervox_cover_slack': coverage.minimum_centervox_cover_slack, 'minimum_proxy_offset_slack': coverage.minimum_proxy_offset_slack, 'maximum_base_ellipsoid_value': coverage.maximum_base_ellipsoid_value, 'mvt_missing_candidates': mvt_missing, 'mvt_levels': mvt_levels, 'mvt_index_references': mvt_index_references, 'mvt_cell_lookups_per_robot_query': mvt_cell_lookups}
        common = dict(generation=self.generation, requested_frame=requested_frame, source_cycle=source_cycle, source_time_s=source_time_s, source_q=q, proxies=proxies, frame_row=frame_row, completed_wall_time=time.perf_counter(), ellipsoid_eigenvalues=ellipsoid_eigenvalues, ellipsoid_rotations=ellipsoid_rotations, uncertainty_eigenvalues=uncertainty_eigenvalues)
        if transferable:
            return SerializedPerceptionPacket(**common, source_robot_positions=robot_positions.copy(), occupancy_keys=None if occupancy_keys is None else np.asarray(occupancy_keys), occupancy_states=None if occupancy_states is None else np.asarray(occupancy_states), dense_map_published=include_dense_snapshot, map_delta_published=include_map_delta and (not include_dense_snapshot))
        return PerceptionPacket(**common, occupancy=immutable_occupancy, occupancy_keys=occupancy_keys, occupancy_states=occupancy_states, occupancy_is_delta=include_map_delta and (not include_dense_snapshot), mvt=mvt)

    def close(self) -> None:
        self._map_executor.shutdown(wait=True, cancel_futures=False)
        self.camera.close()
        self.occupancy.close()

def _initialize_process_worker(representation: str, index_mode: str, publish_dense_map: bool, publish_map_delta: bool, centervox_size: float, maximum_uncertainty_union_inflation: float | None, certificate_radius_limit: float | None, uncertainty_fusion_mode: str, direct_thin_axis_inflation: float, direct_tangent_subdivisions: int, direct_partition_mode: str, camera_width: int, camera_height: int, camera_pixel_stride: int, camera_names: tuple[str, ...], scene_version: str, process_cpu_affinity: tuple[int, ...] | None) -> None:
    global _PROCESS_WORKER, _PROCESS_DENSE_MAP, _PROCESS_MAP_DELTA
    import psutil
    worker_process = psutil.Process()
    if platform.system() == 'Windows':
        worker_process.nice(psutil.NORMAL_PRIORITY_CLASS)
    if process_cpu_affinity is not None:
        worker_process.cpu_affinity(list(process_cpu_affinity))
    _PROCESS_WORKER = PerceptionWorker(representation, index_mode, centervox_size, maximum_uncertainty_union_inflation, certificate_radius_limit, uncertainty_fusion_mode, direct_thin_axis_inflation, direct_tangent_subdivisions, direct_partition_mode, camera_width, camera_height, camera_pixel_stride, camera_names, scene_version, track_map_deltas=publish_map_delta)
    _PROCESS_DENSE_MAP = bool(publish_dense_map)
    _PROCESS_MAP_DELTA = bool(publish_map_delta)
    atexit.register(_PROCESS_WORKER.close)

def _process_perception_request(q: np.ndarray, requested_frame: int, source_cycle: int, source_time_s: float) -> SerializedPerceptionPacket:
    if _PROCESS_WORKER is None:
        raise RuntimeError('perception process was not initialized')
    result = _PROCESS_WORKER.process(q, requested_frame, source_cycle, source_time_s, transferable=True, include_dense_snapshot=_PROCESS_DENSE_MAP, include_map_delta=_PROCESS_MAP_DELTA)
    if not isinstance(result, SerializedPerceptionPacket):
        raise AssertionError('process worker returned a native-handle packet')
    return result

def _materialize_packet(packet: PerceptionPacket | SerializedPerceptionPacket, representation: str, index_mode: str, robot_radii: np.ndarray) -> PerceptionPacket:
    if isinstance(packet, PerceptionPacket):
        return packet
    started = time.perf_counter()
    occupancy = None
    if packet.dense_map_published:
        if packet.occupancy_keys is None or packet.occupancy_states is None:
            raise AssertionError('dense-map packet omitted its arrays')
        occupancy = NativeOccupancySnapshot(packet.occupancy_keys, packet.occupancy_states, MAP_VOXEL_SIZE)
    mvt = None
    missing = int(packet.frame_row.get('mvt_missing_candidates', 0))
    try:
        if index_mode != 'full_scan':
            mvt = _build_mvt_only(packet.proxies, representation, robot_radii, simd=index_mode == 'mvt_simd')
    except Exception:
        if occupancy is not None:
            occupancy.close()
        raise
    row = dict(packet.frame_row)
    row['mvt_missing_candidates'] = missing
    row['mvt_levels'] = 0 if mvt is None else mvt.stats.level_count
    row['mvt_index_references'] = 0 if mvt is None else mvt.stats.index_references
    row['mvt_cell_lookups_per_robot_query'] = 0 if mvt is None else mvt.stats.cell_lookups_per_query
    row['publication_materialize_ms'] = (time.perf_counter() - started) * 1000.0
    return PerceptionPacket(generation=packet.generation, requested_frame=packet.requested_frame, source_cycle=packet.source_cycle, source_time_s=packet.source_time_s, source_q=packet.source_q, proxies=packet.proxies, occupancy=occupancy, occupancy_keys=packet.occupancy_keys, occupancy_states=packet.occupancy_states, occupancy_is_delta=packet.map_delta_published, mvt=mvt, frame_row=row, completed_wall_time=packet.completed_wall_time, ellipsoid_eigenvalues=packet.ellipsoid_eigenvalues, ellipsoid_rotations=packet.ellipsoid_rotations, uncertainty_eigenvalues=packet.uncertainty_eigenvalues)

def _update_controller(controller: ProtocolLiuQPController, packet: PerceptionPacket, representation: str, prepared_guard: ContinuousProxyGuard | None=None) -> None:
    proxies = packet.proxies
    kwargs = dict(obstacle_offsets=proxies.proxy_offset_radii, proxy_ids=proxies.proxy_ids, obstacle_index=packet.mvt)
    if representation == 'sphere':
        kwargs['obstacle_radii'] = proxies.sphere_radii
    else:
        kwargs['obstacle_shapes'] = proxies.base_ellipsoid_shapes
        uncertainty_shapes = _controller_uncertainty_shapes(proxies)
        kwargs['obstacle_uncertainty_shapes'] = uncertainty_shapes
        if prepared_guard is not None:
            kwargs['obstacle_eigenvalues'] = prepared_guard.eigenvalues
            kwargs['obstacle_rotations'] = prepared_guard.rotations
            kwargs['obstacle_uncertainty_eigenvalues'] = None if uncertainty_shapes is None else prepared_guard.uncertainty_eigenvalues
    controller.update_obstacles(proxies.centers, **kwargs)

@dataclass(frozen=True)
class ProxySweepCertificate:
    safe: bool
    colliding_pairs: int
    recovery_pairs: int
    worsening_pairs: int
    candidate_pairs: int
    minimum_clearance_m: float

class ContinuousProxyGuard:

    def __init__(self, packet: PerceptionPacket, representation: str) -> None:
        self.proxies = packet.proxies
        self.representation = representation
        self.mvt = packet.mvt
        self.half_extents = _environment_half_extents(self.proxies, representation)
        self.directional_uncertainty_shapes = _controller_uncertainty_shapes(self.proxies) if representation == 'ellipsoid' else None
        if representation == 'ellipsoid':
            if packet.ellipsoid_eigenvalues is None or packet.ellipsoid_rotations is None:
                self.eigenvalues, self.rotations = np.linalg.eigh(np.asarray(self.proxies.base_ellipsoid_shapes, dtype=float))
            else:
                self.eigenvalues = np.asarray(packet.ellipsoid_eigenvalues, dtype=float)
                self.rotations = np.asarray(packet.ellipsoid_rotations, dtype=float)
            self.eigenvalues = np.maximum(self.eigenvalues, 1e-18)
            self.uncertainty_eigenvalues = packet.uncertainty_eigenvalues if packet.uncertainty_eigenvalues is not None else None if self.directional_uncertainty_shapes is None else np.linalg.eigvalsh(self.directional_uncertainty_shapes)
        else:
            self.eigenvalues = None
            self.rotations = None
            self.uncertainty_eigenvalues = None
        self.support_kernel = NativeEllipsoidSupport()
        self.normal_cache: dict[tuple[int, int], np.ndarray] = {}
        self.multiplier_cache: dict[tuple[int, int], float] = {}

    def _candidates(self, start: np.ndarray, end: np.ndarray, radius: float) -> np.ndarray:
        center = 0.5 * (start + end)
        half = 0.5 * np.abs(end - start) + float(radius)
        if self.mvt is None:
            lower = center - half
            upper = center + half
            return np.flatnonzero(np.all(self.proxies.centers - self.half_extents <= upper, axis=1) & np.all(self.proxies.centers + self.half_extents >= lower, axis=1))
        return self.mvt.query_aabb(center, half)

    def _certify_directional_pairs_batched(self, starts: np.ndarray, ends: np.ndarray, radii: np.ndarray) -> ProxySweepCertificate:
        robot_batches: list[np.ndarray] = []
        obstacle_batches: list[np.ndarray] = []
        for robot_index, (start, end, radius) in enumerate(zip(starts, ends, radii)):
            candidates = self._candidates(start, end, float(radius))
            if len(candidates):
                robot_batches.append(np.full(len(candidates), robot_index, dtype=np.int64))
                obstacle_batches.append(np.asarray(candidates, dtype=np.int64))
        if not obstacle_batches:
            return ProxySweepCertificate(safe=True, colliding_pairs=0, recovery_pairs=0, worsening_pairs=0, candidate_pairs=0, minimum_clearance_m=float('inf'))
        robot_indices = np.concatenate(robot_batches)
        obstacle_indices = np.concatenate(obstacle_batches)
        proxy_ids = np.asarray(self.proxies.proxy_ids, dtype=np.int64)[obstacle_indices]
        initial = np.zeros((len(obstacle_indices), 3), dtype=float)
        for pair_index, (robot_index, proxy_id) in enumerate(zip(robot_indices, proxy_ids)):
            previous = self.normal_cache.get((int(robot_index), int(proxy_id)))
            if previous is not None:
                initial[pair_index] = previous
        pair_radii = np.asarray(radii, dtype=float)[robot_indices]
        robot_shapes = np.eye(3)[None, :, :] * (pair_radii * pair_radii)[:, None, None]
        centers = np.asarray(self.proxies.centers, dtype=float)
        shapes = np.asarray(self.proxies.base_ellipsoid_shapes, dtype=float)
        uncertainty = np.asarray(self.directional_uncertainty_shapes, dtype=float)
        first_normals, _, _ = self.support_kernel.normals_sum_pairs_warm(np.asarray(starts, dtype=float)[robot_indices], robot_shapes, centers[obstacle_indices], shapes[obstacle_indices], uncertainty[obstacle_indices], initial, max_iterations=64)
        second_normals, _, _ = self.support_kernel.normals_sum_pairs_warm(np.asarray(ends, dtype=float)[robot_indices], robot_shapes, centers[obstacle_indices], shapes[obstacle_indices], uncertainty[obstacle_indices], first_normals, max_iterations=64)
        for pair_index, (robot_index, proxy_id) in enumerate(zip(robot_indices, proxy_ids)):
            self.normal_cache[int(robot_index), int(proxy_id)] = second_normals[pair_index].copy()
        pair_starts = np.asarray(starts, dtype=float)[robot_indices]
        pair_ends = np.asarray(ends, dtype=float)[robot_indices]
        pair_centers = centers[obstacle_indices]
        pair_shapes = shapes[obstacle_indices]
        pair_uncertainty = uncertainty[obstacle_indices]
        delta0 = pair_centers - pair_starts
        delta1 = pair_centers - pair_ends

        def support(normals: np.ndarray, matrices: np.ndarray) -> np.ndarray:
            return np.sqrt(np.maximum(np.einsum('ni,nij,nj->n', normals, matrices, normals), 0.0))
        d0 = np.maximum(np.einsum('ni,ni->n', first_normals, delta0) - support(first_normals, pair_shapes) - support(first_normals, pair_uncertainty), 0.0)
        d1 = np.maximum(np.einsum('ni,ni->n', second_normals, delta1) - support(second_normals, pair_shapes) - support(second_normals, pair_uncertainty), 0.0)
        lengths = np.linalg.norm(np.asarray(ends, dtype=float) - np.asarray(starts, dtype=float), axis=1)[robot_indices]
        offsets = np.asarray(self.proxies.proxy_offset_radii, dtype=float)[obstacle_indices]
        start_clearance = d0 - pair_radii - offsets
        end_clearance = d1 - pair_radii - offsets
        clearance = 0.5 * (d0 + d1 - lengths) - pair_radii - offsets
        colliding = clearance <= 0.0
        recovery = colliding & (start_clearance <= 0.0) & (end_clearance > start_clearance + 1e-10)
        worsening = colliding & ~recovery
        return ProxySweepCertificate(safe=not bool(np.any(worsening)), colliding_pairs=int(np.count_nonzero(colliding)), recovery_pairs=int(np.count_nonzero(recovery)), worsening_pairs=int(np.count_nonzero(worsening)), candidate_pairs=int(len(obstacle_indices)), minimum_clearance_m=float(np.min(clearance)))

    def _certify_exact_pairs_batched(self, starts: np.ndarray, ends: np.ndarray, radii: np.ndarray) -> ProxySweepCertificate:
        robot_batches: list[np.ndarray] = []
        obstacle_batches: list[np.ndarray] = []
        for robot_index, (start, end, radius) in enumerate(zip(starts, ends, radii)):
            candidates = self._candidates(start, end, float(radius))
            if len(candidates):
                robot_batches.append(np.full(len(candidates), robot_index, dtype=np.int64))
                obstacle_batches.append(np.asarray(candidates, dtype=np.int64))
        if not obstacle_batches:
            return ProxySweepCertificate(safe=True, colliding_pairs=0, recovery_pairs=0, worsening_pairs=0, candidate_pairs=0, minimum_clearance_m=float('inf'))
        robot_indices = np.concatenate(robot_batches)
        obstacle_indices = np.concatenate(obstacle_batches)
        proxy_ids = np.asarray(self.proxies.proxy_ids, dtype=np.int64)[obstacle_indices]
        initial = np.full(len(obstacle_indices), np.nan, dtype=float)
        for pair_index, (robot_index, proxy_id) in enumerate(zip(robot_indices, proxy_ids)):
            previous = self.multiplier_cache.get((int(robot_index), int(proxy_id)))
            if previous is not None:
                initial[pair_index] = previous
        centers = np.asarray(self.proxies.centers, dtype=float)
        pair_starts = np.asarray(starts, dtype=float)[robot_indices]
        pair_ends = np.asarray(ends, dtype=float)[robot_indices]
        first = self.support_kernel.closest_points_pairs_warm(pair_starts, centers[obstacle_indices], self.eigenvalues[obstacle_indices], self.rotations[obstacle_indices], initial)
        second = self.support_kernel.closest_points_pairs_warm(pair_ends, centers[obstacle_indices], self.eigenvalues[obstacle_indices], self.rotations[obstacle_indices], first[2])
        for robot_index, proxy_id, multiplier in zip(robot_indices, proxy_ids, second[2]):
            self.multiplier_cache[int(robot_index), int(proxy_id)] = float(multiplier)
        d0 = np.linalg.norm(first[1] - pair_starts, axis=1)
        d1 = np.linalg.norm(second[1] - pair_ends, axis=1)
        pair_radii = np.asarray(radii, dtype=float)[robot_indices]
        offsets = np.asarray(self.proxies.proxy_offset_radii, dtype=float)[obstacle_indices]
        lengths = np.linalg.norm(np.asarray(ends, dtype=float) - np.asarray(starts, dtype=float), axis=1)[robot_indices]
        start_clearance = d0 - pair_radii - offsets
        end_clearance = d1 - pair_radii - offsets
        clearance = 0.5 * (d0 + d1 - lengths) - pair_radii - offsets
        colliding = clearance <= 0.0
        recovery = colliding & (start_clearance <= 0.0) & (end_clearance > start_clearance + 1e-10)
        worsening = colliding & ~recovery
        return ProxySweepCertificate(safe=not bool(np.any(worsening)), colliding_pairs=int(np.count_nonzero(colliding)), recovery_pairs=int(np.count_nonzero(recovery)), worsening_pairs=int(np.count_nonzero(worsening)), candidate_pairs=int(len(obstacle_indices)), minimum_clearance_m=float(np.min(clearance)))

    def certify(self, starts: np.ndarray, ends: np.ndarray, radii: np.ndarray) -> ProxySweepCertificate:
        if self.representation == 'ellipsoid' and self.directional_uncertainty_shapes is not None:
            return self._certify_directional_pairs_batched(starts, ends, radii)
        if self.representation == 'ellipsoid':
            return self._certify_exact_pairs_batched(starts, ends, radii)
        colliding = 0
        recovery = 0
        worsening = 0
        candidate_pairs = 0
        minimum = float('inf')
        centers = np.asarray(self.proxies.centers, dtype=float)
        offsets = np.asarray(self.proxies.proxy_offset_radii, dtype=float)
        for robot_index, (start, end, radius) in enumerate(zip(starts, ends, radii)):
            delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
            length = float(np.linalg.norm(delta))
            candidates = self._candidates(start, end, float(radius))
            candidate_pairs += int(len(candidates))
            if self.representation == 'sphere':
                obstacle_radii = np.asarray(self.proxies.sphere_radii, dtype=float)
                denominator = float(delta @ delta)
                for obstacle_index in candidates:
                    obstacle = centers[obstacle_index]
                    if denominator <= 1e-24:
                        closest = start
                    else:
                        parameter = float(np.clip((obstacle - start) @ delta / denominator, 0.0, 1.0))
                        closest = start + parameter * delta
                    clearance = float(np.linalg.norm(closest - obstacle) - radius - obstacle_radii[obstacle_index] - offsets[obstacle_index])
                    minimum = min(minimum, clearance)
                    start_clearance = float(np.linalg.norm(start - obstacle) - radius - obstacle_radii[obstacle_index] - offsets[obstacle_index])
                    end_clearance = float(np.linalg.norm(end - obstacle) - radius - obstacle_radii[obstacle_index] - offsets[obstacle_index])
                    if clearance <= 0.0:
                        colliding += 1
                        if start_clearance <= 0.0 and end_clearance > start_clearance + 1e-10:
                            recovery += 1
                        else:
                            worsening += 1
            else:
                shapes = np.asarray(self.proxies.base_ellipsoid_shapes, dtype=float)
                uncertainty_shapes = self.directional_uncertainty_shapes
                if uncertainty_shapes is None:
                    for obstacle_index in candidates:
                        first = closest_point_on_ellipsoid(centers[obstacle_index], shapes[obstacle_index], start, eigenvalues=self.eigenvalues[obstacle_index], rotation=self.rotations[obstacle_index])
                        second = closest_point_on_ellipsoid(centers[obstacle_index], shapes[obstacle_index], end, initial_multiplier=first.multiplier, eigenvalues=self.eigenvalues[obstacle_index], rotation=self.rotations[obstacle_index])
                        d0 = float(np.linalg.norm(first.surface_point - start))
                        d1 = float(np.linalg.norm(second.surface_point - end))
                        lower_distance = 0.5 * (d0 + d1 - length)
                        clearance = float(lower_distance - radius - offsets[obstacle_index])
                        minimum = min(minimum, clearance)
                        start_clearance = d0 - radius - offsets[obstacle_index]
                        end_clearance = d1 - radius - offsets[obstacle_index]
                        if clearance <= 0.0:
                            colliding += 1
                            if start_clearance <= 0.0 and end_clearance > start_clearance + 1e-10:
                                recovery += 1
                            else:
                                worsening += 1
                elif len(candidates):
                    candidate_ids = np.asarray(self.proxies.proxy_ids, dtype=np.int64)[candidates]
                    initial = np.zeros((len(candidates), 3), dtype=float)
                    for local_index, proxy_id in enumerate(candidate_ids):
                        previous = self.normal_cache.get((int(robot_index), int(proxy_id)))
                        if previous is not None:
                            initial[local_index] = previous
                    first_normals, _, _ = self.support_kernel.normals_sum_warm(start, np.eye(3) * float(radius) ** 2, centers[candidates], shapes[candidates], uncertainty_shapes[candidates], initial, max_iterations=64)
                    second_normals, _, _ = self.support_kernel.normals_sum_warm(end, np.eye(3) * float(radius) ** 2, centers[candidates], shapes[candidates], uncertainty_shapes[candidates], first_normals, max_iterations=64)
                    for local_index, proxy_id in enumerate(candidate_ids):
                        self.normal_cache[int(robot_index), int(proxy_id)] = second_normals[local_index].copy()
                    delta0 = centers[candidates] - np.asarray(start, dtype=float)
                    delta1 = centers[candidates] - np.asarray(end, dtype=float)
                    q0 = np.sqrt(np.maximum(np.einsum('ni,nij,nj->n', first_normals, shapes[candidates], first_normals), 0.0))
                    u0 = np.sqrt(np.maximum(np.einsum('ni,nij,nj->n', first_normals, uncertainty_shapes[candidates], first_normals), 0.0))
                    q1 = np.sqrt(np.maximum(np.einsum('ni,nij,nj->n', second_normals, shapes[candidates], second_normals), 0.0))
                    u1 = np.sqrt(np.maximum(np.einsum('ni,nij,nj->n', second_normals, uncertainty_shapes[candidates], second_normals), 0.0))
                    d0_values = np.maximum(np.einsum('ni,ni->n', first_normals, delta0) - q0 - u0, 0.0)
                    d1_values = np.maximum(np.einsum('ni,ni->n', second_normals, delta1) - q1 - u1, 0.0)
                    for local_index, obstacle_index in enumerate(candidates):
                        d0 = float(d0_values[local_index])
                        d1 = float(d1_values[local_index])
                        lower_distance = 0.5 * (d0 + d1 - length)
                        clearance = float(lower_distance - radius - offsets[obstacle_index])
                        minimum = min(minimum, clearance)
                        start_clearance = d0 - radius - offsets[obstacle_index]
                        end_clearance = d1 - radius - offsets[obstacle_index]
                        if clearance <= 0.0:
                            colliding += 1
                            if start_clearance <= 0.0 and end_clearance > start_clearance + 1e-10:
                                recovery += 1
                            else:
                                worsening += 1
        return ProxySweepCertificate(safe=worsening == 0, colliding_pairs=colliding, recovery_pairs=recovery, worsening_pairs=worsening, candidate_pairs=candidate_pairs, minimum_clearance_m=minimum)

def _prepare_publication(packet: PerceptionPacket | SerializedPerceptionPacket, representation: str, index_mode: str, robot_radii: np.ndarray, centervox_size: float) -> tuple[PerceptionPacket, ContinuousProxyGuard, ObservedPointSnapshot]:
    materialized = _materialize_packet(packet, representation, index_mode, robot_radii)
    try:
        guard = ContinuousProxyGuard(materialized, representation)
        observed = snapshot_from_proxies(materialized.proxies, publish_cycle=-1, source_cycle=materialized.source_cycle, centervox_size=centervox_size)
    except Exception:
        materialized.close()
        raise
    return (materialized, guard, observed)

def run_one(representation: str, index_mode: str, output_root: Path=DEFAULT_OUTPUT, *, maximum_cycles: int | None=None, unknown_policy: str='observed_only', realtime_pacing: bool=False, perception_executor: str='process', centervox_size: float=CENTERVOX_SIZE, maximum_uncertainty_union_inflation: float | None=None, certificate_radius_limit: float | None=None, uncertainty_fusion_mode: str='separate_uncertainty', direct_thin_axis_inflation: float=np.sqrt(2.0), direct_tangent_subdivisions: int=1, direct_partition_mode: str='grid', camera_width: int=CAMERA_WIDTH, camera_height: int=CAMERA_HEIGHT, camera_pixel_stride: int=CAMERA_PIXEL_STRIDE, camera_names: tuple[str, ...]=CAMERA_NAMES, scene_version: str='formal', record_dense_map_snapshots: bool=False, sweep_guard_mode: str='enforce', success_tolerance: float=SUCCESS_TOLERANCE, success_hold_cycles: int=SUCCESS_HOLD_CYCLES, stop_on_success: bool=True, perception_cpu_affinity: tuple[int, ...] | None=None, control_cpu_affinity: tuple[int, ...] | None=None, ellipsoid_pair_threads: int=8, control_compute_cpu_affinity: tuple[int, ...] | None=None, publication_cpu_affinity: tuple[int, ...] | None=None, control_process_priority: str='normal', perception_process_priority: str='normal', task_gain: float=2.0, max_task_speed: float=0.18, osqp_adaptive_row_threshold: int=512, osqp_tolerance: float=2e-05, osqp_max_iterations: int=6000) -> dict:
    if representation not in {'sphere', 'ellipsoid'}:
        raise ValueError('representation must be sphere or ellipsoid')
    if index_mode not in {'full_scan', 'mvt_scalar', 'mvt_simd'}:
        raise ValueError('unknown index mode')
    if unknown_policy not in {'monitor', 'observed_only', 'strict'}:
        raise ValueError('unknown_policy must be observed_only, strict, or legacy monitor')
    if unknown_policy == 'monitor':
        unknown_policy = 'observed_only'
    if perception_executor not in {'process', 'thread'}:
        raise ValueError('perception_executor must be process or thread')
    if perception_cpu_affinity is not None and perception_executor != 'process':
        raise ValueError('perception CPU affinity requires the process executor')
    if sweep_guard_mode not in {'enforce', 'audit_only', 'post_control_audit'}:
        raise ValueError('sweep_guard_mode must be enforce, audit_only, or post_control_audit')
    if centervox_size <= 0.0:
        raise ValueError('centervox_size must be positive')
    if camera_width <= 0 or camera_height <= 0 or camera_pixel_stride <= 0:
        raise ValueError('camera dimensions and pixel stride must be positive')
    if maximum_uncertainty_union_inflation is not None and maximum_uncertainty_union_inflation < 1.0:
        raise ValueError('maximum uncertainty union inflation must be >= 1')
    if certificate_radius_limit is not None and certificate_radius_limit <= 0.0:
        raise ValueError('certificate radius limit must be positive')
    if uncertainty_fusion_mode not in {'separate_uncertainty', 'fused_certified_ellipsoid', 'fused_certified_centervox'}:
        raise ValueError('unknown uncertainty_fusion_mode')
    if direct_tangent_subdivisions < 1:
        raise ValueError('direct tangent subdivisions must be positive')
    if direct_partition_mode not in {'grid', 'longest_tangent_binary'}:
        raise ValueError('unknown direct partition mode')
    if uncertainty_fusion_mode == 'fused_certified_centervox':
        if direct_thin_axis_inflation <= 1.0:
            raise ValueError('direct thin-axis inflation must exceed one')
    elif min(abs(direct_thin_axis_inflation - 1.0), abs(direct_thin_axis_inflation - np.sqrt(2.0))) > 1e-12 or direct_tangent_subdivisions != 1 or direct_partition_mode != 'grid':
        raise ValueError('inactive direct thin-axis controls must use identity 1.0 or the legacy default sqrt(2)')
    if success_tolerance <= 0.0:
        raise ValueError('success tolerance must be positive')
    if success_hold_cycles <= 0:
        raise ValueError('success hold cycles must be positive')
    if ellipsoid_pair_threads < 1 or ellipsoid_pair_threads > 64:
        raise ValueError('ellipsoid pair threads must be in [1, 64]')
    if control_process_priority not in {'normal', 'above_normal'}:
        raise ValueError('unsupported control process priority')
    if perception_process_priority != 'normal':
        raise ValueError('the formal perception worker must remain normal priority')
    if task_gain <= 0.0:
        raise ValueError('task gain must be positive')
    if max_task_speed <= 0.0:
        raise ValueError('maximum task speed must be positive')
    if osqp_adaptive_row_threshold < 1:
        raise ValueError('OSQP adaptive row threshold must be positive')
    if osqp_tolerance <= 0.0:
        raise ValueError('OSQP tolerance must be positive')
    if osqp_max_iterations < 1:
        raise ValueError('OSQP maximum iterations must be positive')
    if control_cpu_affinity is not None:
        allowed = set(control_cpu_affinity)
        for name, affinity in (('control compute', control_compute_cpu_affinity), ('publication', publication_cpu_affinity)):
            if affinity is not None and (not set(affinity).issubset(allowed)):
                raise ValueError(f'{name} affinity must be inside control process')
    if control_compute_cpu_affinity is not None and publication_cpu_affinity is not None and set(control_compute_cpu_affinity) & set(publication_cpu_affinity):
        raise ValueError('control compute and publication affinities must be disjoint')
    scene = _protocol_scene(scene_version)
    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(scene.q0, dtype=float))
    robot = build_robot_certificate(model)
    robot_radii = np.asarray([sphere.radius for sphere in robot], dtype=float)
    target = np.asarray(scene.waypoints[-1], dtype=float)
    formal_cycles = int(round(scene.duration / DT))
    cycle_limit = formal_cycles if maximum_cycles is None else int(maximum_cycles)
    if cycle_limit <= 0:
        raise ValueError('maximum_cycles must be positive')
    suffix = '-strict_unknown' if unknown_policy == 'strict' else '-observed_only'
    pace_suffix = '-paced' if realtime_pacing else ''
    map_suffix = '-maplog' if record_dense_map_snapshots else ''
    sweep_suffix = {'enforce': '', 'audit_only': '-sweepaudit', 'post_control_audit': '-postsweepaudit'}[sweep_guard_mode]
    cv_suffix = f'-cv{centervox_size * 1000.0:g}mm'
    union_suffix = '' if maximum_uncertainty_union_inflation is None else f'-ui{maximum_uncertainty_union_inflation:g}'
    radius_suffix = '' if certificate_radius_limit is None else f'-cr{certificate_radius_limit * 1000.0:g}mm'
    fusion_suffix = {'separate_uncertainty': '', 'fused_certified_ellipsoid': '-fusedQ', 'fused_certified_centervox': '-fusedCV'}[uncertainty_fusion_mode]
    direct_suffix = '' if uncertainty_fusion_mode != 'fused_certified_centervox' or (abs(direct_thin_axis_inflation - np.sqrt(2.0)) <= 1e-12 and direct_tangent_subdivisions == 1 and (direct_partition_mode == 'grid')) else f'-thin{direct_thin_axis_inflation:g}' + ('-splitlong2' if direct_partition_mode == 'longest_tangent_binary' else f'-sub{direct_tangent_subdivisions}x{direct_tangent_subdivisions}')
    camera_suffix = f'-cam{camera_width}x{camera_height}s{camera_pixel_stride}-r{CAMERA_MINIMUM_RANGE * 1000.0:g}mm'
    view_suffix = {('ur5_depth_wrist',): '-wrist', ('ur5_depth_wrist', 'ur5_depth_wrist_right'): '-wrist_pair', CAMERA_NAMES_WRIST_SHOULDER: '-wrist_shoulder', CAMERA_NAMES: '-wrist_forearm', CAMERA_NAMES_DUAL_SHOULDER: '-dual_shoulder'}.get(camera_names, '-multi')
    success_suffix = '' if abs(success_tolerance - SUCCESS_TOLERANCE) <= 1e-15 and success_hold_cycles == SUCCESS_HOLD_CYCLES and stop_on_success else f"-tol{success_tolerance * 1000.0:g}mm-hold{success_hold_cycles}-{('stop' if stop_on_success else 'fixed')}"
    run_name = f'A-CV-{scene_version}-{index_mode}-{representation}{cv_suffix}{union_suffix}{radius_suffix}{fusion_suffix}{direct_suffix}{camera_suffix}{view_suffix}{suffix}{pace_suffix}{map_suffix}{sweep_suffix}{success_suffix}-omp{ellipsoid_pair_threads}-prio{control_process_priority}-posqp'
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'scene.xml').write_bytes(build_xml(scene).encode('utf-8'))
    worker = None
    publish_dense_map = unknown_policy == 'strict'
    publish_map_delta = record_dense_map_snapshots and (not publish_dense_map)
    if perception_executor == 'process':
        executor = ProcessPoolExecutor(max_workers=1, initializer=_initialize_process_worker, initargs=(representation, index_mode, publish_dense_map, publish_map_delta, centervox_size, maximum_uncertainty_union_inflation, certificate_radius_limit, uncertainty_fusion_mode, direct_thin_axis_inflation, direct_tangent_subdivisions, direct_partition_mode, camera_width, camera_height, camera_pixel_stride, camera_names, scene_version, perception_cpu_affinity))

        def submit_raw(request):
            return executor.submit(_process_perception_request, *request)
    else:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='local-perception')
        worker = executor.submit(PerceptionWorker, representation, index_mode, centervox_size, maximum_uncertainty_union_inflation, certificate_radius_limit, uncertainty_fusion_mode, direct_thin_axis_inflation, direct_tangent_subdivisions, direct_partition_mode, camera_width, camera_height, camera_pixel_stride, camera_names, scene_version, publish_map_delta).result()

        def submit_raw(request):
            return executor.submit(worker.process, *request, False, publish_dense_map, publish_map_delta)
    initial_q = data.qpos[:len(JOINT_NAMES)].copy()
    initial_raw = submit_raw((initial_q, 0, 0, 0.0)).result()
    initial_packet = _materialize_packet(initial_raw, representation, index_mode, robot_radii)
    active_packet = initial_packet
    controller = _new_controller(representation, model, data, scene, robot, active_packet.proxies, active_packet.mvt, ellipsoid_pair_threads=ellipsoid_pair_threads, ellipsoid_pair_affinity_mask=sum((1 << int(cpu) for cpu in control_compute_cpu_affinity or ())))
    controller.task_gain = float(task_gain)
    controller.max_task_speed = float(max_task_speed)
    controller.osqp_adaptive_row_threshold = int(osqp_adaptive_row_threshold)
    controller.osqp_absolute_tolerance = float(osqp_tolerance)
    controller.osqp_relative_tolerance = float(osqp_tolerance)
    controller.osqp_max_iterations = int(osqp_max_iterations)
    proxy_guard = ContinuousProxyGuard(active_packet, representation)
    frames: list[dict] = []
    cycles: list[dict] = []
    q_history: list[np.ndarray] = []
    ee_history: list[np.ndarray] = []
    observability_snapshots = [snapshot_from_proxies(active_packet.proxies, publish_cycle=0, source_cycle=active_packet.source_cycle, centervox_size=centervox_size)]
    proxy_snapshots = [(0, active_packet.source_cycle, active_packet.proxies)]
    source_configurations = [(0, active_packet.source_cycle, active_packet.source_q.copy())]
    occupancy_snapshots: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    if record_dense_map_snapshots and active_packet.occupancy_keys is not None:
        occupancy_snapshots.append((0, active_packet.source_cycle, active_packet.occupancy_keys, active_packet.occupancy_states))
    active_cycle_records: list[tuple[int, int, tuple]] = []
    initial_row = dict(active_packet.frame_row)
    initial_row.update(publish_cycle=0, publish_time_s=0.0, publish_snapshot_age_cycles=0, publish_snapshot_age_ms=0.0)
    frames.append(initial_row)
    publication_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='snapshot-publication', initializer=_initialize_thread_affinity, initargs=(publication_cpu_affinity,))
    pending_future: Future | None = None
    preparing_future: Future | None = None
    queued_request: tuple[np.ndarray, int, int, float] | None = None
    last_requested_tick = 0
    requested_frames = 1
    submitted_jobs = 1
    replaced_requests = 0
    published_frames = 1
    success_hold = 0
    maximum_success_hold = 0
    current_success_start_cycle = None
    first_sustained_success_start_cycle = None
    first_sustained_success_completion_cycle = None
    maximum_penetration = 0.0
    deadline_misses = 0
    sweep_audit_failures = 0
    sweep_modified_cycles = 0
    run_started = time.perf_counter()
    previous_control_thread_affinity = _set_current_thread_affinity(control_compute_cpu_affinity)
    cyclic_gc_was_enabled = gc.isenabled()
    if cyclic_gc_was_enabled:
        gc.disable()

    def submit_request(request: tuple[np.ndarray, int, int, float]) -> Future:
        nonlocal submitted_jobs
        submitted_jobs += 1
        return submit_raw(request)
    try:
        for cycle in range(cycle_limit):
            cycle_started = time.perf_counter()
            perception_published = False
            if preparing_future is not None and preparing_future.done():
                next_packet, next_guard, prepared_observed = preparing_future.result()
                preparing_future = None
                _update_controller(controller, next_packet, representation, next_guard)
                previous_packet = active_packet
                active_packet = next_packet
                proxy_guard = next_guard
                previous_packet.close()
                published_frames += 1
                perception_published = True
                row = dict(next_packet.frame_row)
                row.update(publish_cycle=cycle, publish_time_s=cycle * DT, publish_snapshot_age_cycles=cycle - next_packet.source_cycle, publish_snapshot_age_ms=(cycle - next_packet.source_cycle) * DT * 1000.0)
                frames.append(row)
                observability_snapshots.append(ObservedPointSnapshot(publish_cycle=cycle, source_cycle=next_packet.source_cycle, points=prepared_observed.points, cover_radii=prepared_observed.cover_radii))
                proxy_snapshots.append((cycle, next_packet.source_cycle, next_packet.proxies))
                source_configurations.append((cycle, next_packet.source_cycle, next_packet.source_q.copy()))
                if record_dense_map_snapshots and next_packet.occupancy_keys is not None:
                    occupancy_snapshots.append((cycle, next_packet.source_cycle, next_packet.occupancy_keys, next_packet.occupancy_states))
            if pending_future is not None and pending_future.done() and (preparing_future is None):
                raw_packet = pending_future.result()
                pending_future = None
                preparing_future = publication_executor.submit(_prepare_publication, raw_packet, representation, index_mode, robot_radii, centervox_size)
                if queued_request is not None:
                    pending_future = submit_request(queued_request)
                    queued_request = None
            q_before = data.qpos[:len(JOINT_NAMES)].copy()
            robot_before = certificate_world_positions(data, robot)
            camera_tick = int(np.floor(cycle * DT * CAMERA_HZ + 1e-12))
            camera_requested = camera_tick != last_requested_tick
            if camera_requested:
                requested_frames += 1
                request = (q_before.copy(), requested_frames - 1, cycle, cycle * DT)
                if pending_future is None:
                    pending_future = submit_request(request)
                else:
                    if queued_request is not None:
                        replaced_requests += 1
                    queued_request = request
                last_requested_tick = camera_tick
            qdot_proposed, metrics = controller.solve(target)
            active_cycle_records.append((cycle, active_packet.generation, controller.last_pair_records))
            sweep_scale = 1.0
            sweep_trials = 0 if sweep_guard_mode == 'post_control_audit' else 1

            def certify_scale(scale: float):
                proposed_q = q_before + scale * qdot_proposed * DT
                set_configuration(model, data, proposed_q)
                robot_after = certificate_world_positions(data, robot)
                set_configuration(model, data, q_before)
                proxy_sweep = None
                dense_sweep = None
                if unknown_policy == 'strict':
                    if active_packet.occupancy is None:
                        raise AssertionError('strict mode requires a dense map snapshot')
                    dense_sweep = active_packet.occupancy.certify_swept_spheres(robot_before, robot_after, robot_radii, margin=0.0)
                    blocked = dense_sweep.occupied_voxels > 0 or dense_sweep.unknown_voxels > 0
                else:
                    proxy_sweep = proxy_guard.certify(robot_before, robot_after, robot_radii)
                    blocked = not proxy_sweep.safe
                return (blocked, proxy_sweep, dense_sweep)
            if sweep_guard_mode == 'post_control_audit':
                blocked = False
                proxy_sweep = None
                dense_sweep = None
                guard_safe = True
                qdot_executed = qdot_proposed.copy()
            else:
                blocked, proxy_sweep, dense_sweep = certify_scale(sweep_scale)
            if sweep_guard_mode == 'enforce':
                while blocked and sweep_scale > 1.0 / 128.0:
                    sweep_scale *= 0.5
                    sweep_trials += 1
                    blocked, proxy_sweep, dense_sweep = certify_scale(sweep_scale)
                guard_safe = not blocked
                qdot_executed = sweep_scale * qdot_proposed if guard_safe else np.zeros_like(qdot_proposed)
                if sweep_scale < 1.0 or not guard_safe:
                    sweep_modified_cycles += 1
            elif sweep_guard_mode == 'audit_only':
                guard_safe = not blocked
                qdot_executed = qdot_proposed.copy()
                if blocked:
                    sweep_audit_failures += 1
            q_after = q_before + qdot_executed * DT
            set_configuration(model, data, q_after)
            ee = attachment_position(model, data)
            error = float(np.linalg.norm(target - ee))
            penetrating = [index for index in range(data.ncon) if float(data.contact[index].dist) < -1e-08]
            minimum_contact_distance = min((float(data.contact[index].dist) for index in range(data.ncon))) if data.ncon else None
            if minimum_contact_distance is not None:
                maximum_penetration = min(maximum_penetration, minimum_contact_distance)
            if error < success_tolerance and (not penetrating):
                if success_hold == 0:
                    current_success_start_cycle = cycle
                success_hold += 1
                maximum_success_hold = max(maximum_success_hold, success_hold)
                if success_hold >= success_hold_cycles and first_sustained_success_start_cycle is None:
                    first_sustained_success_start_cycle = current_success_start_cycle
                    first_sustained_success_completion_cycle = cycle
            else:
                success_hold = 0
                current_success_start_cycle = None
            compute_ms = (time.perf_counter() - cycle_started) * 1000.0
            deadline = run_started + (cycle + 1) * DT
            deadline_lateness_ms = max(0.0, (time.perf_counter() - deadline) * 1000.0)
            if deadline_lateness_ms > 0.0:
                deadline_misses += 1
            if realtime_pacing:
                remaining = deadline - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
            wall_cycle_ms = (time.perf_counter() - cycle_started) * 1000.0
            cycles.append({'cycle': cycle, 'time_s': cycle * DT, 'representation': representation, 'index_mode': index_mode, 'camera_requested': camera_requested, 'perception_published': perception_published, 'active_generation': active_packet.generation, 'snapshot_source_cycle': active_packet.source_cycle, 'snapshot_age_cycles': cycle - active_packet.source_cycle, 'snapshot_age_ms': (cycle - active_packet.source_cycle) * DT * 1000.0, 'status': metrics.status, 'ee_x': float(ee[0]), 'ee_y': float(ee[1]), 'ee_z': float(ee[2]), 'error_m': error, 'proposed_qdot_norm': float(np.linalg.norm(qdot_proposed)), 'executed_qdot_norm': float(np.linalg.norm(qdot_executed)), 'causal_sweep_scale': sweep_scale if guard_safe else 0.0, 'causal_sweep_trials': sweep_trials, 'occupancy_guard_safe': guard_safe, 'occupied_guard_source': 'dense_three_state_map' if unknown_policy == 'strict' else 'centervox_covered_proxies', 'unknown_policy': unknown_policy, 'sweep_occupied_voxels': -1 if dense_sweep is None else dense_sweep.occupied_voxels, 'sweep_unknown_voxels': -1 if dense_sweep is None else dense_sweep.unknown_voxels, 'sweep_checked_voxels': -1 if dense_sweep is None else dense_sweep.checked_voxels, 'proxy_sweep_candidate_pairs': -1 if proxy_sweep is None else proxy_sweep.candidate_pairs, 'proxy_sweep_colliding_pairs': -1 if proxy_sweep is None else proxy_sweep.colliding_pairs, 'proxy_sweep_recovery_pairs': -1 if proxy_sweep is None else proxy_sweep.recovery_pairs, 'proxy_sweep_worsening_pairs': -1 if proxy_sweep is None else proxy_sweep.worsening_pairs, 'proxy_sweep_minimum_clearance_m': None if proxy_sweep is None or not np.isfinite(proxy_sweep.minimum_clearance_m) else proxy_sweep.minimum_clearance_m, 'exact_contact_count': int(data.ncon), 'exact_penetrating_contact_count': len(penetrating), 'minimum_contact_distance_m': minimum_contact_distance, 'control_compute_ms': compute_ms, 'wall_cycle_ms': wall_cycle_ms, 'deadline_lateness_ms': deadline_lateness_ms, **metrics.as_dict()})
            q_history.append(q_after.copy())
            ee_history.append(ee.copy())
            if stop_on_success and success_hold >= success_hold_cycles:
                break
    finally:
        _restore_current_thread_affinity(previous_control_thread_affinity)
        if cyclic_gc_was_enabled:
            gc.enable()
        unfinished_packets: list[PerceptionPacket] = []
        if preparing_future is not None:
            try:
                packet, _, _ = preparing_future.result()
                unfinished_packets.append(packet)
            except Exception:
                pass
        if pending_future is not None:
            try:
                raw_packet = pending_future.result()
                packet, _, _ = _prepare_publication(raw_packet, representation, index_mode, robot_radii, centervox_size)
                unfinished_packets.append(packet)
            except Exception:
                pass
        for packet in unfinished_packets:
            if packet is not active_packet:
                packet.close()
        publication_executor.shutdown(wait=True)
        if worker is not None:
            executor.submit(worker.close).result()
        executor.shutdown(wait=True)
    elapsed = time.perf_counter() - run_started
    final_proxies = active_packet.proxies
    q_trajectory = np.vstack((initial_q, np.asarray(q_history)))
    np.save(run_dir / 'q_history.npy', np.asarray(q_history))
    np.save(run_dir / 'ee_history.npy', np.asarray(ee_history))
    _save_observability_snapshots(run_dir / 'causal_observability_snapshots.npz', observability_snapshots)
    _save_proxy_snapshots(run_dir / 'causal_proxy_snapshots.npz', proxy_snapshots)
    _save_source_configurations(run_dir / 'causal_source_configurations.npz', source_configurations)
    if occupancy_snapshots:
        _save_occupancy_snapshots(run_dir / 'causal_occupancy_snapshots.npz', occupancy_snapshots)
    active_rows = [{'cycle': cycle, 'time_s': cycle * DT, 'active_generation': generation, **record._asdict()} for cycle, generation, records in active_cycle_records for record in records]
    _write_csv(run_dir / 'pair_states.csv', active_rows)
    observability_report = evaluate_observed_before_risk(scene, q_trajectory, observability_snapshots, observation_distance=OBSERVABILITY_DISTANCE, guard_time=OBSERVABILITY_GUARD_TIME, truth_surface_spacing=OBSERVABILITY_TRUTH_SPACING)
    (run_dir / 'observability_report.json').write_text(json.dumps(observability_report.as_dict(), indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(run_dir / 'final_causal_proxies.npz', proxy_ids=final_proxies.proxy_ids, centers=final_proxies.centers, sphere_radii=final_proxies.sphere_radii, base_sphere_radii=final_proxies.base_sphere_radii, ellipsoid_shapes=final_proxies.base_ellipsoid_shapes, ellipsoid_outer_shapes=final_proxies.ellipsoid_outer_shapes, proxy_uncertainty_shapes=final_proxies.proxy_uncertainty_shapes, uncertainty_offsets=final_proxies.proxy_offset_radii, filtered_points=final_proxies.filtered_points, filtered_cluster_indices=final_proxies.filtered_cluster_indices, filtered_uncertainty_shapes=final_proxies.filtered_uncertainty_shapes)
    active_packet.close()
    _write_csv(run_dir / 'cycles.csv', cycles)
    _write_csv(run_dir / 'perception_frames.csv', frames)
    errors = np.asarray([row['error_m'] for row in cycles])
    ee_array = np.asarray(ee_history)
    compute_ms = np.asarray([row['control_compute_ms'] for row in cycles])
    controller_ms = np.asarray([row['total_controller_ms'] for row in cycles])
    support_iterations = np.asarray([row['support_normal_iterations'] for row in cycles], dtype=float)
    support_residuals = np.asarray([row['support_normal_max_residual'] for row in cycles], dtype=float)
    qp_iterations = np.asarray([row['qp_iterations'] for row in cycles], dtype=float)
    qp_primal_residuals = np.asarray([row['qp_primal_residual'] for row in cycles], dtype=float)
    qp_dual_residuals = np.asarray([row['qp_dual_residual'] for row in cycles], dtype=float)
    frame_ms = np.asarray([row['frame_pipeline_ms'] for row in frames])
    snapshot_ages = np.asarray([row['snapshot_age_ms'] for row in cycles])
    task_success = success_hold >= success_hold_cycles
    ever_sustained_success = first_sustained_success_start_cycle is not None
    steady_window_cycles = min(len(errors), max(success_hold_cycles, int(round(1.0 / DT))))
    steady_errors = errors[-steady_window_cycles:]
    realtime_controller_threshold_ms = 1000.0 * DT
    controller_p99_ms = float(np.percentile(controller_ms, 99))
    realtime_controller_passed = bool(realtime_pacing and controller_p99_ms <= realtime_controller_threshold_ms)
    all_centervox_coverage_checks_passed = all((row['coverage_ok'] for row in frames))
    all_mvt_oracle_checks_passed = all((row['mvt_missing_candidates'] == 0 for row in frames))
    all_proxy_radius_limit_checks_passed = all((row['certificate_radius_limit_ok'] for row in frames))
    no_exact_penetration = bool(maximum_penetration >= -1e-12 and all((row['exact_penetrating_contact_count'] == 0 for row in cycles)))
    evidence_eligible = bool(task_success and observability_report.passed and all_centervox_coverage_checks_passed and all_mvt_oracle_checks_passed and all_proxy_radius_limit_checks_passed and no_exact_penetration and (sweep_audit_failures == 0) and (sweep_modified_cycles == 0) and (sweep_guard_mode == 'audit_only') and realtime_controller_passed)
    summary = {'experiment': f'protocol_{scene_version}_async_causal_local_perception_liuqp', 'scene_name': scene.name, 'scene_version': scene_version, 'run_name': run_name, 'representation': representation, 'state_clearance_definition': 'raw_certified_surface_gap_before_hard_safety_margin', 'hard_barrier_clearance_definition': 'surface_clearance_minus_safety_margin', 'safety_margin_m': float(controller.safety_margin), 'near_surface_gap_threshold_m': float(controller.near_distance), 'contact_surface_gap_threshold_m': float(controller.contact_distance), 'index_mode': index_mode, 'mvt_multilevel_enabled': index_mode != 'full_scan', 'mvt_simd_enabled': index_mode == 'mvt_simd', 'mvt_simd_width_float_lanes': 8 if index_mode == 'mvt_simd' else 1, 'mvt_exact_narrowphase_fused_native': bool(representation in {'sphere', 'ellipsoid'} and index_mode == 'mvt_simd' and (uncertainty_fusion_mode == 'fused_certified_centervox')), 'native_sphere_mvt_prune_fused': bool(representation == 'sphere' and index_mode == 'mvt_simd'), 'sphere_narrowphase_solver': 'mvt27_avx2_fused_analytic_closest_openmp_ordered_erase_remove' if representation == 'sphere' else None, 'sphere_pair_threads': controller.ellipsoid_pair_threads if representation == 'sphere' else 0, 'ellipsoid_support_solver': 'mvt27_avx2_fused_safeguarded_multiplier_newton_bisection_warm_openmp8_lazy_ordered_erase_remove' if representation == 'ellipsoid' else None, 'ellipsoid_support_pair_threads': controller.ellipsoid_pair_threads if representation == 'ellipsoid' else 0, 'ellipsoid_pair_affinity_mask': controller.ellipsoid_pair_affinity_mask, 'ellipsoid_worker_pin_mode': 'shared_explicit_compute_mask' if controller.ellipsoid_pair_affinity_mask else 'process_scheduler', 'control_compute_cpu_affinity': None if control_compute_cpu_affinity is None else list(control_compute_cpu_affinity), 'snapshot_publication_cpu_affinity': None if publication_cpu_affinity is None else list(publication_cpu_affinity), 'control_process_priority': control_process_priority, 'perception_process_priority': perception_process_priority, 'task_gain': float(controller.task_gain), 'max_task_speed_m_per_s': float(controller.max_task_speed), 'configured_osqp_adaptive_row_threshold': int(controller.osqp_adaptive_row_threshold), 'support_iterations_per_cycle_p50': float(np.percentile(support_iterations, 50)), 'support_iterations_per_cycle_p99': float(np.percentile(support_iterations, 99)), 'support_maximum_kkt_residual': float(np.max(support_residuals)), 'success': task_success, 'ever_sustained_success': ever_sustained_success, 'success_tolerance_m': float(success_tolerance), 'success_comparison': 'strict_less_than', 'success_hold_cycles': int(success_hold_cycles), 'stop_on_success': bool(stop_on_success), 'fixed_duration_completed': bool(len(cycles) == cycle_limit), 'maximum_success_hold_cycles': int(maximum_success_hold), 'evidence_eligible': evidence_eligible, 'evidence_gate_definition': 'strict_goal_hold_and_observability_and_coverage_and_mvt_oracle_and_no_penetration_and_unmodified_sweep_and_controller_p99_le_dt', 'observability_passed': observability_report.passed, 'observability_audit_method': observability_report.audit_method, 'observability_event_code': observability_report.event_code, 'observability_ever_at_risk_samples': observability_report.ever_at_risk_samples, 'observability_observed_before_deadline_samples': observability_report.observed_before_deadline_samples, 'observability_late_samples': observability_report.late_samples, 'observability_never_observed_samples': observability_report.never_observed_samples, 'observability_snapshot_count': observability_report.snapshot_count, 'truth_observability_audit_after_control_only': True, 'truth_observability_feedback_to_control': False, 'cycles': len(cycles), 'simulated_duration_s': len(cycles) * DT, 'wall_duration_s': elapsed, 'first_success_time_s': None if first_sustained_success_start_cycle is None else first_sustained_success_start_cycle * DT, 'first_success_completion_time_s': None if first_sustained_success_completion_cycle is None else first_sustained_success_completion_cycle * DT, 'final_error_m': float(errors[-1]), 'minimum_error_m': float(np.min(errors)), 'steady_window_cycles': int(steady_window_cycles), 'steady_error_mean_m': float(np.mean(steady_errors)), 'steady_error_p95_m': float(np.percentile(steady_errors, 95)), 'steady_error_max_m': float(np.max(steady_errors)), 'maximum_ee_x_m': float(np.max(ee_array[:, 0])), 'exact_penetrating_cycles': int(sum((row['exact_penetrating_contact_count'] > 0 for row in cycles))), 'maximum_penetration_m': float(maximum_penetration), 'occupancy_guard_rejections': int(sum((not row['occupancy_guard_safe'] for row in cycles))), 'unknown_monitor_cycles': int(sum((row['sweep_unknown_voxels'] > 0 for row in cycles))), 'per_cycle_unknown_sweep_evaluated': unknown_policy == 'strict', 'occupied_hard_guard': 'dense_three_state_map' if unknown_policy == 'strict' else 'continuous_centervox_proxy_certificate', 'control_compute_ms_p50': float(np.percentile(compute_ms, 50)), 'control_compute_ms_p95': float(np.percentile(compute_ms, 95)), 'control_compute_ms_p99': float(np.percentile(compute_ms, 99)), 'controller_ms_p99': controller_p99_ms, 'realtime_controller_threshold_ms': realtime_controller_threshold_ms, 'realtime_controller_passed': realtime_controller_passed, 'deadline_misses': deadline_misses, 'realtime_pacing': realtime_pacing, 'perception_executor': perception_executor, 'perception_process_isolated_from_control': perception_executor == 'process', 'perception_process_cpu_affinity': None if perception_cpu_affinity is None else list(perception_cpu_affinity), 'control_process_cpu_affinity': None if control_cpu_affinity is None else list(control_cpu_affinity), 'proxy_and_eigensystem_update_off_control_thread': True, 'workspace_qp_rows_batched': True, 'persistent_osqp_workspace': controller.persistent_qp_workspace, 'persistent_osqp_dual_warm_start_policy': 'semantic_row_identity_remap_preserve_unchanged_zero_new_rows' if controller.persistent_qp_workspace else None, 'persistent_osqp_dual_warm_start_cycles': int(sum((row['qp_dual_warm_start_used'] for row in cycles))), 'persistent_osqp_dual_warm_start_rows_p50': float(np.percentile([row['qp_dual_warm_start_rows'] for row in cycles], 50)), 'persistent_osqp_dual_warm_start_rows_p99': float(np.percentile([row['qp_dual_warm_start_rows'] for row in cycles], 99)), 'persistent_osqp_row_capacity': controller._qp_row_capacity, 'osqp_rho': controller.osqp_rho, 'osqp_adaptive_rho': controller.osqp_adaptive_rho, 'osqp_rho_policy': 'qp_row_count_conditioned_shared', 'osqp_adaptive_row_threshold': controller.osqp_adaptive_row_threshold, 'osqp_low_row_mode': 'rho_0.1_adaptive', 'osqp_dense_row_mode': f"rho_{controller.osqp_rho:g}_{('adaptive' if controller.osqp_adaptive_rho else 'fixed')}", 'osqp_adaptive_low_row_cycles': int(sum((row['qp_rho_mode'] == 'adaptive_low_rows' for row in cycles))), 'osqp_fixed_dense_row_cycles': int(sum((row['qp_rho_mode'] == 'fixed_dense_rows' for row in cycles))), 'osqp_max_iterations': int(controller.osqp_max_iterations), 'osqp_absolute_tolerance': float(controller.osqp_absolute_tolerance), 'osqp_relative_tolerance': float(controller.osqp_relative_tolerance), 'qp_iterations_p50': float(np.percentile(qp_iterations, 50)), 'qp_iterations_p95': float(np.percentile(qp_iterations, 95)), 'qp_iterations_p99': float(np.percentile(qp_iterations, 99)), 'qp_iterations_max': int(np.max(qp_iterations)), 'qp_primal_residual_max': float(np.max(qp_primal_residuals)), 'qp_dual_residual_max': float(np.max(qp_dual_residuals)), 'cyclic_gc_disabled_during_control': cyclic_gc_was_enabled, 'near_state_hessian_batched': True, 'perception_normal_estimation_threads': NORMAL_ESTIMATION_WORKERS, 'snapshot_handle_materialization_atomic_inline': False, 'snapshot_handle_materialization_off_control_thread': True, 'mvt_oracle_execution_process': 'perception_worker', 'published_mvt_rebuilt_from_audited_snapshot': True, 'requested_camera_frames': requested_frames, 'submitted_perception_jobs': submitted_jobs, 'published_perception_frames': published_frames, 'causal_proxy_snapshots_saved': len(proxy_snapshots), 'causal_source_configurations_saved': len(source_configurations), 'causal_occupancy_snapshots_saved': len(occupancy_snapshots), 'dense_map_snapshots_recorded': bool(occupancy_snapshots), 'candidate_pair_rows_saved_online': 0, 'candidate_pairs_reconstructable_post_control': True, 'active_pair_rows_saved': len(active_rows), 'coalesced_replaced_requests': replaced_requests, 'unsubmitted_latest_request_at_stop': queued_request is not None, 'perception_pipeline_ms_p50': float(np.percentile(frame_ms, 50)), 'perception_pipeline_ms_p95': float(np.percentile(frame_ms, 95)), 'perception_pipeline_ms_p99': float(np.percentile(frame_ms, 99)), 'effective_published_perception_hz_wall': published_frames / elapsed, 'snapshot_age_ms_p50': float(np.percentile(snapshot_ages, 50)), 'snapshot_age_ms_p95': float(np.percentile(snapshot_ages, 95)), 'snapshot_age_ms_max': float(np.max(snapshot_ages)), 'initial_depth_points': frames[0]['raw_depth_points'], 'final_center_voxels': frames[-1]['center_voxels'], 'final_proxy_count': frames[-1]['proxy_count'], 'centervox_filter_size_m': centervox_size, 'proxy_radius_policy': 'adaptive_cluster' if certificate_radius_limit is None else 'uniform_final_certificate', 'certificate_radius_limit_m': certificate_radius_limit, 'uniform_sphere_effective_radius_m': certificate_radius_limit, 'ellipsoid_outer_longest_radius_limit_m': certificate_radius_limit, 'all_proxy_radius_limit_checks_passed': all_proxy_radius_limit_checks_passed, 'maximum_observed_sphere_effective_radius_m': float(max((row['sphere_effective_radius_max_m'] for row in frames))), 'maximum_observed_ellipsoid_outer_longest_radius_m': float(max((row['ellipsoid_outer_longest_radius_max_m'] for row in frames))), 'depth_backend': 'mujoco_mj_multiRay_first_visible', 'camera_width_px': int(camera_width), 'camera_height_px': int(camera_height), 'camera_pixel_stride': int(camera_pixel_stride), 'camera_minimum_range_m': CAMERA_MINIMUM_RANGE, 'camera_sample_rays_per_view': int(len(range(camera_pixel_stride // 2, camera_width, camera_pixel_stride)) * len(range(camera_pixel_stride // 2, camera_height, camera_pixel_stride))), 'maximum_uncertainty_union_inflation_limit': maximum_uncertainty_union_inflation, 'uncertainty_fusion_mode': uncertainty_fusion_mode, 'direct_centervox_thin_axis_inflation': float(direct_thin_axis_inflation), 'direct_centervox_tangent_subdivisions': int(direct_tangent_subdivisions), 'direct_centervox_partition_mode': direct_partition_mode, 'direct_centervox_subproxies_per_cell': int(2 if direct_partition_mode == 'longest_tangent_binary' else direct_tangent_subdivisions ** 2), 'maximum_observed_uncertainty_union_inflation': float(final_proxies.maximum_uncertainty_union_inflation), 'all_centervox_coverage_checks_passed': all_centervox_coverage_checks_passed, 'all_mvt_oracle_checks_passed': all_mvt_oracle_checks_passed, 'local_moving_cameras': list(camera_names), 'camera_count': len(camera_names), 'incremental_free_occupied_unknown_map': True, 'global_map_preloaded': False, 'future_frames_used': False, 'perception_snapshot_immutable': True, 'snapshot_source_configuration_logged': True, 'unknown_motion_policy': unknown_policy, 'sweep_guard_mode': sweep_guard_mode, 'sweep_audit_complete_online': sweep_guard_mode != 'post_control_audit', 'sweep_guard_modified_cycles': sweep_modified_cycles, 'sweep_audit_failures': sweep_audit_failures, 'liu_qp_command_executed_without_guard_modification': sweep_guard_mode in {'audit_only', 'post_control_audit'}, 'target_count': 1, 'path_planner': None, 'intermediate_targets': None, 'random_dither': False, 'truth_collision_backtracking': False, 'robot_certificate_spheres': len(robot), 'source_hashes': _source_hashes(), 'platform': platform.platform(), 'python': platform.python_version(), 'mujoco': mujoco.__version__}
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary

def _protocol_scene(version):
    if version != 'camera_quarter':
        raise ValueError('This package contains only the accepted drawer scene')
    return formal_drawer_camera_quarter_scene()

def _source_hashes():
    from package_integrity import source_hashes
    return source_hashes()
