"""Read-only MuJoCo replay of the two accepted recorded experiments."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import queue
import time
import mujoco
import numpy as np
from geometry import optimal_support_normal_with_uncertainty
from robot import build_robot_certificate, certificate_world_state
from scene import _protocol_scene
ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = ROOT / 'baselines'
DEFAULT_SPHERE = FINAL_ROOT / 'v44_sphere'
DEFAULT_ELLIPSOID = FINAL_ROOT / 'v43_ellipsoid'
IDENTITY = np.eye(3).reshape(-1)
SPHERE_RGBA = np.array([0.96, 0.22, 0.05, 0.14], dtype=np.float32)
ELLIPSOID_RGBA = np.array([1.0, 0.55, 0.02, 0.2], dtype=np.float32)
OUTER_RGBA = np.array([1.0, 0.92, 0.12, 0.065], dtype=np.float32)
UNCERTAINTY_RGBA = np.array([0.7, 0.3, 1.0, 0.1], dtype=np.float32)
ROBOT_PROXY_RGBA = np.array([0.05, 0.78, 1.0, 0.2], dtype=np.float32)
POINT_RGBA = np.array([0.15, 0.95, 0.42, 0.72], dtype=np.float32)
FREE_RGBA = np.array([0.12, 0.52, 1.0, 0.1], dtype=np.float32)
OCCUPIED_RGBA = np.array([0.95, 0.08, 0.18, 0.24], dtype=np.float32)
CANDIDATE_RGBA = np.array([0.05, 0.9, 0.95, 0.28], dtype=np.float32)
ACTIVE_RGBA = np.array([0.95, 0.05, 0.78, 0.48], dtype=np.float32)
TARGET_RGBA = np.array([0.2, 0.95, 0.2, 0.9], dtype=np.float32)
LIMIT_ROBOT_RGBA = np.array([1.0, 0.02, 0.45, 0.92], dtype=np.float32)
LIMIT_SAFETY_RGBA = np.array([1.0, 0.02, 0.45, 0.16], dtype=np.float32)
LIMIT_OBSTACLE_RGBA = np.array([1.0, 0.9, 0.02, 0.88], dtype=np.float32)
LIMIT_SUPPORT_RGBA = np.array([1.0, 1.0, 1.0, 0.98], dtype=np.float32)
LIMIT_SAFE_SEGMENT_RGBA = np.array([1.0, 0.05, 0.05, 0.98], dtype=np.float32)
LIMIT_REMAINING_RGBA = np.array([0.15, 1.0, 0.3, 0.98], dtype=np.float32)
LIMIT_VIOLATION_RGBA = np.array([1.0, 0.05, 0.05, 0.98], dtype=np.float32)
LIMIT_RECOVERY_RGBA = np.array([1.0, 0.28, 0.02, 0.98], dtype=np.float32)
DEFAULT_SAFETY_MARGIN_M = 0.006
MAX_DRAWN_POINTS = 3000
MAX_DRAWN_MAP_STATE = 300

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _shape_to_axes_rotation(shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, rotation = np.linalg.eigh(np.asarray(shape, dtype=float))
    order = np.argsort(values)
    rotation = rotation[:, order]
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return (np.sqrt(np.maximum(values[order], 1e-18)), rotation)

def _append_geom(scene, geom_type, size, position, rotation, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, int(geom_type), np.asarray(size, dtype=float), np.asarray(position, dtype=float), np.asarray(rotation, dtype=float).reshape(-1), np.asarray(rgba, dtype=np.float32))
    geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1

def _append_connector(scene, start, stop, rgba, width: float=6.0) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, int(mujoco.mjtGeom.mjGEOM_LINE), np.zeros(3), np.zeros(3), IDENTITY, np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, int(mujoco.mjtGeom.mjGEOM_LINE), float(width), np.asarray(start, dtype=float), np.asarray(stop, dtype=float))
    geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1

class RaggedSnapshots:

    def __init__(self, path: Path) -> None:
        archive = np.load(path)
        self.publish_cycles = np.asarray(archive['publish_cycles'], dtype=np.int64)
        self.source_cycles = np.asarray(archive['source_cycles'], dtype=np.int64)
        self.offsets = np.asarray(archive['offsets'], dtype=np.int64)
        self.arrays = {name: np.asarray(archive[name]) for name in archive.files if name not in {'publish_cycles', 'source_cycles', 'offsets'}}
        if len(self.offsets) != len(self.publish_cycles) + 1:
            raise ValueError(f'invalid ragged offsets: {path}')
        if np.any(np.diff(self.publish_cycles) < 0):
            raise ValueError(f'publication cycles are not ordered: {path}')
        if any((len(value) != int(self.offsets[-1]) for value in self.arrays.values())):
            raise ValueError(f'ragged payload length mismatch: {path}')

    def at_cycle(self, cycle: int) -> tuple[int, dict[str, np.ndarray]]:
        index = int(np.searchsorted(self.publish_cycles, cycle, side='right') - 1)
        index = max(index, 0)
        start, stop = map(int, self.offsets[index:index + 2])
        return (index, {name: value[start:stop] for name, value in self.arrays.items()})

class CycleProxySelection:

    def __init__(self, path: Path) -> None:
        self.by_cycle: dict[int, set[int]] = {}
        self.minimum_by_cycle: dict[int, dict[str, object]] = {}
        with path.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                cycle = int(row['cycle'])
                self.by_cycle.setdefault(cycle, set()).add(int(row['proxy_id']))
                if row.get('clearance') not in (None, ''):
                    clearance = float(row['clearance'])
                    current = self.minimum_by_cycle.get(cycle)
                    if current is None or clearance < float(current['clearance']):
                        self.minimum_by_cycle[cycle] = {'clearance': clearance, 'surface_clearance': float(row.get('surface_clearance', clearance + 0.006)), 'state': row.get('state', 'unknown'), 'robot_index': int(row.get('robot_index', -1)), 'obstacle_index': int(row.get('obstacle_index', -1)), 'proxy_id': int(row['proxy_id']), 'contact_recovery_row': row.get('contact_recovery_row', 'False').lower() == 'true'}

    def at_cycle(self, cycle: int) -> set[int]:
        return self.by_cycle.get(int(cycle), set())

    def minimum_at_cycle(self, cycle: int) -> dict[str, object] | None:
        return self.minimum_by_cycle.get(int(cycle))

class RaggedCycleProxySelection:

    def __init__(self, path: Path) -> None:
        archive = np.load(path)
        self.offsets = np.asarray(archive['offsets'], dtype=np.int64)
        self.proxy_ids = np.asarray(archive['proxy_ids'], dtype=np.int64)

    def at_cycle(self, cycle: int) -> set[int]:
        cycle = int(cycle)
        start, stop = map(int, self.offsets[cycle:cycle + 2])
        return set(map(int, self.proxy_ids[start:stop]))

    def minimum_at_cycle(self, cycle: int):
        return None

class EmptyCycleProxySelection:

    def at_cycle(self, cycle: int) -> set[int]:
        del cycle
        return set()

    def minimum_at_cycle(self, cycle: int):
        del cycle
        return None

class OccupancyDeltaSnapshots:
    _BIAS = np.uint64(1 << 20)
    _MASK = np.uint64((1 << 21) - 1)

    def __init__(self, path: Path) -> None:
        archive = np.load(path)
        self.publish_cycles = np.asarray(archive['publish_cycles'], dtype=np.int64)
        self.source_cycles = np.asarray(archive['source_cycles'], dtype=np.int64)
        self.offsets = np.asarray(archive['offsets'], dtype=np.int64)
        keys = np.asarray(archive['voxel_keys'], dtype=np.int64)
        shifted = (keys + int(self._BIAS)).astype(np.uint64)
        self.delta_keys = shifted[:, 0] | shifted[:, 1] << np.uint64(21) | shifted[:, 2] << np.uint64(42)
        self.delta_states = np.asarray(archive['voxel_states'], dtype=np.int8)
        self.voxel_size = float(archive['voxel_size_m'])
        self._snapshot = -1
        self._keys = np.empty(0, dtype=np.uint64)
        self._states = np.empty(0, dtype=np.int8)

    def _apply(self, index: int) -> None:
        start, stop = map(int, self.offsets[index:index + 2])
        delta_keys = self.delta_keys[start:stop]
        delta_states = self.delta_states[start:stop]
        if not len(self._keys):
            order = np.argsort(delta_keys, kind='stable')
            keys = delta_keys[order]
            states = delta_states[order]
        else:
            keys = np.concatenate((self._keys, delta_keys))
            states = np.concatenate((self._states, delta_states))
            order = np.argsort(keys, kind='stable')
            keys = keys[order]
            states = states[order]
        if len(keys):
            last = np.r_[keys[1:] != keys[:-1], True]
            keys = keys[last]
            states = states[last]
            known = states != 0
            keys = keys[known]
            states = states[known]
        self._keys, self._states = (keys, states)

    def at_cycle(self, cycle: int) -> tuple[int, np.ndarray, np.ndarray]:
        target = int(np.searchsorted(self.publish_cycles, cycle, side='right') - 1)
        target = max(target, 0)
        if target < self._snapshot:
            self._snapshot = -1
            self._keys = np.empty(0, dtype=np.uint64)
            self._states = np.empty(0, dtype=np.int8)
        while self._snapshot < target:
            self._snapshot += 1
            self._apply(self._snapshot)
        return (target, self._keys, self._states)

    @classmethod
    def centers(cls, keys: np.ndarray, voxel_size: float) -> np.ndarray:
        x = (keys & cls._MASK).astype(np.int64) - int(cls._BIAS)
        y = (keys >> np.uint64(21) & cls._MASK).astype(np.int64) - int(cls._BIAS)
        z = (keys >> np.uint64(42) & cls._MASK).astype(np.int64) - int(cls._BIAS)
        return (np.column_stack((x, y, z)).astype(float) + 0.5) * voxel_size

class FormalRun:

    def __init__(self, directory: Path, *, replay_only: bool=False) -> None:
        self.directory = directory
        self.summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        with (directory / 'cycles.csv').open(newline='', encoding='utf-8') as stream:
            self.cycles = list(csv.DictReader(stream))
        self.q = np.asarray(np.load(directory / 'q_history.npy'), dtype=float)
        self.points = RaggedSnapshots(directory / 'causal_observability_snapshots.npz')
        self.proxies = RaggedSnapshots(directory / 'causal_proxy_snapshots.npz')
        compressed_candidates = directory / 'candidate_proxy_ids.npz'
        candidate_pairs = directory / 'candidate_pairs.csv'
        if compressed_candidates.exists():
            self.candidates = RaggedCycleProxySelection(compressed_candidates)
            self.candidate_selection_available = True
        elif candidate_pairs.exists():
            self.candidates = CycleProxySelection(candidate_pairs)
            self.candidate_selection_available = True
        elif replay_only:
            self.candidates = EmptyCycleProxySelection()
            self.candidate_selection_available = False
        else:
            raise FileNotFoundError(candidate_pairs)
        self.active = CycleProxySelection(directory / 'pair_states.csv')
        recorded_occupancy = directory / 'causal_occupancy_snapshots.npz'
        replayed_occupancy = directory / 'causal_occupancy_replay_deltas.npz'
        self.occupancy = OccupancyDeltaSnapshots(recorded_occupancy if recorded_occupancy.exists() else replayed_occupancy)
        candidate_report = directory / 'candidate_replay_report.json'
        sweep_report = directory / 'post_control_sweep_audit_report.json'
        self.candidate_report = json.loads(candidate_report.read_text(encoding='utf-8')) if candidate_report.exists() else None
        self.sweep_report = json.loads(sweep_report.read_text(encoding='utf-8')) if sweep_report.exists() else None
        recorded_report = directory / 'recorded_occupancy_replay_report.json'
        legacy_report = directory / 'occupancy_replay_report.json'
        occupancy_report = recorded_report if recorded_report.exists() else legacy_report
        self.occupancy_report = json.loads(occupancy_report.read_text(encoding='utf-8')) if occupancy_report.exists() else None
        if not replay_only and (self.candidate_report is None or self.sweep_report is None or self.occupancy_report is None):
            raise FileNotFoundError(f'formal audit sidecars are incomplete below {directory}')
        if len(self.q) != len(self.cycles):
            raise ValueError(f'q/cycle length mismatch: {directory}')
        if self.summary['causal_proxy_snapshots_saved'] != len(self.proxies.publish_cycles):
            raise ValueError(f'summary/proxy snapshot mismatch: {directory}')

class FormalComparison:

    def __init__(self, sphere_dir: Path, ellipsoid_dir: Path, *, replay_only: bool=False) -> None:
        self.replay_only = bool(replay_only)
        if sphere_dir.resolve() == ellipsoid_dir.resolve():
            run = FormalRun(sphere_dir, replay_only=self.replay_only)
            representation = str(run.summary['representation'])
            if representation not in {'sphere', 'ellipsoid'}:
                raise ValueError('single replay has an invalid representation')
            self.runs = {representation: run}
            self.locked_mode = representation
            self.model = mujoco.MjModel.from_xml_path(str(sphere_dir / 'scene.xml'))
            self.data = mujoco.MjData(self.model)
            self.robot_spheres = build_robot_certificate(self.model)
            self.target = np.asarray(_protocol_scene(run.summary['scene_version']).waypoints[-1], dtype=float)
            return
        self.runs = {'sphere': FormalRun(sphere_dir, replay_only=self.replay_only), 'ellipsoid': FormalRun(ellipsoid_dir, replay_only=self.replay_only)}
        self.locked_mode = None
        sphere = self.runs['sphere']
        ellipsoid = self.runs['ellipsoid']
        if _sha256(sphere_dir / 'scene.xml') != _sha256(ellipsoid_dir / 'scene.xml'):
            raise ValueError('sphere and ellipsoid scene XML hashes differ')
        if not self.replay_only and sphere.summary['source_hashes'] != ellipsoid.summary['source_hashes']:
            raise ValueError('sphere and ellipsoid source hashes differ')
        if sphere.summary['scene_version'] != ellipsoid.summary['scene_version']:
            raise ValueError('sphere and ellipsoid scene versions differ')
        self.model = mujoco.MjModel.from_xml_path(str(sphere_dir / 'scene.xml'))
        self.data = mujoco.MjData(self.model)
        self.robot_spheres = build_robot_certificate(self.model)
        self.target = np.asarray(_protocol_scene(sphere.summary['scene_version']).waypoints[-1], dtype=float)

    def set_cycle(self, mode: str, cycle: int) -> None:
        run = self.runs[mode]
        self.data.qpos[:6] = run.q[cycle]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def populate(self, scene, mode: str, cycle: int, *, clear_scene: bool=True, show_points: bool, show_obstacles: bool, show_robot: bool, show_core: bool, show_outer: bool, show_uncertainty: bool, show_map: bool, show_candidates: bool, show_active: bool, show_limiting: bool=True) -> dict[str, int]:
        if clear_scene:
            scene.ngeom = 0
        run = self.runs[mode]
        proxy_index, proxies = run.proxies.at_cycle(cycle)
        point_index, observed = run.points.at_cycle(cycle)
        candidate_ids = run.candidates.at_cycle(cycle)
        active_ids = run.active.at_cycle(cycle)
        drawn_points = 0
        if show_points:
            points = observed['points']
            if len(points):
                stride = max(1, int(np.ceil(len(points) / MAX_DRAWN_POINTS)))
                for point in points[::stride]:
                    _append_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.0015), point, IDENTITY, POINT_RGBA)
                    drawn_points += 1
        drawn_obstacles = 0
        if mode == 'sphere':
            for proxy_id, center, radius, offset in zip(proxies['proxy_ids'], proxies['centers'], proxies['sphere_radii'], proxies['uncertainty_offsets']):
                selected_active = show_active and int(proxy_id) in active_ids
                selected_candidate = show_candidates and int(proxy_id) in candidate_ids
                if not (show_obstacles or selected_candidate or selected_active):
                    continue
                rgba = ACTIVE_RGBA if selected_active else CANDIDATE_RGBA if selected_candidate else SPHERE_RGBA
                _append_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, float(radius + offset)), center, IDENTITY, rgba)
                drawn_obstacles += 1
        else:
            outer_matches_core = bool(np.array_equal(proxies['ellipsoid_outer_shapes'], proxies['ellipsoid_shapes']))
            if show_outer and (not (show_core and outer_matches_core)):
                for proxy_id, center, shape in zip(proxies['proxy_ids'], proxies['centers'], proxies['ellipsoid_outer_shapes']):
                    selected_active = show_active and int(proxy_id) in active_ids
                    selected_candidate = show_candidates and int(proxy_id) in candidate_ids
                    if not (show_obstacles or selected_candidate or selected_active):
                        continue
                    axes, rotation = _shape_to_axes_rotation(shape)
                    _append_geom(scene, mujoco.mjtGeom.mjGEOM_ELLIPSOID, axes, center, rotation, ACTIVE_RGBA if selected_active else CANDIDATE_RGBA if selected_candidate else OUTER_RGBA)
                    drawn_obstacles += 1
            if show_core:
                for proxy_id, center, shape in zip(proxies['proxy_ids'], proxies['centers'], proxies['ellipsoid_shapes']):
                    selected_active = show_active and int(proxy_id) in active_ids
                    selected_candidate = show_candidates and int(proxy_id) in candidate_ids
                    if not (show_obstacles or selected_candidate or selected_active):
                        continue
                    axes, rotation = _shape_to_axes_rotation(shape)
                    _append_geom(scene, mujoco.mjtGeom.mjGEOM_ELLIPSOID, axes, center, rotation, ACTIVE_RGBA if selected_active else CANDIDATE_RGBA if selected_candidate else ELLIPSOID_RGBA)
                    drawn_obstacles += 1
            if show_uncertainty:
                for proxy_id, center, shape in zip(proxies['proxy_ids'], proxies['centers'], proxies['proxy_uncertainty_shapes']):
                    selected_active = show_active and int(proxy_id) in active_ids
                    selected_candidate = show_candidates and int(proxy_id) in candidate_ids
                    if not (show_obstacles or selected_candidate or selected_active):
                        continue
                    axes, rotation = _shape_to_axes_rotation(shape)
                    _append_geom(scene, mujoco.mjtGeom.mjGEOM_ELLIPSOID, axes, center, rotation, UNCERTAINTY_RGBA)
                    drawn_obstacles += 1
        map_index, map_keys, map_states = run.occupancy.at_cycle(cycle)
        free_count = int(np.count_nonzero(map_states == 1))
        occupied_count = int(np.count_nonzero(map_states == 2))
        drawn_free = 0
        drawn_occupied = 0
        if show_map:
            half = np.full(3, 0.5 * run.occupancy.voxel_size)
            for state, rgba in ((1, FREE_RGBA), (2, OCCUPIED_RGBA)):
                selected = map_keys[map_states == state]
                stride = max(1, int(np.ceil(len(selected) / MAX_DRAWN_MAP_STATE)))
                sample = selected[::stride][:MAX_DRAWN_MAP_STATE]
                for position in OccupancyDeltaSnapshots.centers(sample, run.occupancy.voxel_size):
                    _append_geom(scene, mujoco.mjtGeom.mjGEOM_BOX, half, position, IDENTITY, rgba)
                if state == 1:
                    drawn_free = len(sample)
                else:
                    drawn_occupied = len(sample)
        drawn_robot = 0
        robot_positions = None
        robot_radii = None
        if show_robot or show_limiting:
            robot_positions, _, robot_radii = certificate_world_state(self.model, self.data, self.robot_spheres)
        if show_robot:
            for position, radius in zip(robot_positions, robot_radii):
                _append_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, float(radius)), position, IDENTITY, ROBOT_PROXY_RGBA)
                drawn_robot += 1
        row = run.cycles[cycle]
        limiting_robot_index = int(row.get('limiting_robot_index', -1))
        limiting_obstacle_index = int(row.get('limiting_obstacle_index', -1))
        limiting_proxy_id = int(row.get('limiting_proxy_id', -1))
        limiting_pair_drawn = 0
        limiting_logged_clearance = float(row.get('min_clearance', 'nan'))
        limiting_state = 'none'
        limiting_contact_recovery = False
        limiting_logged_surface_clearance = float('nan')
        limiting_record = run.active.minimum_at_cycle(cycle)
        if limiting_record is not None:
            limiting_state = str(limiting_record['state'])
            limiting_logged_surface_clearance = float(limiting_record['surface_clearance'])
            limiting_contact_recovery = bool(limiting_record['contact_recovery_row'])
        limiting_reconstructed_clearance = float('nan')
        limiting_raw_gap = float('nan')
        limiting_robot_body = 'none'
        safety_margin = float(run.summary.get('safety_margin_m', DEFAULT_SAFETY_MARGIN_M))
        if show_limiting and robot_positions is not None and (0 <= limiting_robot_index < len(robot_positions)) and (limiting_proxy_id >= 0):
            proxy_ids = np.asarray(proxies['proxy_ids'], dtype=np.int64)
            if not (0 <= limiting_obstacle_index < len(proxy_ids) and int(proxy_ids[limiting_obstacle_index]) == limiting_proxy_id):
                matches = np.flatnonzero(proxy_ids == limiting_proxy_id)
                limiting_obstacle_index = int(matches[0]) if len(matches) else -1
            if 0 <= limiting_obstacle_index < len(proxy_ids):
                robot_center = robot_positions[limiting_robot_index]
                robot_radius = float(robot_radii[limiting_robot_index])
                obstacle_center = proxies['centers'][limiting_obstacle_index]
                obstacle_offset = float(proxies['uncertainty_offsets'][limiting_obstacle_index])
                limiting_robot_body = self.robot_spheres[limiting_robot_index].body_name
                if mode == 'sphere':
                    obstacle_radius = float(proxies['sphere_radii'][limiting_obstacle_index]) + obstacle_offset
                    delta = obstacle_center - robot_center
                    distance = float(np.linalg.norm(delta))
                    normal = np.array([1.0, 0.0, 0.0]) if distance <= 1e-12 else delta / distance
                    obstacle_surface = obstacle_center - obstacle_radius * normal
                    _append_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, obstacle_radius), obstacle_center, IDENTITY, LIMIT_OBSTACLE_RGBA)
                else:
                    obstacle_shape = proxies['ellipsoid_shapes'][limiting_obstacle_index]
                    uncertainty_shape = proxies['proxy_uncertainty_shapes'][limiting_obstacle_index]
                    support = optimal_support_normal_with_uncertainty(robot_center, obstacle_center, obstacle_shape, uncertainty_shape)
                    normal = support.normal
                    obstacle_surface = support.obstacle_surface_point - obstacle_offset * normal
                    axes, rotation = _shape_to_axes_rotation(obstacle_shape)
                    _append_geom(scene, mujoco.mjtGeom.mjGEOM_ELLIPSOID, axes, obstacle_center, rotation, LIMIT_OBSTACLE_RGBA)
                robot_surface = robot_center + robot_radius * normal
                safety_boundary = robot_surface + safety_margin * normal
                limiting_raw_gap = float(normal @ (obstacle_surface - robot_surface))
                limiting_reconstructed_clearance = limiting_raw_gap - safety_margin
                _append_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, robot_radius), robot_center, IDENTITY, LIMIT_ROBOT_RGBA)
                _append_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, robot_radius + safety_margin), robot_center, IDENTITY, LIMIT_SAFETY_RGBA)
                for point, rgba in ((robot_surface, LIMIT_ROBOT_RGBA), (safety_boundary, LIMIT_SAFE_SEGMENT_RGBA), (obstacle_surface, LIMIT_SUPPORT_RGBA)):
                    _append_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.004), point, IDENTITY, rgba)
                _append_connector(scene, robot_surface, safety_boundary, LIMIT_SAFE_SEGMENT_RGBA, width=8.0)
                _append_connector(scene, safety_boundary, obstacle_surface, LIMIT_RECOVERY_RGBA if limiting_contact_recovery else LIMIT_REMAINING_RGBA if limiting_reconstructed_clearance >= 0.0 else LIMIT_VIOLATION_RGBA, width=8.0)
                limiting_pair_drawn = 1
        _append_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.018), self.target, IDENTITY, TARGET_RGBA)
        return {'proxy_snapshot': proxy_index, 'proxy_publish_cycle': int(run.proxies.publish_cycles[proxy_index]), 'proxy_source_cycle': int(run.proxies.source_cycles[proxy_index]), 'proxy_count': len(proxies['centers']), 'point_snapshot': point_index, 'observed_point_count': len(observed['points']), 'drawn_points': drawn_points, 'drawn_obstacles': drawn_obstacles, 'drawn_robot': drawn_robot, 'candidate_proxy_count': len(candidate_ids), 'active_proxy_count': len(active_ids), 'map_snapshot': map_index, 'free_voxel_count': free_count, 'occupied_voxel_count': occupied_count, 'drawn_free_voxels': drawn_free, 'drawn_occupied_voxels': drawn_occupied, 'limiting_pair_drawn': limiting_pair_drawn, 'limiting_robot_index': limiting_robot_index, 'limiting_obstacle_index': limiting_obstacle_index, 'limiting_proxy_id': limiting_proxy_id, 'limiting_robot_body': limiting_robot_body, 'limiting_logged_clearance_m': limiting_logged_clearance, 'limiting_logged_surface_clearance_m': limiting_logged_surface_clearance, 'limiting_state': limiting_state, 'limiting_contact_recovery': limiting_contact_recovery, 'limiting_reconstructed_clearance_m': limiting_reconstructed_clearance, 'limiting_raw_gap_m': limiting_raw_gap, 'safety_margin_m': safety_margin}

    def check(self) -> None:
        scene = mujoco.MjvScene(self.model, 20000)
        for mode, run in self.runs.items():
            if not self.replay_only:
                assert run.candidate_report['passed']
                assert run.candidate_report['all_per_cycle_candidate_counts_matched']
                assert run.candidate_report['cycles'] == len(run.cycles)
                assert run.sweep_report['passed']
                assert run.sweep_report['liu_qp_commands_modified'] is False
                assert run.sweep_report['unsafe_cycles'] == 0
                assert run.sweep_report['cycles'] == len(run.cycles)
                assert run.occupancy_report['passed']
                if 'all_online_frame_state_counts_matched' in run.occupancy_report:
                    assert run.occupancy_report['all_online_frame_state_counts_matched']
                else:
                    assert run.occupancy_report['cycles_match_source_proxy_and_frame_logs']
                assert run.occupancy_report['snapshot_count'] == len(run.occupancy.publish_cycles)
            assert np.all(run.points.source_cycles <= run.points.publish_cycles)
            assert np.all(run.proxies.source_cycles <= run.proxies.publish_cycles)
            assert np.all(run.occupancy.source_cycles <= run.occupancy.publish_cycles)
            if run.candidate_selection_available:
                assert all((run.active.at_cycle(cycle) <= run.candidates.at_cycle(cycle) for cycle in range(len(run.cycles))))
            for cycle in (0, len(run.cycles) // 2, len(run.cycles) - 1):
                self.set_cycle(mode, cycle)
                counts = self.populate(scene, mode, cycle, show_points=True, show_obstacles=True, show_robot=True, show_core=True, show_outer=False, show_uncertainty=False, show_map=True, show_candidates=True, show_active=True)
                assert counts['drawn_robot'] == len(self.robot_spheres) == 65
                assert counts['proxy_publish_cycle'] <= cycle
            print(f"{mode}: cycles={len(run.cycles)}, proxy_snapshots={len(run.proxies.publish_cycles)}, point_snapshots={len(run.points.publish_cycles)}, success={run.summary['success']}, deadline_misses={run.summary['deadline_misses']}")
        if self.replay_only:
            print('replay-only load verified: recorded trajectory, causal points, proxies, occupancy, and QP-active rows are readable; missing formal sidecars were not claimed as audited')
            return
        prefix = 'single-run artifacts loaded; ' if self.locked_mode is not None else 'scene/source hashes match; '
        print(prefix + 'causal publication ordering, candidate replay, active-subset, occupancy replay, and unmodified continuous sweep audit verified')

    def run(self, initial: str, speed: float) -> None:
        import mujoco.viewer
        commands: queue.SimpleQueue[int] = queue.SimpleQueue()
        mode = self.locked_mode or initial
        cycle = 0
        paused = False
        loop = True
        overlays_visible = True
        show_points = True
        show_obstacles = True
        show_robot = True
        show_core = True
        show_outer = False
        show_uncertainty = False
        show_map = False
        show_candidates = False
        show_active = True
        show_limiting = True
        step_once = False
        playback_speed = speed
        if self.locked_mode is None:
            print('1 sphere | 2 ellipsoid | V all overlays | P causal points')
        else:
            print(f'single-run replay locked to {self.locked_mode} | V all overlays | P causal points')
        print('O all generated proxies | B broadphase candidates | Q final QP proxies')
        print('M replayed free/occupied map | R robot certificate spheres')
        print('C exact QP ellipsoids | E outer envelope (identical in fused U=0 runs) | U uncertainty')
        print('H limiting pair: magenta robot, yellow obstacle, red safety, green h, orange recovery')
        print('Space pause | N step | 0 restart | F final | G best error | L loop | -/= speed')
        print('V never hides the exact MuJoCo UR5 body or physical drawer.')
        if any((not run.candidate_selection_available for run in self.runs.values())):
            print('Replay note: this result did not persist broadphase candidate IDs; B is empty, while O and Q remain exact recorded overlays.')
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=commands.put, show_left_ui=False, show_right_ui=True) as viewer:
            viewer.opt.geomgroup[1] = 0
            viewer.cam.lookat[:] = np.array([0.42, 0.4, 0.52])
            viewer.cam.distance = 1.42
            viewer.cam.azimuth = 0.0
            viewer.cam.elevation = -15.0
            while viewer.is_running():
                while not commands.empty():
                    keycode = commands.get()
                    char = chr(keycode).upper() if 0 <= keycode < 128 else ''
                    if char == '1' and 'sphere' in self.runs:
                        mode, cycle, paused = ('sphere', 0, False)
                    elif char == '2' and 'ellipsoid' in self.runs:
                        mode, cycle, paused = ('ellipsoid', 0, False)
                    elif char == 'V':
                        overlays_visible = not overlays_visible
                    elif char == 'P':
                        show_points = not show_points
                    elif char == 'O':
                        show_obstacles = not show_obstacles
                    elif char == 'R':
                        show_robot = not show_robot
                    elif char == 'M':
                        show_map = not show_map
                    elif char == 'B':
                        show_candidates = not show_candidates
                    elif char == 'Q':
                        show_active = not show_active
                    elif char == 'C':
                        show_core = not show_core
                    elif char == 'E':
                        show_outer = not show_outer
                    elif char == 'U':
                        show_uncertainty = not show_uncertainty
                    elif char == 'H':
                        show_limiting = not show_limiting
                    elif char in (' ',):
                        paused = not paused
                    elif char == 'N':
                        paused, step_once = (True, True)
                    elif char == '0':
                        cycle = 0
                    elif char == 'F':
                        cycle = len(self.runs[mode].cycles) - 1
                        paused = True
                    elif char == 'G':
                        cycle = int(np.argmin([float(item['error_m']) for item in self.runs[mode].cycles]))
                        paused = True
                    elif char == 'L':
                        loop = not loop
                    elif char in ('-', '_'):
                        playback_speed = max(0.125, playback_speed / 2.0)
                    elif char in ('=', '+'):
                        playback_speed = min(8.0, playback_speed * 2.0)
                run = self.runs[mode]
                cycle = min(cycle, len(run.cycles) - 1)
                row = run.cycles[cycle]
                started = time.perf_counter()
                with viewer.lock():
                    self.set_cycle(mode, cycle)
                    counts = self.populate(viewer.user_scn, mode, cycle, show_points=overlays_visible and show_points, show_obstacles=overlays_visible and show_obstacles, show_robot=overlays_visible and show_robot, show_core=show_core, show_outer=show_outer, show_uncertainty=show_uncertainty, show_map=overlays_visible and show_map, show_candidates=overlays_visible and show_candidates, show_active=overlays_visible and show_active, show_limiting=overlays_visible and show_limiting)
                label = f'{mode} LiuQP — ' + ('reached target' if run.summary['success'] else 'did not reach target')
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, label, f"t={float(row['time_s']):.2f}s  error={1000 * float(row['error_m']):.1f} mm  QP={row['status']}  speed={playback_speed:g}x\ncausal proxy gen={counts['proxy_snapshot']} publish={counts['proxy_publish_cycle']} source={counts['proxy_source_cycle']} proxies={counts['proxy_count']}\nobserved CenterVox={counts['observed_point_count']} (drawn={counts['drawn_points']})  robot spheres={counts['drawn_robot']}\nAABB candidate proxies={counts['candidate_proxy_count']}  final QP proxies={counts['active_proxy_count']}\nlimit pair=R{counts['limiting_robot_index']} ({counts['limiting_robot_body']}) <-> P{counts['limiting_proxy_id']}  state={counts['limiting_state']}  c(log)={1000 * counts['limiting_logged_surface_clearance_m']:.2f} mm  h(log)={1000 * counts['limiting_logged_clearance_m']:.2f} mm  h(recon)={1000 * counts['limiting_reconstructed_clearance_m']:.2f} mm\nexact proxy surface gap={1000 * counts['limiting_raw_gap_m']:.2f} mm  hard safety={1000 * counts['safety_margin_m']:.1f} mm  MuJoCo penetrating contacts={row['exact_penetrating_contact_count']}\nmap snapshot={counts['map_snapshot']}  FREE={counts['free_voxel_count']}  OCCUPIED={counts['occupied_voxel_count']}  UNKNOWN=all absent bounded-grid keys\nFused ellipsoid QP uses exact point-to-E(c,Q) gap; U=0 and offset=0, with no post-core inflation.\n1/2 mode | V overlays | P points | O all proxies | B candidates | Q QP-active | M map | R robot balls | H limiting pair"))
                viewer.sync()
                if not paused or step_once:
                    cycle += 1
                    step_once = False
                    if cycle >= len(run.cycles):
                        if loop:
                            cycle = 0
                        else:
                            cycle = len(run.cycles) - 1
                            paused = True
                remaining = 0.02 / playback_speed - (time.perf_counter() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
