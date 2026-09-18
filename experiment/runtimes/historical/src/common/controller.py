"""Extracted original v4.3 ellipsoid dependency; see docs/source_manifest.json."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import time
from typing import NamedTuple, Protocol
import mujoco
import numpy as np
import osqp
from scipy import sparse
from geometry import closest_point_on_ellipsoid, optimal_support_normal_with_uncertainty
from robot import DT, JOINT_NAMES, CertificateSphere, SceneDefinition, attachment_position, certificate_world_state
from native_support import NativeEllipsoidSupport

class CandidateIndex(Protocol):

    def query_sphere(self, center: np.ndarray, radius: float) -> np.ndarray:
        ...

class ObstacleState(str, Enum):
    NORMAL = 'NORMAL'
    NEAR = 'NEAR'
    CONTACT_RECOVERY = 'CONTACT/RECOVERY'

def classify_clearance(surface_clearance: float, *, near_distance: float, contact_distance: float) -> ObstacleState:
    if surface_clearance <= contact_distance:
        return ObstacleState.CONTACT_RECOVERY
    if surface_clearance < near_distance:
        return ObstacleState.NEAR
    return ObstacleState.NORMAL

@dataclass(frozen=True, slots=True)
class ProtocolSeparatingPlane:
    robot_index: int
    obstacle_index: int
    proxy_id: int
    normal_to_obstacle: np.ndarray
    surface_point: np.ndarray
    offset: float
    clearance: float
    newton_iterations: int = 0
    bisection_iterations: int = 0
    closest_point_residual: float = 0.0
    support_iterations: int = 0
    support_residual: float = 0.0
    directional_support: bool = False

class PairRecord(NamedTuple):
    robot_index: int
    obstacle_index: int
    proxy_id: int
    surface_clearance: float
    clearance: float
    state: str
    qp_collision_row: bool
    near_penalty: bool
    contact_recovery_row: bool

@dataclass
class ProtocolStepMetrics:
    status: str
    representation: str
    solve_ms: float
    setup_ms: float
    geometry_ms: float
    prune_ms: float
    total_controller_ms: float
    ee_error: float
    task_speed: float
    qdot_norm: float
    min_clearance: float
    raw_pairs: int
    broadphase_candidate_pairs: int
    broadphase_rejected_pairs: int
    active_obstacle_rows: int
    redundant_proxies_removed: int
    normal_pairs: int
    near_penalty_terms: int
    contact_repulsion_rows: int
    workspace_rows: int
    closest_point_newton_iterations: int
    closest_point_bisection_iterations: int
    closest_point_max_residual: float
    multiplier_warm_start_hits: int
    support_normal_iterations: int
    support_normal_max_residual: float
    support_normal_warm_start_hits: int
    qp_iterations: int
    qp_primal_residual: float
    qp_dual_residual: float
    qp_rho_mode: str
    qp_dual_warm_start_used: bool
    qp_dual_warm_start_rows: int
    qp_row_sha256: str
    limiting_robot_index: int
    limiting_obstacle_index: int
    limiting_proxy_id: int

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)

class ProtocolLiuQPController:

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, scene: SceneDefinition, robot_spheres: list[CertificateSphere], obstacle_centers: np.ndarray, *, representation: str, obstacle_radii: np.ndarray | None=None, obstacle_shapes: np.ndarray | None=None, obstacle_uncertainty_shapes: np.ndarray | None=None, obstacle_offsets: np.ndarray | None=None, proxy_ids: np.ndarray | None=None, safety_margin: float=0.006, near_distance: float=0.04, contact_distance: float=0.0, obstacle_index: CandidateIndex | None=None, redundant_plane_pruning: bool=True, native_exact_batch: bool=True, native_exact_pair_batch: bool=True, native_exact_lazy_prune: bool=True, native_directional_pair_batch: bool=True, ellipsoid_pair_threads: int=8, ellipsoid_pair_affinity_mask: int=0, persistent_qp_workspace: bool=True, osqp_rho: float=0.01, osqp_adaptive_rho: bool=False, qp_workspace_min_rows: int=768, osqp_adaptive_row_threshold: int=512) -> None:
        if representation not in {'sphere', 'ellipsoid'}:
            raise ValueError('representation must be sphere or ellipsoid')
        self.model = model
        self.data = data
        self.scene = scene
        self.robot_spheres = robot_spheres
        self.representation = representation
        self.safety_margin = float(safety_margin)
        self.near_distance = float(near_distance)
        self.contact_distance = float(contact_distance)
        self.obstacle_index = obstacle_index
        self.redundant_plane_pruning = bool(redundant_plane_pruning)
        self.native_exact_batch = bool(native_exact_batch)
        self.native_exact_pair_batch = bool(native_exact_pair_batch)
        self.native_exact_lazy_prune = bool(native_exact_lazy_prune)
        self.native_directional_pair_batch = bool(native_directional_pair_batch)
        self.persistent_qp_workspace = bool(persistent_qp_workspace)
        if osqp_rho <= 0.0:
            raise ValueError('OSQP rho must be positive')
        if qp_workspace_min_rows < 1:
            raise ValueError('QP workspace minimum rows must be positive')
        if osqp_adaptive_row_threshold < 1:
            raise ValueError('OSQP adaptive row threshold must be positive')
        self.osqp_rho = float(osqp_rho)
        self.osqp_adaptive_rho = bool(osqp_adaptive_rho)
        self.qp_workspace_min_rows = int(qp_workspace_min_rows)
        self.osqp_adaptive_row_threshold = int(osqp_adaptive_row_threshold)
        self.osqp_absolute_tolerance = 2e-05
        self.osqp_relative_tolerance = 2e-05
        self.osqp_max_iterations = 6000
        self._current_osqp_rho = self.osqp_rho
        self._current_osqp_adaptive = self.osqp_adaptive_rho
        self._current_osqp_mode = 'uninitialized'
        self.nv = len(JOINT_NAMES)
        self.dt = DT
        self.ee_site_id = model.site('attachment_site').id
        self.task_gain = 2.0
        self.max_task_speed = 0.18
        self.task_weight = 90.0
        self.near_weight = 22.0
        self.velocity_regularization = 1.0
        self.repulsive_speed = 0.035
        self.velocity_limits = np.array([1.0, 1.0, 1.0, 1.4, 1.4, 1.4])
        self.joint_min = np.array([model.jnt_range[model.joint(name).id, 0] for name in JOINT_NAMES])
        self.joint_max = np.array([model.jnt_range[model.joint(name).id, 1] for name in JOINT_NAMES])
        self.joint_padding = 0.015
        self.workspace_A = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
        self.workspace_b = np.array([0.95, 0.95, 0.95, 0.95, 0.0, 1.35])
        self.previous_velocity = np.zeros(self.nv)
        self._multiplier_cache: dict[tuple[int, int], float] = {}
        self._multiplier_dense = np.empty((len(robot_spheres), 0), dtype=float)
        self._normal_cache: dict[tuple[int, int], np.ndarray] = {}
        self._support_kernel = NativeEllipsoidSupport()
        self.ellipsoid_pair_threads = self._support_kernel.set_pair_threads(int(ellipsoid_pair_threads))
        self.ellipsoid_pair_affinity_mask = self._support_kernel.set_pair_affinity_mask(int(ellipsoid_pair_affinity_mask))
        self._last_support_batch_stats = (0, 0.0, 0)
        self._last_closest_batch_stats = (0, 0, 0.0, 0, False)
        self._qp_solver: osqp.OSQP | None = None
        self._qp_row_capacity = 0
        self._qp_A_template: sparse.csc_matrix | None = None
        self._qp_P_template: sparse.csc_matrix | None = None
        self._prefetched_raw_counts: dict[int, int] = {}
        self._prefetched_closest_stats: dict[int, tuple[int, int, float, int]] = {}
        self.last_pair_records: tuple[PairRecord, ...] = ()
        self._previous_qp_full_row_identity: tuple[tuple[str, int, int], ...] | None = None
        self._previous_qp_dual: np.ndarray | None = None
        self.last_qp_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self.last_qp_solver_info = None
        self.update_obstacles(obstacle_centers, obstacle_radii=obstacle_radii, obstacle_shapes=obstacle_shapes, obstacle_uncertainty_shapes=obstacle_uncertainty_shapes, obstacle_offsets=obstacle_offsets, proxy_ids=proxy_ids, obstacle_index=obstacle_index)

    def update_obstacles(self, centers: np.ndarray, *, obstacle_radii: np.ndarray | None=None, obstacle_shapes: np.ndarray | None=None, obstacle_uncertainty_shapes: np.ndarray | None=None, obstacle_offsets: np.ndarray | None=None, proxy_ids: np.ndarray | None=None, obstacle_index: CandidateIndex | None=None, obstacle_eigenvalues: np.ndarray | None=None, obstacle_rotations: np.ndarray | None=None, obstacle_uncertainty_eigenvalues: np.ndarray | None=None) -> None:
        centers = np.asarray(centers, dtype=float).reshape(-1, 3)
        count = len(centers)
        previous_ids = np.asarray(getattr(self, 'proxy_ids', np.empty(0, dtype=np.int64)), dtype=np.int64)
        previous_dense = self._multiplier_dense
        ids = np.arange(count, dtype=np.int64) if proxy_ids is None else np.asarray(proxy_ids, dtype=np.int64).reshape(-1)
        if len(ids) != count or len(np.unique(ids)) != count:
            raise ValueError('proxy_ids must be unique and match obstacle count')
        offsets = np.zeros(count, dtype=float) if obstacle_offsets is None else np.asarray(obstacle_offsets, dtype=float).reshape(-1)
        if len(offsets) != count or np.any(offsets < 0.0):
            raise ValueError('obstacle_offsets must be nonnegative and match obstacle count')
        self.obstacle_centers = centers
        self.proxy_ids = ids
        self.obstacle_offsets = offsets
        if self.representation == 'sphere':
            if obstacle_radii is None:
                raise ValueError('sphere representation requires obstacle_radii')
            self.obstacle_radii = np.asarray(obstacle_radii, dtype=float).reshape(-1)
            if len(self.obstacle_radii) != count:
                raise ValueError('obstacle_radii must match centers')
            self.obstacle_shapes = None
            self.obstacle_uncertainty_shapes = None
            self.obstacle_eigenvalues = None
            self.obstacle_rotations = None
        else:
            if obstacle_shapes is None:
                raise ValueError('ellipsoid representation requires obstacle_shapes')
            self.obstacle_shapes = np.asarray(obstacle_shapes, dtype=float).reshape(-1, 3, 3)
            if len(self.obstacle_shapes) != count:
                raise ValueError('obstacle_shapes must match centers')
            if obstacle_eigenvalues is None or obstacle_rotations is None:
                self.obstacle_eigenvalues, self.obstacle_rotations = np.linalg.eigh(self.obstacle_shapes)
            else:
                self.obstacle_eigenvalues = np.asarray(obstacle_eigenvalues, dtype=float).reshape(-1, 3)
                self.obstacle_rotations = np.asarray(obstacle_rotations, dtype=float).reshape(-1, 3, 3)
                if len(self.obstacle_eigenvalues) != count or len(self.obstacle_rotations) != count:
                    raise ValueError('precomputed obstacle eigensystem must match centers')
            if np.any(self.obstacle_eigenvalues <= 0.0):
                raise ValueError('all obstacle ellipsoids must be positive definite')
            if obstacle_uncertainty_shapes is None:
                self.obstacle_uncertainty_shapes = None
            else:
                uncertainty = np.asarray(obstacle_uncertainty_shapes, dtype=float).reshape(-1, 3, 3)
                if len(uncertainty) != count:
                    raise ValueError('obstacle_uncertainty_shapes must match centers')
                uncertainty = 0.5 * (uncertainty + np.swapaxes(uncertainty, 1, 2))
                if np.any(np.linalg.eigvalsh(uncertainty) < -1e-12):
                    raise ValueError('uncertainty shapes must be positive semidefinite')
                self.obstacle_uncertainty_shapes = uncertainty
                if obstacle_uncertainty_eigenvalues is None:
                    self.obstacle_uncertainty_eigenvalues = np.linalg.eigvalsh(uncertainty)
                else:
                    self.obstacle_uncertainty_eigenvalues = np.asarray(obstacle_uncertainty_eigenvalues, dtype=float).reshape(-1, 3)
                    if len(self.obstacle_uncertainty_eigenvalues) != count:
                        raise ValueError('precomputed uncertainty eigenvalues must match centers')
            self.obstacle_radii = None
            if obstacle_uncertainty_shapes is None:
                self.obstacle_uncertainty_eigenvalues = None
        valid_ids = set(map(int, ids))
        self._multiplier_cache = {key: value for key, value in self._multiplier_cache.items() if key[1] in valid_ids}
        self._normal_cache = {key: value for key, value in self._normal_cache.items() if key[1] in valid_ids}
        aligned_dense = np.full((len(self.robot_spheres), count), np.nan, dtype=float)
        if len(previous_ids) and previous_dense.shape[1] == len(previous_ids):
            _, previous_indices, current_indices = np.intersect1d(previous_ids, ids, assume_unique=True, return_indices=True)
            aligned_dense[:, current_indices] = previous_dense[:, previous_indices]
        self._multiplier_dense = aligned_dense
        self.obstacle_index = obstacle_index

    def task_feedback(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ee = attachment_position(self.model, self.data)
        jacobian = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacobian, jac_rot, self.ee_site_id)
        jacobian = jacobian[:, :self.nv]
        desired = self.task_gain * (np.asarray(target, dtype=float) - ee)
        speed = float(np.linalg.norm(desired))
        if speed > self.max_task_speed:
            desired *= self.max_task_speed / speed
        return (ee, jacobian, desired)

    def joint_velocity_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        q = self.data.qpos[:self.nv]
        position_lower = (self.joint_min + self.joint_padding - q) / self.dt
        position_upper = (self.joint_max - self.joint_padding - q) / self.dt
        return (np.maximum(-self.velocity_limits, position_lower), np.minimum(self.velocity_limits, position_upper))

    def _candidate_indices(self, center: np.ndarray, radius: float) -> np.ndarray:
        if self.obstacle_index is None:
            return np.arange(len(self.obstacle_centers), dtype=np.int64)
        values = np.asarray(self.obstacle_index.query_sphere(center, float(radius)), dtype=np.int64)
        if len(values) and (np.min(values) < 0 or np.max(values) >= len(self.obstacle_centers)):
            raise RuntimeError('candidate index returned an invalid proxy index')
        return np.unique(values)

    def separating_plane(self, robot_index: int, robot_center: np.ndarray, robot_radius: float, obstacle_index: int) -> ProtocolSeparatingPlane:
        obstacle_center = self.obstacle_centers[obstacle_index]
        proxy_id = int(self.proxy_ids[obstacle_index])
        if self.representation == 'sphere':
            delta = obstacle_center - robot_center
            distance = float(np.linalg.norm(delta))
            normal = np.array([1.0, 0.0, 0.0]) if distance <= 1e-12 else delta / distance
            effective_radius = float(self.obstacle_radii[obstacle_index]) + float(self.obstacle_offsets[obstacle_index])
            surface = obstacle_center - effective_radius * normal
            clearance = distance - float(robot_radius) - effective_radius - self.safety_margin
            return ProtocolSeparatingPlane(robot_index=robot_index, obstacle_index=obstacle_index, proxy_id=proxy_id, normal_to_obstacle=normal, surface_point=surface, offset=float(normal @ surface), clearance=float(clearance))
        cache_key = (int(robot_index), proxy_id)
        uncertainty_shape = None if self.obstacle_uncertainty_shapes is None else self.obstacle_uncertainty_shapes[obstacle_index]
        if uncertainty_shape is not None and float(np.trace(uncertainty_shape)) > 1e-20:
            support_result = optimal_support_normal_with_uncertainty(robot_center, obstacle_center, self.obstacle_shapes[obstacle_index], uncertainty_shape, initial_normal=self._normal_cache.get(cache_key))
            self._normal_cache[cache_key] = support_result.normal.copy()
            offset_radius = float(self.obstacle_offsets[obstacle_index])
            expanded_surface = support_result.obstacle_surface_point - offset_radius * support_result.normal
            return ProtocolSeparatingPlane(robot_index=robot_index, obstacle_index=obstacle_index, proxy_id=proxy_id, normal_to_obstacle=support_result.normal, surface_point=expanded_surface, offset=float(support_result.normal @ expanded_surface), clearance=float(support_result.clearance_without_robot - offset_radius - robot_radius - self.safety_margin), support_iterations=support_result.iterations, support_residual=support_result.residual, directional_support=True)
        initial = self._multiplier_cache.get(cache_key)
        closest = closest_point_on_ellipsoid(obstacle_center, self.obstacle_shapes[obstacle_index], robot_center, initial_multiplier=initial, eigenvalues=self.obstacle_eigenvalues[obstacle_index], rotation=self.obstacle_rotations[obstacle_index])
        self._multiplier_cache[cache_key] = closest.multiplier
        distance = float(np.linalg.norm(closest.surface_point - robot_center))
        offset_radius = float(self.obstacle_offsets[obstacle_index])
        expanded_surface = closest.surface_point - offset_radius * closest.normal
        return ProtocolSeparatingPlane(robot_index=robot_index, obstacle_index=obstacle_index, proxy_id=proxy_id, normal_to_obstacle=closest.normal, surface_point=expanded_surface, offset=float(closest.normal @ expanded_surface), clearance=float(distance - offset_radius - robot_radius - self.safety_margin), newton_iterations=int(closest.newton_iterations), bisection_iterations=int(closest.bisection_iterations), closest_point_residual=float(closest.residual))

    def _ordering_lower_bound(self, robot_center: np.ndarray, robot_radius: float, candidates: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(self.obstacle_centers[candidates] - robot_center, axis=1)
        if self.representation == 'sphere':
            extent = self.obstacle_radii[candidates] + self.obstacle_offsets[candidates]
        else:
            extent = np.sqrt(np.maximum(np.max(self.obstacle_eigenvalues[candidates], axis=1), 0.0)) + self.obstacle_offsets[candidates]
            if self.obstacle_uncertainty_shapes is not None:
                extent += np.sqrt(np.maximum(np.max(self.obstacle_uncertainty_eigenvalues[candidates], axis=1), 0.0))
        return distances - extent - float(robot_radius) - self.safety_margin

    def _minimum_support(self, indices: np.ndarray, normal: np.ndarray) -> np.ndarray:
        if self.representation == 'sphere':
            return self.obstacle_centers[indices] @ normal - self.obstacle_radii[indices] - self.obstacle_offsets[indices]
        extents = np.sqrt(np.maximum(np.einsum('i,nij,j->n', normal, self.obstacle_shapes[indices], normal), 0.0))
        if self.obstacle_uncertainty_shapes is not None:
            extents += np.sqrt(np.maximum(np.einsum('i,nij,j->n', normal, self.obstacle_uncertainty_shapes[indices], normal), 0.0))
        return self.obstacle_centers[indices] @ normal - extents - self.obstacle_offsets[indices]

    def _prefetch_directional_pairs(self, positions: np.ndarray, radii: np.ndarray) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]], float]:
        started = time.perf_counter()
        batches: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = {}
        robot_batches: list[np.ndarray] = []
        obstacle_batches: list[np.ndarray] = []
        initial_batches: list[np.ndarray] = []
        warm_hits: dict[int, int] = {}
        counts: dict[int, int] = {}
        queried = None
        if self.obstacle_index is not None and hasattr(self.obstacle_index, 'query_spheres'):
            queried = self.obstacle_index.query_spheres(positions, radii)
            if len(queried) != len(positions):
                raise RuntimeError('candidate batch returned the wrong row count')
        for robot_index, (center, radius) in enumerate(zip(positions, radii)):
            if queried is None:
                candidates = self._candidate_indices(center, float(radius))
            else:
                candidates = np.asarray(queried[robot_index], dtype=np.int64)
                if len(candidates) and (np.min(candidates) < 0 or np.max(candidates) >= len(self.obstacle_centers)):
                    raise RuntimeError('candidate batch returned an invalid proxy index')
                candidates = np.unique(candidates)
            if len(candidates):
                lower_bounds = self._ordering_lower_bound(center, float(radius), candidates)
                ordered = np.asarray(candidates[np.lexsort((candidates, lower_bounds))], dtype=np.int64)
            else:
                ordered = np.empty(0, dtype=np.int64)
            counts[robot_index] = int(len(ordered))
            initial = np.zeros((len(ordered), 3), dtype=float)
            local_warm = 0
            for local_index, obstacle_index in enumerate(ordered):
                key = (int(robot_index), int(self.proxy_ids[obstacle_index]))
                previous = self._normal_cache.get(key)
                if previous is not None:
                    initial[local_index] = previous
                    local_warm += 1
            warm_hits[robot_index] = local_warm
            if len(ordered):
                robot_batches.append(np.full(len(ordered), robot_index, dtype=np.int64))
                obstacle_batches.append(ordered)
                initial_batches.append(initial)
        if not obstacle_batches:
            empty_i = np.empty(0, dtype=np.int32)
            empty_f = np.empty(0, dtype=float)
            empty_n = np.empty((0, 3), dtype=float)
            for robot_index in range(len(positions)):
                batches[robot_index] = (np.empty(0, dtype=np.int64), empty_n, empty_i, empty_f, 0)
            return (batches, (time.perf_counter() - started) * 1000.0)
        robot_indices = np.concatenate(robot_batches)
        obstacle_indices = np.ascontiguousarray(np.concatenate(obstacle_batches), dtype=np.int32)
        initial = np.concatenate(initial_batches)
        pair_radii = np.asarray(radii, dtype=float)[robot_indices]
        robot_shapes = np.eye(3)[None, :, :] * (pair_radii * pair_radii)[:, None, None]
        normals, iterations, residuals = self._support_kernel.normals_sum_pairs_newton_warm(np.asarray(positions, dtype=float)[robot_indices], robot_shapes, self.obstacle_centers[obstacle_indices], self.obstacle_shapes[obstacle_indices], self.obstacle_uncertainty_shapes[obstacle_indices], initial, max_iterations=16)
        retry_mask = residuals > 1e-07
        if np.any(retry_mask):
            retry_normals, retry_iterations, retry_residuals = self._support_kernel.normals_sum_pairs_newton_warm(np.asarray(positions, dtype=float)[robot_indices[retry_mask]], robot_shapes[retry_mask], self.obstacle_centers[obstacle_indices[retry_mask]], self.obstacle_shapes[obstacle_indices[retry_mask]], self.obstacle_uncertainty_shapes[obstacle_indices[retry_mask]], np.zeros((int(np.count_nonzero(retry_mask)), 3), dtype=float), max_iterations=64)
            normals[retry_mask] = retry_normals
            iterations[retry_mask] += retry_iterations
            residuals[retry_mask] = retry_residuals
        if len(residuals) and float(np.max(residuals)) > 1e-07:
            raise RuntimeError('ellipsoid support normal failed the 1e-7 KKT residual gate')
        for robot_index, obstacle_index, normal in zip(robot_indices, obstacle_indices, normals):
            key = (int(robot_index), int(self.proxy_ids[obstacle_index]))
            self._normal_cache[key] = normal.copy()
        cursor = 0
        for robot_index in range(len(positions)):
            count = counts[robot_index]
            stop = cursor + count
            ordered = obstacle_indices[cursor:stop] if count else np.empty(0, dtype=np.int64)
            batches[robot_index] = (ordered, normals[cursor:stop], iterations[cursor:stop], residuals[cursor:stop], warm_hits[robot_index])
            cursor = stop
        return (batches, (time.perf_counter() - started) * 1000.0)

    def _prefetch_exact_closest_pairs(self, positions: np.ndarray, radii: np.ndarray) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]], float, float]:
        started = time.perf_counter()
        batches: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]] = {}
        robot_batches: list[np.ndarray] = []
        obstacle_batches: list[np.ndarray] = []
        initial_batches: list[np.ndarray] = []
        warm_hits: dict[int, int] = {}
        counts: dict[int, int] = {}
        lazy_mode = self.native_exact_lazy_prune and self.redundant_plane_pruning
        fused_mvt_mode = bool(lazy_mode and self.obstacle_index is not None and getattr(self.obstacle_index, 'simd', False) and hasattr(self.obstacle_index, 'native_handle') and hasattr(self._support_kernel, 'closest_prune_pairs_mvt_warm'))
        if fused_mvt_mode:
            active_groups, normals, surfaces, _multipliers, newton, bisection, residuals, computed, warm_start, obstacle_indices, group_offsets = self._support_kernel.closest_prune_pairs_mvt_warm(self.obstacle_index.native_handle, float(self.obstacle_index.query_padding), np.asarray(positions, dtype=float), np.asarray(radii, dtype=float), self.obstacle_centers, self.obstacle_eigenvalues, self.obstacle_rotations, self.obstacle_shapes, self.obstacle_offsets, self._multiplier_dense, contact_distance=self.contact_distance)
            raw_counts = np.diff(group_offsets)
            self._prefetched_raw_counts = {int(index): int(raw_counts[index]) for index in range(len(positions))}
            self._prefetched_closest_stats = {}
            for robot_index in range(len(positions)):
                begin = int(group_offsets[robot_index])
                end = int(group_offsets[robot_index + 1])
                local_active = np.asarray(active_groups[robot_index], dtype=np.int64)
                selected = begin + local_active
                local_computed = computed[begin:end]
                solved = np.flatnonzero(local_computed) + begin
                local_warm = int(np.count_nonzero(warm_start[begin:end] & local_computed))
                self._prefetched_closest_stats[robot_index] = (int(np.sum(newton[solved])) if len(solved) else 0, int(np.sum(bisection[solved])) if len(solved) else 0, float(np.max(residuals[solved])) if len(solved) else 0.0, local_warm)
                batches[robot_index] = (np.asarray(obstacle_indices[selected], dtype=np.int32), np.asarray(normals[selected], dtype=float), np.asarray(surfaces[selected], dtype=float), np.asarray(newton[selected], dtype=np.int32), np.asarray(bisection[selected], dtype=np.int32), np.asarray(residuals[selected], dtype=float), local_warm, np.arange(len(selected), dtype=np.int32))
            return (batches, (time.perf_counter() - started) * 1000.0, 0.0)
        queried = None
        if self.obstacle_index is not None and hasattr(self.obstacle_index, 'query_spheres'):
            queried = self.obstacle_index.query_spheres(positions, radii)
            if len(queried) != len(positions):
                raise RuntimeError('candidate batch returned the wrong row count')
        for robot_index, (center, radius) in enumerate(zip(positions, radii)):
            if queried is None:
                candidates = self._candidate_indices(center, float(radius))
            else:
                candidates = np.asarray(queried[robot_index], dtype=np.int64)
                if len(candidates) and (np.min(candidates) < 0 or np.max(candidates) >= len(self.obstacle_centers)):
                    raise RuntimeError('candidate batch returned an invalid proxy index')
                candidates = np.unique(candidates)
            if len(candidates) and (not lazy_mode):
                lower_bounds = self._ordering_lower_bound(center, float(radius), candidates)
                ordered = np.asarray(candidates[np.lexsort((candidates, lower_bounds))], dtype=np.int64)
            elif not len(candidates):
                ordered = np.empty(0, dtype=np.int64)
            else:
                ordered = np.asarray(candidates, dtype=np.int64)
            counts[robot_index] = int(len(ordered))
            if not lazy_mode:
                initial = self._multiplier_dense[robot_index, ordered].copy()
                local_warm = int(np.count_nonzero(np.isfinite(initial)))
                warm_hits[robot_index] = local_warm
            if len(ordered):
                if not lazy_mode:
                    robot_batches.append(np.full(len(ordered), robot_index, dtype=np.int64))
                obstacle_batches.append(ordered)
                if not lazy_mode:
                    initial_batches.append(initial)
        if not obstacle_batches:
            empty_i = np.empty(0, dtype=np.int32)
            empty_f = np.empty(0, dtype=float)
            empty_n = np.empty((0, 3), dtype=float)
            for robot_index in range(len(positions)):
                batches[robot_index] = (np.empty(0, dtype=np.int64), empty_n, empty_n, empty_i, empty_i, empty_f, 0, empty_i)
            return (batches, (time.perf_counter() - started) * 1000.0, 0.0)
        robot_indices = np.concatenate(robot_batches) if robot_batches else np.empty(0, dtype=np.int64)
        obstacle_indices = np.concatenate(obstacle_batches)
        initial = np.concatenate(initial_batches) if initial_batches else np.empty(0, dtype=float)
        group_offsets = np.r_[0, np.cumsum([counts[index] for index in range(len(positions))], dtype=np.int64)].astype(np.int32)
        self._prefetched_raw_counts = {int(index): int(counts[index]) for index in range(len(positions))}
        self._prefetched_closest_stats = {}
        if self.native_exact_lazy_prune and self.redundant_plane_pruning:
            active_groups, normals, surfaces, multipliers, newton, bisection, residuals, computed, warm_start, sorted_obstacle_indices = self._support_kernel.closest_prune_pairs_warm(np.asarray(positions, dtype=float), np.asarray(radii, dtype=float), obstacle_indices, self.obstacle_centers, self.obstacle_eigenvalues, self.obstacle_rotations, self.obstacle_shapes, self.obstacle_offsets, self._multiplier_dense, group_offsets, contact_distance=self.contact_distance)
            obstacle_indices = np.asarray(sorted_obstacle_indices, dtype=np.int32)
            for robot_index in range(len(positions)):
                begin = int(group_offsets[robot_index])
                end = int(group_offsets[robot_index + 1])
                local_active = np.asarray(active_groups[robot_index], dtype=np.int64)
                selected = begin + local_active
                local_computed = computed[begin:end]
                solved = np.flatnonzero(local_computed) + begin
                local_warm = int(np.count_nonzero(warm_start[begin:end] & local_computed))
                self._prefetched_closest_stats[robot_index] = (int(np.sum(newton[solved])) if len(solved) else 0, int(np.sum(bisection[solved])) if len(solved) else 0, float(np.max(residuals[solved])) if len(solved) else 0.0, local_warm)
                batches[robot_index] = (obstacle_indices[selected], normals[selected], surfaces[selected], newton[selected], bisection[selected], residuals[selected], local_warm, np.arange(len(selected), dtype=np.int32))
            return (batches, (time.perf_counter() - started) * 1000.0, 0.0)
        normals, surfaces, multipliers, newton, bisection, residuals = self._support_kernel.closest_points_pairs_warm(np.asarray(positions, dtype=float)[robot_indices], self.obstacle_centers[obstacle_indices], self.obstacle_eigenvalues[obstacle_indices], self.obstacle_rotations[obstacle_indices], initial)
        self._multiplier_dense[robot_indices, obstacle_indices] = multipliers
        prune_started = time.perf_counter()
        if self.redundant_plane_pruning:
            pair_offsets = self.obstacle_offsets[obstacle_indices]
            expanded_surfaces = surfaces - pair_offsets[:, None] * normals
            plane_offsets = np.einsum('ni,ni->n', normals, expanded_surfaces)
            grouped_active = self._support_kernel.prune_planes_grouped(self.obstacle_centers[obstacle_indices], self.obstacle_shapes[obstacle_indices], pair_offsets, normals, plane_offsets, group_offsets)
        else:
            grouped_active = [np.arange(group_offsets[index + 1] - group_offsets[index], dtype=np.int32) for index in range(len(positions))]
        grouped_prune_ms = (time.perf_counter() - prune_started) * 1000.0
        cursor = 0
        for robot_index in range(len(positions)):
            count = counts[robot_index]
            stop = cursor + count
            ordered = obstacle_indices[cursor:stop] if count else np.empty(0, dtype=np.int64)
            batches[robot_index] = (ordered, normals[cursor:stop], surfaces[cursor:stop], newton[cursor:stop], bisection[cursor:stop], residuals[cursor:stop], warm_hits[robot_index], grouped_active[robot_index])
            cursor = stop
        return (batches, (time.perf_counter() - started) * 1000.0, grouped_prune_ms)

    def _prefetch_sphere_pairs(self, positions: np.ndarray, radii: np.ndarray) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]], float, float]:
        if self.obstacle_index is None or not getattr(self.obstacle_index, 'simd', False) or (not hasattr(self.obstacle_index, 'native_handle')) or (not hasattr(self._support_kernel, 'sphere_prune_pairs_mvt')):
            raise RuntimeError('native sphere fusion requires a SIMD MVT handle')
        started = time.perf_counter()
        active_groups, normals, surfaces, obstacle_indices, group_offsets = self._support_kernel.sphere_prune_pairs_mvt(self.obstacle_index.native_handle, float(self.obstacle_index.query_padding), np.asarray(positions, dtype=float), np.asarray(radii, dtype=float), self.obstacle_centers, self.obstacle_radii, self.obstacle_offsets, contact_distance=self.contact_distance)
        raw_counts = np.diff(group_offsets)
        self._prefetched_raw_counts = {int(index): int(raw_counts[index]) for index in range(len(positions))}
        self._prefetched_closest_stats = {}
        batches = {}
        for robot_index in range(len(positions)):
            begin = int(group_offsets[robot_index])
            local_active = np.asarray(active_groups[robot_index], dtype=np.int64)
            selected = begin + local_active
            active_count = len(selected)
            batches[robot_index] = (np.asarray(obstacle_indices[selected], dtype=np.int32), np.asarray(normals[selected], dtype=float), np.asarray(surfaces[selected], dtype=float), np.zeros(active_count, dtype=np.int32), np.zeros(active_count, dtype=np.int32), np.zeros(active_count, dtype=float), 0, np.arange(active_count, dtype=np.int32))
            self._prefetched_closest_stats[robot_index] = (0, 0, 0.0, 0)
        return (batches, (time.perf_counter() - started) * 1000.0, 0.0)

    def active_planes_for_robot_sphere(self, robot_index: int, center: np.ndarray, radius: float, directional_batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None=None, exact_batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray] | None=None) -> tuple[list[ProtocolSeparatingPlane], int, int, float, float]:
        geometry_started = time.perf_counter()
        candidates = exact_batch[0] if exact_batch is not None else self._candidate_indices(center, radius) if directional_batch is None else directional_batch[0]
        candidate_count = int(self._prefetched_raw_counts.get(robot_index, len(candidates))) if exact_batch is not None and self.native_exact_lazy_prune else len(candidates)
        self._last_support_batch_stats = (0, 0.0, 0)
        self._last_closest_batch_stats = (0, 0, 0.0, 0, False)
        if not candidate_count:
            elapsed = (time.perf_counter() - geometry_started) * 1000.0
            return ([], 0, 0, elapsed, 0.0)
        if directional_batch is None and exact_batch is None:
            lower_bounds = self._ordering_lower_bound(center, radius, candidates)
            remaining = list(candidates[np.lexsort((candidates, lower_bounds))])
        else:
            remaining = list(candidates)
        if exact_batch is not None and (self.representation == 'sphere' or (self.representation == 'ellipsoid' and self.obstacle_uncertainty_shapes is None and self.native_exact_batch)):
            if exact_batch is None:
                ordered = np.asarray(remaining, dtype=np.int64)
                initial = np.full(len(ordered), np.nan, dtype=float)
                warm_hits = 0
                for local_index, obstacle_index in enumerate(ordered):
                    key = (int(robot_index), int(self.proxy_ids[obstacle_index]))
                    previous = self._multiplier_cache.get(key)
                    if previous is not None:
                        initial[local_index] = previous
                        warm_hits += 1
                normals, surfaces, multipliers, newton_iterations, bisection_iterations, residuals = self._support_kernel.closest_points_warm(center, self.obstacle_centers[ordered], self.obstacle_eigenvalues[ordered], self.obstacle_rotations[ordered], initial)
                for obstacle_index, multiplier in zip(ordered, multipliers):
                    key = (int(robot_index), int(self.proxy_ids[obstacle_index]))
                    self._multiplier_cache[key] = float(multiplier)
            else:
                ordered, normals, surfaces, newton_iterations, bisection_iterations, residuals, warm_hits, grouped_active = exact_batch
            offsets = self.obstacle_offsets[ordered]
            expanded_surfaces = surfaces - offsets[:, None] * normals
            plane_offsets = np.einsum('ni,ni->n', normals, expanded_surfaces)
            clearances = np.linalg.norm(surfaces - center[None, :], axis=1) - offsets - float(radius) - self.safety_margin
            prune_started = time.perf_counter()
            if exact_batch is not None:
                active_local = list(map(int, grouped_active))
                active_set = set(active_local)
                for contact_local in np.flatnonzero(clearances + self.safety_margin <= self.contact_distance):
                    if int(contact_local) not in active_set:
                        active_local.append(int(contact_local))
                        active_set.add(int(contact_local))
            elif self.redundant_plane_pruning:
                active_local = list(map(int, self._support_kernel.prune_planes(self.obstacle_centers[ordered], self.obstacle_shapes[ordered], offsets, normals, plane_offsets)))
                active_set = set(active_local)
                for contact_local in np.flatnonzero(clearances + self.safety_margin <= self.contact_distance):
                    if int(contact_local) not in active_set:
                        active_local.append(int(contact_local))
                        active_set.add(int(contact_local))
            else:
                active_local = list(range(len(ordered)))
            prune_ms = (time.perf_counter() - prune_started) * 1000.0
            prefetched_stats = self._prefetched_closest_stats.get(robot_index) if exact_batch is not None and self.native_exact_lazy_prune else None
            self._last_closest_batch_stats = (int(np.sum(newton_iterations)), int(np.sum(bisection_iterations)), float(np.max(residuals)) if len(residuals) else 0.0, int(warm_hits), True) if prefetched_stats is None else (int(prefetched_stats[0]), int(prefetched_stats[1]), float(prefetched_stats[2]), int(prefetched_stats[3]), True)
            active = [ProtocolSeparatingPlane(robot_index=robot_index, obstacle_index=int(ordered[index]), proxy_id=int(self.proxy_ids[ordered[index]]), normal_to_obstacle=normals[index], surface_point=expanded_surfaces[index], offset=float(plane_offsets[index]), clearance=float(clearances[index]), newton_iterations=int(newton_iterations[index]), bisection_iterations=int(bisection_iterations[index]), closest_point_residual=float(residuals[index])) for index in active_local]
            elapsed = (time.perf_counter() - geometry_started) * 1000.0
            return (active, candidate_count, candidate_count - len(active), elapsed, min(prune_ms, elapsed))
        if self.representation == 'ellipsoid' and self.obstacle_uncertainty_shapes is not None:
            if directional_batch is None:
                ordered = np.asarray(remaining, dtype=np.int64)
                initial = np.zeros((len(ordered), 3), dtype=float)
                warm_hits = 0
                for local_index, obstacle_index in enumerate(ordered):
                    key = (int(robot_index), int(self.proxy_ids[obstacle_index]))
                    previous = self._normal_cache.get(key)
                    if previous is not None:
                        initial[local_index] = previous
                        warm_hits += 1
                normals, iterations, residuals = self._support_kernel.normals_sum_newton_warm(center, np.eye(3) * float(radius) ** 2, self.obstacle_centers[ordered], self.obstacle_shapes[ordered], self.obstacle_uncertainty_shapes[ordered], initial, max_iterations=16)
                retry_mask = residuals > 1e-07
                if np.any(retry_mask):
                    retry_normals, retry_iterations, retry_residuals = self._support_kernel.normals_sum_newton_warm(center, np.eye(3) * float(radius) ** 2, self.obstacle_centers[ordered[retry_mask]], self.obstacle_shapes[ordered[retry_mask]], self.obstacle_uncertainty_shapes[ordered[retry_mask]], np.zeros((int(np.count_nonzero(retry_mask)), 3), dtype=float), max_iterations=64)
                    normals[retry_mask] = retry_normals
                    iterations[retry_mask] += retry_iterations
                    residuals[retry_mask] = retry_residuals
                if len(residuals) and float(np.max(residuals)) > 1e-07:
                    raise RuntimeError('ellipsoid support normal failed the 1e-7 KKT residual gate')
                proxy_ids = self.proxy_ids[ordered]
                for proxy_id, normal in zip(proxy_ids, normals):
                    self._normal_cache[int(robot_index), int(proxy_id)] = normal.copy()
            else:
                ordered, normals, iterations, residuals, warm_hits = directional_batch
                proxy_ids = self.proxy_ids[ordered]
            obstacle_extent = np.sqrt(np.maximum(np.einsum('ni,nij,nj->n', normals, self.obstacle_shapes[ordered], normals), 0.0))
            uncertainty_extent = np.sqrt(np.maximum(np.einsum('ni,nij,nj->n', normals, self.obstacle_uncertainty_shapes[ordered], normals), 0.0))
            qn = np.einsum('nij,nj->ni', self.obstacle_shapes[ordered], normals)
            un = np.einsum('nij,nj->ni', self.obstacle_uncertainty_shapes[ordered], normals)
            offset_radii = self.obstacle_offsets[ordered]
            surfaces = self.obstacle_centers[ordered] - qn / np.maximum(obstacle_extent[:, None], 1e-12) - un / np.maximum(uncertainty_extent[:, None], 1e-12) - offset_radii[:, None] * normals
            plane_offsets = np.einsum('ni,ni->n', normals, surfaces)
            clearances = np.einsum('ni,ni->n', normals, self.obstacle_centers[ordered] - center) - radius - obstacle_extent - uncertainty_extent - offset_radii - self.safety_margin
            self._last_support_batch_stats = (int(np.sum(iterations)), float(np.max(residuals)) if len(residuals) else 0.0, warm_hits)
            prune_ms = 0.0
            if self.redundant_plane_pruning:
                prune_started = time.perf_counter()
                active_local = list(map(int, self._support_kernel.prune_planes_sum(self.obstacle_centers[ordered], self.obstacle_shapes[ordered], self.obstacle_uncertainty_shapes[ordered], offset_radii, normals, plane_offsets)))
                prune_ms = (time.perf_counter() - prune_started) * 1000.0
                active_set = set(active_local)
                for contact_local in np.flatnonzero(clearances + self.safety_margin <= self.contact_distance):
                    if int(contact_local) not in active_set:
                        active_local.append(int(contact_local))
                        active_set.add(int(contact_local))
            else:
                active_local = list(range(len(ordered)))
            active = [ProtocolSeparatingPlane(robot_index=robot_index, obstacle_index=int(ordered[index]), proxy_id=int(proxy_ids[index]), normal_to_obstacle=normals[index], surface_point=surfaces[index], offset=float(plane_offsets[index]), clearance=float(clearances[index]), support_iterations=int(iterations[index]), support_residual=float(residuals[index]), directional_support=True) for index in active_local]
            elapsed = (time.perf_counter() - geometry_started) * 1000.0
            return (active, candidate_count, candidate_count - len(active), elapsed, min(prune_ms, elapsed))
        active: list[ProtocolSeparatingPlane] = []
        if not self.redundant_plane_pruning:
            active = [self.separating_plane(robot_index, center, radius, int(index)) for index in remaining]
            elapsed = (time.perf_counter() - geometry_started) * 1000.0
            return (active, candidate_count, 0, elapsed, 0.0)
        tolerance = 1e-10
        prune_ms = 0.0
        while remaining:
            kept = int(remaining[0])
            plane = self.separating_plane(robot_index, center, radius, kept)
            active.append(plane)
            indices = np.asarray(remaining, dtype=np.int64)
            prune_started = time.perf_counter()
            behind = self._minimum_support(indices, plane.normal_to_obstacle) >= plane.offset - tolerance
            prune_ms += (time.perf_counter() - prune_started) * 1000.0
            remaining = [int(index) for index, remove in zip(indices, behind) if not bool(remove)]
        elapsed = (time.perf_counter() - geometry_started) * 1000.0
        return (active, candidate_count, candidate_count - len(active), elapsed, min(prune_ms, elapsed))

    def _append_workspace_constraints(self, rows: list[np.ndarray], lower: list[float], upper: list[float], position: np.ndarray, jacobian: np.ndarray, radius: float) -> int:
        for normal, offset in zip(self.workspace_A, self.workspace_b):
            rows.append(normal @ jacobian)
            lower.append(-np.inf)
            upper.append(float((offset - radius - normal @ position) / self.dt))
        return len(self.workspace_A)

    @staticmethod
    def _rows_hash(matrix: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> str:
        digest = hashlib.sha256()
        for array in (matrix, lower, upper):
            rounded = np.ascontiguousarray(np.round(array, 12))
            digest.update(memoryview(rounded).cast('B'))
        return digest.hexdigest()

    @staticmethod
    def _remap_dual_by_row_identity(previous_identity: tuple[tuple[str, int, int], ...] | None, previous_dual: np.ndarray | None, current_identity: tuple[tuple[str, int, int], ...], capacity: int) -> tuple[np.ndarray, int]:
        mapped = np.zeros(int(capacity), dtype=float)
        if previous_identity is None or previous_dual is None:
            return (mapped, 0)
        previous_row_index = {identity: index for index, identity in enumerate(previous_identity)}
        reused = 0
        for current_index, identity in enumerate(current_identity):
            previous_index = previous_row_index.get(identity)
            if previous_index is not None and previous_index < len(previous_dual):
                mapped[current_index] = previous_dual[previous_index]
                reused += 1
        return (mapped, reused)

    def _persistent_solver_update(self, hessian: np.ndarray, gradient: np.ndarray, matrix: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> osqp.OSQP:
        row_count = len(matrix)
        if row_count <= self.osqp_adaptive_row_threshold:
            current_rho = 0.1
            current_adaptive = True
            current_mode = 'adaptive_low_rows'
        else:
            current_rho = self.osqp_rho
            current_adaptive = self.osqp_adaptive_rho
            current_mode = 'fixed_dense_rows'
        mode_changed = current_mode != self._current_osqp_mode
        self._current_osqp_rho = current_rho
        self._current_osqp_adaptive = current_adaptive
        self._current_osqp_mode = current_mode
        if mode_changed and self._qp_solver is not None:
            self._qp_solver = None
            self._qp_A_template = None
            self._qp_P_template = None
            self._qp_row_capacity = 0
        if row_count > self._qp_row_capacity:
            block = 256
            capacity = max(self.qp_workspace_min_rows, block * ((row_count + block - 1) // block))
            a_template = sparse.csc_matrix(np.ones((capacity, self.nv), dtype=float))
            p_template = sparse.triu(sparse.csc_matrix(np.ones((self.nv, self.nv), dtype=float)), format='csc')
            self._qp_solver = osqp.OSQP()
            self._qp_A_template = a_template
            self._qp_P_template = p_template
            self._qp_row_capacity = capacity
            padded = np.zeros((capacity, self.nv), dtype=float)
            padded[:row_count] = matrix
            a_values = np.ascontiguousarray(padded.T).reshape(-1)
            p_values = np.concatenate([hessian[p_template.indices[p_template.indptr[column]:p_template.indptr[column + 1]], column] for column in range(self.nv)])
            lo = np.full(capacity, -np.inf, dtype=float)
            hi = np.full(capacity, np.inf, dtype=float)
            lo[:row_count] = lower
            hi[:row_count] = upper
            a_template.data[:] = a_values
            p_template.data[:] = p_values
            self._qp_solver.setup(P=p_template, q=gradient, A=a_template, l=lo, u=hi, verbose=False, polishing=False, warm_starting=True, eps_abs=self.osqp_absolute_tolerance, eps_rel=self.osqp_relative_tolerance, max_iter=self.osqp_max_iterations, rho=current_rho, adaptive_rho=current_adaptive)
        else:
            if self._qp_solver is None or self._qp_A_template is None or self._qp_P_template is None:
                raise AssertionError('persistent OSQP workspace lost its pattern')
            padded = np.zeros((self._qp_row_capacity, self.nv), dtype=float)
            padded[:row_count] = matrix
            a_values = np.ascontiguousarray(padded.T).reshape(-1)
            p_template = self._qp_P_template
            p_values = np.concatenate([hessian[p_template.indices[p_template.indptr[column]:p_template.indptr[column + 1]], column] for column in range(self.nv)])
            lo = np.full(self._qp_row_capacity, -np.inf, dtype=float)
            hi = np.full(self._qp_row_capacity, np.inf, dtype=float)
            lo[:row_count] = lower
            hi[:row_count] = upper
            self._qp_solver.update(Px=p_values, q=gradient, Ax=a_values, l=lo, u=hi)
        return self._qp_solver

    def solve(self, target: np.ndarray) -> tuple[np.ndarray, ProtocolStepMetrics]:
        total_started = time.perf_counter()
        ee, ee_jacobian, desired_velocity = self.task_feedback(target)
        positions, jacobians_full, radii = certificate_world_state(self.model, self.data, self.robot_spheres)
        jacobians = jacobians_full[:, :, :self.nv]
        hessian = self.velocity_regularization * np.eye(self.nv)
        hessian += self.task_weight * (ee_jacobian.T @ ee_jacobian)
        gradient = -self.task_weight * (ee_jacobian.T @ desired_velocity)
        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []
        qp_full_row_identity: list[tuple[str, int, int]] = []
        qdot_lower, qdot_upper = self.joint_velocity_bounds()
        for joint in range(self.nv):
            row = np.zeros(self.nv)
            row[joint] = 1.0
            rows.append(row)
            lower.append(float(qdot_lower[joint]))
            upper.append(float(qdot_upper[joint]))
            qp_full_row_identity.append(('joint_velocity', joint, -1))
        pair_records: list[PairRecord] = []
        workspace_rows = 0
        candidate_pairs = 0
        redundant_removed = 0
        active_rows = 0
        normal_pairs = 0
        near_terms = 0
        contact_rows = 0
        newton_iterations = 0
        bisection_iterations = 0
        warm_hits = 0
        support_iterations = 0
        support_warm_hits = 0
        support_max_residual = 0.0
        max_residual = 0.0
        minimum_clearance = np.inf
        limiting = (-1, -1, -1)
        geometry_ms = 0.0
        prune_ms = 0.0
        exact_closest_batch = bool(self.representation == 'ellipsoid' and self.obstacle_uncertainty_shapes is None and self.native_exact_batch)
        exact_batches = None
        if exact_closest_batch and self.native_exact_pair_batch:
            exact_batches, geometry_ms, prune_ms = self._prefetch_exact_closest_pairs(positions, radii)
        elif self.representation == 'sphere' and self.native_exact_batch and self.native_exact_pair_batch and self.redundant_plane_pruning and (self.obstacle_index is not None) and getattr(self.obstacle_index, 'simd', False) and hasattr(self.obstacle_index, 'native_handle'):
            exact_batches, geometry_ms, prune_ms = self._prefetch_sphere_pairs(positions, radii)
        directional_batches = None
        if self.native_directional_pair_batch and self.representation == 'ellipsoid' and (self.obstacle_uncertainty_shapes is not None):
            directional_batches, geometry_ms = self._prefetch_directional_pairs(positions, radii)
        workspace_projected = np.einsum('wi,rij->rwj', self.workspace_A, jacobians)
        workspace_upper = (self.workspace_b[None, :] - radii[:, None] - positions @ self.workspace_A.T) / self.dt
        workspace_lower = [-np.inf] * len(self.workspace_A)
        near_projected_rows: list[np.ndarray] = []
        for robot_index, (position, jacobian, radius) in enumerate(zip(positions, jacobians, radii)):
            rows.extend(workspace_projected[robot_index])
            lower.extend(workspace_lower)
            upper.extend(map(float, workspace_upper[robot_index]))
            qp_full_row_identity.extend((('workspace', robot_index, workspace_index) for workspace_index in range(len(self.workspace_A))))
            workspace_rows += len(self.workspace_A)
            cache_before = frozenset() if exact_batches is not None else set(self._multiplier_cache)
            active, candidates, removed, local_geometry_ms, local_prune_ms = self.active_planes_for_robot_sphere(robot_index, position, float(radius), None if directional_batches is None else directional_batches[robot_index], None if exact_batches is None else exact_batches[robot_index])
            geometry_ms += local_geometry_ms
            prune_ms += local_prune_ms
            candidate_pairs += candidates
            redundant_removed += removed
            batch_iterations, batch_residual, batch_warm = self._last_support_batch_stats
            support_iterations += batch_iterations
            support_max_residual = max(support_max_residual, batch_residual)
            support_warm_hits += batch_warm
            closest_batch_newton, closest_batch_bisection, closest_batch_residual, closest_batch_warm, closest_batch_used = self._last_closest_batch_stats
            if closest_batch_used:
                newton_iterations += closest_batch_newton
                bisection_iterations += closest_batch_bisection
                max_residual = max(max_residual, closest_batch_residual)
                warm_hits += closest_batch_warm
            for plane in active:
                cache_key = (robot_index, plane.proxy_id)
                if not closest_batch_used and (not plane.directional_support) and (self.representation == 'ellipsoid') and (cache_key in cache_before):
                    warm_hits += 1
                if not closest_batch_used:
                    newton_iterations += plane.newton_iterations
                    bisection_iterations += plane.bisection_iterations
                    max_residual = max(max_residual, plane.closest_point_residual)
                if plane.clearance < minimum_clearance:
                    minimum_clearance = plane.clearance
                    limiting = (robot_index, plane.obstacle_index, plane.proxy_id)
                projected = plane.normal_to_obstacle @ jacobian
                rows.append(projected)
                lower.append(-np.inf)
                upper.append(float(plane.clearance / self.dt))
                qp_full_row_identity.append(('collision', robot_index, int(plane.proxy_id)))
                active_rows += 1
                surface_clearance = plane.clearance + self.safety_margin
                state = classify_clearance(surface_clearance, near_distance=self.near_distance, contact_distance=self.contact_distance)
                near = state is ObstacleState.NEAR
                contact = state is ObstacleState.CONTACT_RECOVERY
                if state is ObstacleState.NORMAL:
                    normal_pairs += 1
                elif near:
                    near_projected_rows.append(projected)
                    near_terms += 1
                else:
                    penetration = max(0.0, -surface_clearance)
                    repulsion = self.repulsive_speed * min(1.0, 0.25 + penetration / max(self.safety_margin, 1e-12))
                    rows.append(projected)
                    lower.append(-np.inf)
                    upper.append(-float(repulsion))
                    qp_full_row_identity.append(('contact_recovery', robot_index, int(plane.proxy_id)))
                    contact_rows += 1
                pair_records.append(PairRecord(robot_index=robot_index, obstacle_index=plane.obstacle_index, proxy_id=plane.proxy_id, surface_clearance=float(surface_clearance), clearance=float(plane.clearance), state=state.value, qp_collision_row=True, near_penalty=near, contact_recovery_row=contact))
        if near_projected_rows:
            near_matrix = np.asarray(near_projected_rows, dtype=float)
            hessian += self.near_weight * (near_matrix.T @ near_matrix)
        self.last_pair_records = tuple(pair_records)
        qp_full_row_identity_tuple = tuple(qp_full_row_identity)
        if len(qp_full_row_identity_tuple) != len(rows):
            raise RuntimeError('QP row identity count does not match the assembled QP rows')
        dense_matrix = np.ascontiguousarray(np.asarray(rows, dtype=np.float64))
        lower_array = np.ascontiguousarray(np.asarray(lower, dtype=np.float64))
        upper_array = np.ascontiguousarray(np.asarray(upper, dtype=np.float64))
        row_hash = self._rows_hash(dense_matrix, lower_array, upper_array)
        self.last_qp_arrays = (np.asarray(hessian, dtype=np.float64).copy(), np.asarray(gradient, dtype=np.float64).copy(), dense_matrix.copy(), lower_array.copy(), upper_array.copy())
        setup_started = time.perf_counter()
        if self.persistent_qp_workspace:
            solver = self._persistent_solver_update(hessian, gradient, dense_matrix, lower_array, upper_array)
        else:
            if len(dense_matrix) <= self.osqp_adaptive_row_threshold:
                current_rho = 0.1
                current_adaptive = True
                self._current_osqp_mode = 'adaptive_low_rows'
            else:
                current_rho = self.osqp_rho
                current_adaptive = self.osqp_adaptive_rho
                self._current_osqp_mode = 'fixed_dense_rows'
            self._current_osqp_rho = current_rho
            self._current_osqp_adaptive = current_adaptive
            solver = osqp.OSQP()
            solver.setup(P=sparse.triu(sparse.csc_matrix(hessian), format='csc'), q=gradient, A=sparse.csc_matrix(dense_matrix), l=lower_array, u=upper_array, verbose=False, polishing=False, warm_starting=True, eps_abs=self.osqp_absolute_tolerance, eps_rel=self.osqp_relative_tolerance, max_iter=self.osqp_max_iterations, rho=current_rho, adaptive_rho=current_adaptive)
        if self.persistent_qp_workspace:
            mapped_dual, dual_warm_start_rows = self._remap_dual_by_row_identity(self._previous_qp_full_row_identity, self._previous_qp_dual, qp_full_row_identity_tuple, self._qp_row_capacity)
            dual_warm_start_used = dual_warm_start_rows > 0
            solver.warm_start(x=self.previous_velocity, y=mapped_dual)
        elif np.any(self.previous_velocity):
            solver.warm_start(x=self.previous_velocity)
            dual_warm_start_used = False
            dual_warm_start_rows = 0
        else:
            dual_warm_start_used = False
            dual_warm_start_rows = 0
        setup_ms = (time.perf_counter() - setup_started) * 1000.0
        solve_started = time.perf_counter()
        result = solver.solve(raise_error=False)
        self.last_qp_solver_info = result.info
        solve_ms = (time.perf_counter() - solve_started) * 1000.0
        status = result.info.status.lower()
        if result.x is None or not status.startswith('solved'):
            qdot = np.zeros(self.nv)
        else:
            qdot = np.clip(np.asarray(result.x, dtype=float), qdot_lower, qdot_upper)
        self.previous_velocity = qdot.copy()
        self._previous_qp_full_row_identity = qp_full_row_identity_tuple
        result_dual = getattr(result, 'y', None)
        self._previous_qp_dual = None if result_dual is None else np.asarray(result_dual, dtype=float).copy()
        total_ms = (time.perf_counter() - total_started) * 1000.0
        raw_pairs = len(self.robot_spheres) * len(self.obstacle_centers)
        minimum = float(minimum_clearance) if np.isfinite(minimum_clearance) else float('inf')
        return (qdot, ProtocolStepMetrics(status=status, representation=self.representation, solve_ms=float(solve_ms), setup_ms=float(setup_ms), geometry_ms=float(geometry_ms), prune_ms=float(prune_ms), total_controller_ms=float(total_ms), ee_error=float(np.linalg.norm(np.asarray(target) - ee)), task_speed=float(np.linalg.norm(desired_velocity)), qdot_norm=float(np.linalg.norm(qdot)), min_clearance=minimum, raw_pairs=int(raw_pairs), broadphase_candidate_pairs=int(candidate_pairs), broadphase_rejected_pairs=int(raw_pairs - candidate_pairs), active_obstacle_rows=int(active_rows), redundant_proxies_removed=int(redundant_removed), normal_pairs=int(normal_pairs), near_penalty_terms=int(near_terms), contact_repulsion_rows=int(contact_rows), workspace_rows=int(workspace_rows), closest_point_newton_iterations=int(newton_iterations), closest_point_bisection_iterations=int(bisection_iterations), closest_point_max_residual=float(max_residual), multiplier_warm_start_hits=int(warm_hits), support_normal_iterations=int(support_iterations), support_normal_max_residual=float(support_max_residual), support_normal_warm_start_hits=int(support_warm_hits), qp_iterations=int(getattr(result.info, 'iter', 0)), qp_primal_residual=float(getattr(result.info, 'prim_res', float('inf'))), qp_dual_residual=float(getattr(result.info, 'dual_res', float('inf'))), qp_rho_mode=self._current_osqp_mode, qp_dual_warm_start_used=dual_warm_start_used, qp_dual_warm_start_rows=int(dual_warm_start_rows), qp_row_sha256=row_hash, limiting_robot_index=int(limiting[0]), limiting_obstacle_index=int(limiting[1]), limiting_proxy_id=int(limiting[2])))
