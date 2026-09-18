"""Extracted original v4.3 ellipsoid dependency; see docs/source_manifest.json."""
from __future__ import annotations
import ctypes
import numpy as np
from voxel_index import DLL

class NativeEllipsoidSupport:

    def __init__(self) -> None:
        if not DLL.exists():
            raise FileNotFoundError(f'native MVT DLL is missing; run build_native_mvt.py: {DLL}')
        self._lib = ctypes.CDLL(str(DLL))
        dp = ctypes.POINTER(ctypes.c_double)
        ip = ctypes.POINTER(ctypes.c_int32)
        self._lib.ellipsoid_set_pair_threads.argtypes = [ctypes.c_int32]
        self._lib.ellipsoid_set_pair_threads.restype = ctypes.c_int32
        self._lib.ellipsoid_get_pair_threads.argtypes = []
        self._lib.ellipsoid_get_pair_threads.restype = ctypes.c_int32
        self._lib.ellipsoid_set_pair_affinity_mask.argtypes = [ctypes.c_uint64]
        self._lib.ellipsoid_set_pair_affinity_mask.restype = ctypes.c_uint64
        self._lib.ellipsoid_get_pair_affinity_mask.argtypes = []
        self._lib.ellipsoid_get_pair_affinity_mask.restype = ctypes.c_uint64
        self._lib.ellipsoid_support_normals.argtypes = [dp, dp, dp, dp, ctypes.c_int32, ctypes.c_int32, dp, ip]
        self._lib.ellipsoid_support_normals.restype = ctypes.c_int32
        self._lib.ellipsoid_support_normals_sum.argtypes = [dp, dp, dp, dp, dp, ctypes.c_int32, ctypes.c_int32, dp, ip]
        self._lib.ellipsoid_support_normals_sum.restype = ctypes.c_int32
        self._lib.ellipsoid_support_normals_sum_warm.argtypes = [dp, dp, dp, dp, dp, dp, ctypes.c_int32, ctypes.c_int32, dp, ip, dp]
        self._lib.ellipsoid_support_normals_sum_warm.restype = ctypes.c_int32
        self._lib.ellipsoid_support_normals_sum_newton_warm.argtypes = [dp, dp, dp, dp, dp, dp, ctypes.c_int32, ctypes.c_int32, dp, ip, dp]
        self._lib.ellipsoid_support_normals_sum_newton_warm.restype = ctypes.c_int32
        self._lib.ellipsoid_support_normals_sum_pairs_warm.argtypes = [dp, dp, dp, dp, dp, dp, ctypes.c_int32, ctypes.c_int32, dp, ip, dp]
        self._lib.ellipsoid_support_normals_sum_pairs_warm.restype = ctypes.c_int32
        self._lib.ellipsoid_support_normals_sum_pairs_newton_warm.argtypes = [dp, dp, dp, dp, dp, dp, ctypes.c_int32, ctypes.c_int32, dp, ip, dp]
        self._lib.ellipsoid_support_normals_sum_pairs_newton_warm.restype = ctypes.c_int32
        self._lib.ellipsoid_closest_points_warm.argtypes = [dp, dp, dp, dp, dp, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_double, dp, dp, dp, ip, ip, dp]
        self._lib.ellipsoid_closest_points_warm.restype = ctypes.c_int32
        self._lib.ellipsoid_closest_points_pairs_warm.argtypes = [dp, dp, dp, dp, dp, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_double, dp, dp, dp, ip, ip, dp]
        self._lib.ellipsoid_closest_points_pairs_warm.restype = ctypes.c_int32
        self._lib.ellipsoid_prune_planes.argtypes = [dp, dp, dp, dp, dp, ctypes.c_int32, ip, ctypes.c_int32]
        self._lib.ellipsoid_prune_planes.restype = ctypes.c_int32
        self._lib.ellipsoid_prune_planes_grouped.argtypes = [dp, dp, dp, dp, dp, ip, ctypes.c_int32, ctypes.c_int32, ip, ip]
        self._lib.ellipsoid_prune_planes_grouped.restype = ctypes.c_int32
        self._lib.ellipsoid_closest_prune_pairs_warm.argtypes = [dp, dp, ip, ctypes.c_int32, dp, dp, dp, dp, dp, dp, ip, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_double, ctypes.c_double, dp, dp, dp, ip, ip, dp, ip, ip, ip, ip]
        self._lib.ellipsoid_closest_prune_pairs_warm.restype = ctypes.c_int32
        self._lib.mvt_ellipsoid_closest_prune_pairs_warm.argtypes = [ctypes.c_void_p, dp, dp, ctypes.c_double, ctypes.c_int32, ctypes.c_int32, ip, ip, ctypes.c_int32, dp, dp, dp, dp, dp, dp, ctypes.c_int32, ctypes.c_int32, ctypes.c_double, ctypes.c_double, dp, dp, dp, ip, ip, dp, ip, ip, ip, ip]
        self._lib.mvt_ellipsoid_closest_prune_pairs_warm.restype = ctypes.c_int32
        self._lib.mvt_sphere_prune_pairs.argtypes = [ctypes.c_void_p, dp, dp, ctypes.c_double, ctypes.c_int32, ctypes.c_int32, ip, ip, ctypes.c_int32, dp, dp, dp, ctypes.c_double, dp, dp, ip, ip]
        self._lib.mvt_sphere_prune_pairs.restype = ctypes.c_int32
        self._lib.ellipsoid_prune_planes_sum.argtypes = [dp, dp, dp, dp, dp, dp, ctypes.c_int32, ip, ctypes.c_int32]
        self._lib.ellipsoid_prune_planes_sum.restype = ctypes.c_int32
        self._fused_candidate_capacity = 0
        self._fused_candidate_buffers: dict[str, np.ndarray] = {}
        self._fused_sphere_capacity = 0
        self._fused_sphere_buffers: dict[str, np.ndarray] = {}

    def set_pair_threads(self, value: int) -> int:
        configured = int(self._lib.ellipsoid_set_pair_threads(int(value)))
        if configured != int(value):
            raise ValueError('native ellipsoid pair threads must be in [1, 64]')
        return configured

    @property
    def pair_threads(self) -> int:
        return int(self._lib.ellipsoid_get_pair_threads())

    def set_pair_affinity_mask(self, value: int) -> int:
        if value < 0 or value >= 1 << 64:
            raise ValueError('native ellipsoid affinity mask must fit uint64')
        return int(self._lib.ellipsoid_set_pair_affinity_mask(int(value)))

    @property
    def pair_affinity_mask(self) -> int:
        return int(self._lib.ellipsoid_get_pair_affinity_mask())

    def normals(self, robot_center: np.ndarray, robot_shape: np.ndarray, obstacle_centers: np.ndarray, obstacle_shapes: np.ndarray, *, max_iterations: int=24) -> tuple[np.ndarray, np.ndarray]:
        center = np.ascontiguousarray(robot_center, dtype=np.float64).reshape(3)
        shape = np.ascontiguousarray(robot_shape, dtype=np.float64).reshape(3, 3)
        centers = np.ascontiguousarray(obstacle_centers, dtype=np.float64).reshape(-1, 3)
        shapes = np.ascontiguousarray(obstacle_shapes, dtype=np.float64).reshape(-1, 3, 3)
        if len(centers) != len(shapes):
            raise ValueError('obstacle centers and shapes must have equal length')
        output = np.empty((len(centers), 3), dtype=np.float64)
        iterations = np.empty(len(centers), dtype=np.int32)
        dp = ctypes.POINTER(ctypes.c_double)
        count = self._lib.ellipsoid_support_normals(center.ctypes.data_as(dp), shape.ctypes.data_as(dp), centers.ctypes.data_as(dp), shapes.ctypes.data_as(dp), len(centers), int(max_iterations), output.ctypes.data_as(dp), iterations.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if count != len(centers):
            raise RuntimeError(f'native ellipsoid support failed with code {count}')
        return (output, iterations)

    def normals_sum(self, robot_center: np.ndarray, robot_shape: np.ndarray, obstacle_centers: np.ndarray, obstacle_shapes: np.ndarray, uncertainty_shapes: np.ndarray, *, max_iterations: int=24) -> tuple[np.ndarray, np.ndarray]:
        center = np.ascontiguousarray(robot_center, dtype=np.float64).reshape(3)
        shape = np.ascontiguousarray(robot_shape, dtype=np.float64).reshape(3, 3)
        centers = np.ascontiguousarray(obstacle_centers, dtype=np.float64).reshape(-1, 3)
        shapes = np.ascontiguousarray(obstacle_shapes, dtype=np.float64).reshape(-1, 3, 3)
        uncertainty = np.ascontiguousarray(uncertainty_shapes, dtype=np.float64).reshape(-1, 3, 3)
        if not len(centers) == len(shapes) == len(uncertainty):
            raise ValueError('support-sum arrays must have equal length')
        output = np.empty((len(centers), 3), dtype=np.float64)
        iterations = np.empty(len(centers), dtype=np.int32)
        dp = ctypes.POINTER(ctypes.c_double)
        count = self._lib.ellipsoid_support_normals_sum(center.ctypes.data_as(dp), shape.ctypes.data_as(dp), centers.ctypes.data_as(dp), shapes.ctypes.data_as(dp), uncertainty.ctypes.data_as(dp), len(centers), int(max_iterations), output.ctypes.data_as(dp), iterations.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if count != len(centers):
            raise RuntimeError(f'native ellipsoid support-sum failed with code {count}')
        return (output, iterations)

    def normals_sum_warm(self, robot_center: np.ndarray, robot_shape: np.ndarray, obstacle_centers: np.ndarray, obstacle_shapes: np.ndarray, uncertainty_shapes: np.ndarray, initial_normals: np.ndarray, *, max_iterations: int=32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = np.ascontiguousarray(robot_center, dtype=np.float64).reshape(3)
        shape = np.ascontiguousarray(robot_shape, dtype=np.float64).reshape(3, 3)
        centers = np.ascontiguousarray(obstacle_centers, dtype=np.float64).reshape(-1, 3)
        shapes = np.ascontiguousarray(obstacle_shapes, dtype=np.float64).reshape(-1, 3, 3)
        uncertainty = np.ascontiguousarray(uncertainty_shapes, dtype=np.float64).reshape(-1, 3, 3)
        initial = np.ascontiguousarray(initial_normals, dtype=np.float64).reshape(-1, 3)
        if not len(centers) == len(shapes) == len(uncertainty) == len(initial):
            raise ValueError('warm support-sum arrays must have equal length')
        output = np.empty((len(centers), 3), dtype=np.float64)
        iterations = np.empty(len(centers), dtype=np.int32)
        residuals = np.empty(len(centers), dtype=np.float64)
        dp = ctypes.POINTER(ctypes.c_double)
        count = self._lib.ellipsoid_support_normals_sum_warm(center.ctypes.data_as(dp), shape.ctypes.data_as(dp), centers.ctypes.data_as(dp), shapes.ctypes.data_as(dp), uncertainty.ctypes.data_as(dp), initial.ctypes.data_as(dp), len(centers), int(max_iterations), output.ctypes.data_as(dp), iterations.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), residuals.ctypes.data_as(dp))
        if count != len(centers):
            raise RuntimeError(f'native warm support-sum failed with code {count}')
        return (output, iterations, residuals)

    def closest_points_warm(self, point: np.ndarray, obstacle_centers: np.ndarray, eigenvalues: np.ndarray, rotations: np.ndarray, initial_multipliers: np.ndarray, *, maximum_newton_iterations: int=10, maximum_bisection_iterations: int=48, tolerance: float=1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        point = np.ascontiguousarray(point, dtype=np.float64).reshape(3)
        centers = np.ascontiguousarray(obstacle_centers, dtype=np.float64).reshape(-1, 3)
        values = np.ascontiguousarray(eigenvalues, dtype=np.float64).reshape(-1, 3)
        rotations = np.ascontiguousarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
        initial = np.ascontiguousarray(initial_multipliers, dtype=np.float64).reshape(-1)
        count = len(centers)
        if not len(values) == len(rotations) == len(initial) == count:
            raise ValueError('closest-point batches must have equal length')
        normals = np.empty((count, 3), dtype=np.float64)
        surfaces = np.empty((count, 3), dtype=np.float64)
        multipliers = np.empty(count, dtype=np.float64)
        newton = np.empty(count, dtype=np.int32)
        bisection = np.empty(count, dtype=np.int32)
        residuals = np.empty(count, dtype=np.float64)
        dp = ctypes.POINTER(ctypes.c_double)
        ip = ctypes.POINTER(ctypes.c_int32)
        written = self._lib.ellipsoid_closest_points_warm(point.ctypes.data_as(dp), centers.ctypes.data_as(dp), values.ctypes.data_as(dp), rotations.ctypes.data_as(dp), initial.ctypes.data_as(dp), count, int(maximum_newton_iterations), int(maximum_bisection_iterations), float(tolerance), normals.ctypes.data_as(dp), surfaces.ctypes.data_as(dp), multipliers.ctypes.data_as(dp), newton.ctypes.data_as(ip), bisection.ctypes.data_as(ip), residuals.ctypes.data_as(dp))
        if written != count:
            raise RuntimeError(f'native ellipsoid closest-point batch failed with code {written}')
        return (normals, surfaces, multipliers, newton, bisection, residuals)

    def closest_points_pairs_warm(self, points: np.ndarray, obstacle_centers: np.ndarray, eigenvalues: np.ndarray, rotations: np.ndarray, initial_multipliers: np.ndarray, *, maximum_newton_iterations: int=10, maximum_bisection_iterations: int=48, tolerance: float=1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        centers = np.ascontiguousarray(obstacle_centers, dtype=np.float64).reshape(-1, 3)
        values = np.ascontiguousarray(eigenvalues, dtype=np.float64).reshape(-1, 3)
        rotations = np.ascontiguousarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
        initial = np.ascontiguousarray(initial_multipliers, dtype=np.float64).reshape(-1)
        count = len(points)
        if not len(centers) == len(values) == len(rotations) == len(initial) == count:
            raise ValueError('pairwise closest-point arrays must have equal length')
        normals = np.empty((count, 3), dtype=np.float64)
        surfaces = np.empty((count, 3), dtype=np.float64)
        multipliers = np.empty(count, dtype=np.float64)
        newton = np.empty(count, dtype=np.int32)
        bisection = np.empty(count, dtype=np.int32)
        residuals = np.empty(count, dtype=np.float64)
        dp = ctypes.POINTER(ctypes.c_double)
        ip = ctypes.POINTER(ctypes.c_int32)
        written = self._lib.ellipsoid_closest_points_pairs_warm(points.ctypes.data_as(dp), centers.ctypes.data_as(dp), values.ctypes.data_as(dp), rotations.ctypes.data_as(dp), initial.ctypes.data_as(dp), count, int(maximum_newton_iterations), int(maximum_bisection_iterations), float(tolerance), normals.ctypes.data_as(dp), surfaces.ctypes.data_as(dp), multipliers.ctypes.data_as(dp), newton.ctypes.data_as(ip), bisection.ctypes.data_as(ip), residuals.ctypes.data_as(dp))
        if written != count:
            raise RuntimeError(f'native pairwise closest-point batch failed with code {written}')
        return (normals, surfaces, multipliers, newton, bisection, residuals)

    def prune_planes_grouped(self, obstacle_centers: np.ndarray, obstacle_shapes: np.ndarray, obstacle_offsets: np.ndarray, normals: np.ndarray, plane_offsets: np.ndarray, group_offsets: np.ndarray) -> list[np.ndarray]:
        centers = np.ascontiguousarray(obstacle_centers, dtype=np.float64).reshape(-1, 3)
        shapes = np.ascontiguousarray(obstacle_shapes, dtype=np.float64).reshape(-1, 3, 3)
        offsets = np.ascontiguousarray(obstacle_offsets, dtype=np.float64).reshape(-1)
        normals = np.ascontiguousarray(normals, dtype=np.float64).reshape(-1, 3)
        planes = np.ascontiguousarray(plane_offsets, dtype=np.float64).reshape(-1)
        groups = np.ascontiguousarray(group_offsets, dtype=np.int32).reshape(-1)
        count = len(centers)
        if not len(shapes) == len(offsets) == len(normals) == len(planes) == count:
            raise ValueError('grouped prune candidate arrays must have equal length')
        if len(groups) == 0 or groups[0] != 0 or groups[-1] != count:
            raise ValueError('group offsets must start at zero and end at count')
        if np.any(np.diff(groups) < 0):
            raise ValueError('group offsets must be nondecreasing')
        group_count = len(groups) - 1
        active = np.empty(count, dtype=np.int32)
        active_counts = np.empty(group_count, dtype=np.int32)
        dp = ctypes.POINTER(ctypes.c_double)
        ip = ctypes.POINTER(ctypes.c_int32)
        written = self._lib.ellipsoid_prune_planes_grouped(centers.ctypes.data_as(dp), shapes.ctypes.data_as(dp), offsets.ctypes.data_as(dp), normals.ctypes.data_as(dp), planes.ctypes.data_as(dp), groups.ctypes.data_as(ip), group_count, count, active.ctypes.data_as(ip), active_counts.ctypes.data_as(ip))
        if written != group_count:
            raise RuntimeError(f'native grouped plane pruning failed with code {written}')
        return [active[groups[index]:groups[index] + active_counts[index]].copy() for index in range(group_count)]

    def closest_prune_pairs_warm(self, robot_centers: np.ndarray, robot_radii: np.ndarray, obstacle_indices: np.ndarray, obstacle_centers: np.ndarray, eigenvalues: np.ndarray, rotations: np.ndarray, obstacle_shapes: np.ndarray, obstacle_offsets: np.ndarray, multiplier_dense: np.ndarray, group_offsets: np.ndarray, *, contact_distance: float=0.0, maximum_newton_iterations: int=10, maximum_bisection_iterations: int=48, tolerance: float=1e-12) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        robot_centers = np.ascontiguousarray(robot_centers, dtype=np.float64).reshape(-1, 3)
        radii = np.ascontiguousarray(robot_radii, dtype=np.float64).reshape(-1)
        obstacle_indices = np.ascontiguousarray(obstacle_indices, dtype=np.int32).reshape(-1)
        centers = np.ascontiguousarray(obstacle_centers, dtype=np.float64).reshape(-1, 3)
        values = np.ascontiguousarray(eigenvalues, dtype=np.float64).reshape(-1, 3)
        rotations = np.ascontiguousarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
        shapes = np.ascontiguousarray(obstacle_shapes, dtype=np.float64).reshape(-1, 3, 3)
        offsets = np.ascontiguousarray(obstacle_offsets, dtype=np.float64).reshape(-1)
        multipliers_dense = np.asarray(multiplier_dense)
        if multipliers_dense.dtype != np.float64 or not multipliers_dense.flags.c_contiguous:
            raise ValueError('multiplier_dense must be contiguous float64')
        multipliers_dense = multipliers_dense.reshape(len(robot_centers), -1)
        groups = np.ascontiguousarray(group_offsets, dtype=np.int32).reshape(-1)
        count = len(obstacle_indices)
        if not len(centers) == len(values) == len(rotations) == len(shapes) == len(offsets):
            raise ValueError('global obstacle arrays must have equal length')
        if len(radii) != len(robot_centers):
            raise ValueError('one radius is required per robot center')
        if multipliers_dense.shape != (len(robot_centers), len(centers)):
            raise ValueError('multiplier_dense shape must be robot x obstacle')
        if len(groups) == 0 or groups[0] != 0 or groups[-1] != count:
            raise ValueError('group offsets must start at zero and end at count')
        if np.any(np.diff(groups) < 0):
            raise ValueError('group offsets must be nondecreasing')
        group_count = len(groups) - 1
        normals = np.empty((count, 3), dtype=np.float64)
        surfaces = np.empty((count, 3), dtype=np.float64)
        multipliers = np.empty(count, dtype=np.float64)
        newton = np.empty(count, dtype=np.int32)
        bisection = np.empty(count, dtype=np.int32)
        residuals = np.empty(count, dtype=np.float64)
        computed = np.zeros(count, dtype=np.int32)
        warm_start = np.zeros(count, dtype=np.int32)
        active = np.empty(count, dtype=np.int32)
        active_counts = np.empty(group_count, dtype=np.int32)
        dp = ctypes.POINTER(ctypes.c_double)
        ip = ctypes.POINTER(ctypes.c_int32)
        written = self._lib.ellipsoid_closest_prune_pairs_warm(robot_centers.ctypes.data_as(dp), radii.ctypes.data_as(dp), obstacle_indices.ctypes.data_as(ip), len(centers), centers.ctypes.data_as(dp), values.ctypes.data_as(dp), rotations.ctypes.data_as(dp), shapes.ctypes.data_as(dp), offsets.ctypes.data_as(dp), multipliers_dense.ctypes.data_as(dp), groups.ctypes.data_as(ip), group_count, count, int(maximum_newton_iterations), int(maximum_bisection_iterations), float(tolerance), float(contact_distance), normals.ctypes.data_as(dp), surfaces.ctypes.data_as(dp), multipliers.ctypes.data_as(dp), newton.ctypes.data_as(ip), bisection.ctypes.data_as(ip), residuals.ctypes.data_as(dp), computed.ctypes.data_as(ip), warm_start.ctypes.data_as(ip), active.ctypes.data_as(ip), active_counts.ctypes.data_as(ip))
        if written != group_count:
            raise RuntimeError(f'native lazy closest-prune failed with code {written}')
        active_groups = [active[groups[index]:groups[index] + active_counts[index]].copy() for index in range(group_count)]
        return (active_groups, normals, surfaces, multipliers, newton, bisection, residuals, computed.astype(bool), warm_start.astype(bool), obstacle_indices)

    def closest_prune_pairs_mvt_warm(self, mvt_handle: int, query_padding: float, robot_centers: np.ndarray, robot_radii: np.ndarray, obstacle_centers: np.ndarray, eigenvalues: np.ndarray, rotations: np.ndarray, obstacle_shapes: np.ndarray, obstacle_offsets: np.ndarray, multiplier_dense: np.ndarray, *, contact_distance: float=0.0, maximum_newton_iterations: int=10, maximum_bisection_iterations: int=48, tolerance: float=1e-12):
        robot_centers = np.ascontiguousarray(robot_centers, dtype=np.float64).reshape(-1, 3)
        radii = np.ascontiguousarray(robot_radii, dtype=np.float64).reshape(-1)
        centers = np.ascontiguousarray(obstacle_centers, dtype=np.float64).reshape(-1, 3)
        values = np.ascontiguousarray(eigenvalues, dtype=np.float64).reshape(-1, 3)
        rotations = np.ascontiguousarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
        shapes = np.ascontiguousarray(obstacle_shapes, dtype=np.float64).reshape(-1, 3, 3)
        offsets = np.ascontiguousarray(obstacle_offsets, dtype=np.float64).reshape(-1)
        multipliers_dense = np.asarray(multiplier_dense)
        if multipliers_dense.dtype != np.float64 or not multipliers_dense.flags.c_contiguous:
            raise ValueError('multiplier_dense must be contiguous float64')
        multipliers_dense = multipliers_dense.reshape(len(robot_centers), -1)
        if not int(mvt_handle):
            raise ValueError('a live native MVT handle is required')
        if query_padding < 0.0:
            raise ValueError('query padding must be nonnegative')
        if len(radii) != len(robot_centers):
            raise ValueError('one radius is required per robot center')
        if not len(centers) == len(values) == len(rotations) == len(shapes) == len(offsets):
            raise ValueError('global obstacle arrays must have equal length')
        if multipliers_dense.shape != (len(robot_centers), len(centers)):
            raise ValueError('multiplier_dense shape must be robot x obstacle')
        group_count = len(robot_centers)
        requested_capacity = max(4096, 2 * len(centers))
        dp = ctypes.POINTER(ctypes.c_double)
        ip = ctypes.POINTER(ctypes.c_int32)
        while True:
            if requested_capacity > self._fused_candidate_capacity:
                capacity = 1 << (requested_capacity - 1).bit_length()
                self._fused_candidate_capacity = capacity
                self._fused_candidate_buffers = {'obstacles': np.empty(capacity, dtype=np.int32), 'normals': np.empty((capacity, 3), dtype=np.float64), 'surfaces': np.empty((capacity, 3), dtype=np.float64), 'multipliers': np.empty(capacity, dtype=np.float64), 'newton': np.empty(capacity, dtype=np.int32), 'bisection': np.empty(capacity, dtype=np.int32), 'residuals': np.empty(capacity, dtype=np.float64), 'computed': np.empty(capacity, dtype=np.int32), 'warm': np.empty(capacity, dtype=np.int32), 'active': np.empty(capacity, dtype=np.int32)}
            buffers = self._fused_candidate_buffers
            groups = np.empty(group_count + 1, dtype=np.int32)
            active_counts = np.empty(group_count, dtype=np.int32)
            written = self._lib.mvt_ellipsoid_closest_prune_pairs_warm(ctypes.c_void_p(int(mvt_handle)), robot_centers.ctypes.data_as(dp), radii.ctypes.data_as(dp), float(query_padding), group_count, self._fused_candidate_capacity, buffers['obstacles'].ctypes.data_as(ip), groups.ctypes.data_as(ip), len(centers), centers.ctypes.data_as(dp), values.ctypes.data_as(dp), rotations.ctypes.data_as(dp), shapes.ctypes.data_as(dp), offsets.ctypes.data_as(dp), multipliers_dense.ctypes.data_as(dp), int(maximum_newton_iterations), int(maximum_bisection_iterations), float(tolerance), float(contact_distance), buffers['normals'].ctypes.data_as(dp), buffers['surfaces'].ctypes.data_as(dp), buffers['multipliers'].ctypes.data_as(dp), buffers['newton'].ctypes.data_as(ip), buffers['bisection'].ctypes.data_as(ip), buffers['residuals'].ctypes.data_as(dp), buffers['computed'].ctypes.data_as(ip), buffers['warm'].ctypes.data_as(ip), buffers['active'].ctypes.data_as(ip), active_counts.ctypes.data_as(ip))
            if written != -20:
                break
            requested_capacity = 2 * self._fused_candidate_capacity
        if written < 0:
            raise RuntimeError(f'native fused MVT closest-prune failed with code {written}')
        count = int(written)
        if groups[0] != 0 or groups[-1] != count or np.any(np.diff(groups) < 0):
            raise RuntimeError('native fused MVT returned invalid group offsets')
        active_groups = [buffers['active'][groups[index]:groups[index] + active_counts[index]].copy() for index in range(group_count)]
        return (active_groups, buffers['normals'][:count], buffers['surfaces'][:count], buffers['multipliers'][:count], buffers['newton'][:count], buffers['bisection'][:count], buffers['residuals'][:count], buffers['computed'][:count] != 0, buffers['warm'][:count] != 0, buffers['obstacles'][:count], groups)

    def sphere_prune_pairs_mvt(self, mvt_handle: int, query_padding: float, robot_centers: np.ndarray, robot_radii: np.ndarray, obstacle_centers: np.ndarray, obstacle_radii: np.ndarray, obstacle_offsets: np.ndarray, *, contact_distance: float=0.0):
        robot_centers = np.ascontiguousarray(robot_centers, dtype=np.float64).reshape(-1, 3)
        robot_radii = np.ascontiguousarray(robot_radii, dtype=np.float64).reshape(-1)
        centers = np.ascontiguousarray(obstacle_centers, dtype=np.float64).reshape(-1, 3)
        radii = np.ascontiguousarray(obstacle_radii, dtype=np.float64).reshape(-1)
        offsets = np.ascontiguousarray(obstacle_offsets, dtype=np.float64).reshape(-1)
        if not int(mvt_handle):
            raise ValueError('a live native MVT handle is required')
        if query_padding < 0.0:
            raise ValueError('query padding must be nonnegative')
        if len(robot_radii) != len(robot_centers):
            raise ValueError('one radius is required per robot center')
        if not len(centers) == len(radii) == len(offsets):
            raise ValueError('global sphere obstacle arrays must have equal length')
        group_count = len(robot_centers)
        requested_capacity = max(4096, 2 * len(centers))
        dp = ctypes.POINTER(ctypes.c_double)
        ip = ctypes.POINTER(ctypes.c_int32)
        while True:
            if requested_capacity > self._fused_sphere_capacity:
                capacity = 1 << (requested_capacity - 1).bit_length()
                self._fused_sphere_capacity = capacity
                self._fused_sphere_buffers = {'obstacles': np.empty(capacity, dtype=np.int32), 'normals': np.empty((capacity, 3), dtype=np.float64), 'surfaces': np.empty((capacity, 3), dtype=np.float64), 'active': np.empty(capacity, dtype=np.int32)}
            buffers = self._fused_sphere_buffers
            groups = np.empty(group_count + 1, dtype=np.int32)
            active_counts = np.empty(group_count, dtype=np.int32)
            written = self._lib.mvt_sphere_prune_pairs(ctypes.c_void_p(int(mvt_handle)), robot_centers.ctypes.data_as(dp), robot_radii.ctypes.data_as(dp), float(query_padding), group_count, self._fused_sphere_capacity, buffers['obstacles'].ctypes.data_as(ip), groups.ctypes.data_as(ip), len(centers), centers.ctypes.data_as(dp), radii.ctypes.data_as(dp), offsets.ctypes.data_as(dp), float(contact_distance), buffers['normals'].ctypes.data_as(dp), buffers['surfaces'].ctypes.data_as(dp), buffers['active'].ctypes.data_as(ip), active_counts.ctypes.data_as(ip))
            if written != -20:
                break
            requested_capacity = 2 * self._fused_sphere_capacity
        if written < 0:
            raise RuntimeError(f'native fused MVT sphere-prune failed with code {written}')
        count = int(written)
        if groups[0] != 0 or groups[-1] != count or np.any(np.diff(groups) < 0):
            raise RuntimeError('native fused sphere path returned invalid groups')
        active_groups = [buffers['active'][groups[index]:groups[index] + active_counts[index]].copy() for index in range(group_count)]
        return (active_groups, buffers['normals'][:count], buffers['surfaces'][:count], buffers['obstacles'][:count], groups)

    def normals_sum_pairs_warm(self, robot_centers: np.ndarray, robot_shapes: np.ndarray, obstacle_centers: np.ndarray, obstacle_shapes: np.ndarray, uncertainty_shapes: np.ndarray, initial_normals: np.ndarray, *, max_iterations: int=32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        robot_centers = np.ascontiguousarray(robot_centers, dtype=np.float64).reshape(-1, 3)
        robot_shapes = np.ascontiguousarray(robot_shapes, dtype=np.float64).reshape(-1, 3, 3)
        obstacle_centers = np.ascontiguousarray(obstacle_centers, dtype=np.float64).reshape(-1, 3)
        obstacle_shapes = np.ascontiguousarray(obstacle_shapes, dtype=np.float64).reshape(-1, 3, 3)
        uncertainty = np.ascontiguousarray(uncertainty_shapes, dtype=np.float64).reshape(-1, 3, 3)
        initial = np.ascontiguousarray(initial_normals, dtype=np.float64).reshape(-1, 3)
        count = len(robot_centers)
        if not len(robot_shapes) == len(obstacle_centers) == len(obstacle_shapes) == len(uncertainty) == len(initial) == count:
            raise ValueError('pairwise support arrays must have equal length')
        output = np.empty((count, 3), dtype=np.float64)
        iterations = np.empty(count, dtype=np.int32)
        residuals = np.empty(count, dtype=np.float64)
        dp = ctypes.POINTER(ctypes.c_double)
        written = self._lib.ellipsoid_support_normals_sum_pairs_warm(robot_centers.ctypes.data_as(dp), robot_shapes.ctypes.data_as(dp), obstacle_centers.ctypes.data_as(dp), obstacle_shapes.ctypes.data_as(dp), uncertainty.ctypes.data_as(dp), initial.ctypes.data_as(dp), count, int(max_iterations), output.ctypes.data_as(dp), iterations.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), residuals.ctypes.data_as(dp))
        if written != count:
            raise RuntimeError(f'native pairwise warm support failed with code {written}')
        return (output, iterations, residuals)

    def normals_sum_newton_warm(self, robot_center: np.ndarray, robot_shape: np.ndarray, obstacle_centers: np.ndarray, obstacle_shapes: np.ndarray, uncertainty_shapes: np.ndarray, initial_normals: np.ndarray, *, max_iterations: int=16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = np.ascontiguousarray(robot_center, dtype=np.float64).reshape(3)
        shape = np.ascontiguousarray(robot_shape, dtype=np.float64).reshape(3, 3)
        centers = np.ascontiguousarray(obstacle_centers, dtype=np.float64).reshape(-1, 3)
        shapes = np.ascontiguousarray(obstacle_shapes, dtype=np.float64).reshape(-1, 3, 3)
        uncertainty = np.ascontiguousarray(uncertainty_shapes, dtype=np.float64).reshape(-1, 3, 3)
        initial = np.ascontiguousarray(initial_normals, dtype=np.float64).reshape(-1, 3)
        count = len(centers)
        if not len(shapes) == len(uncertainty) == len(initial) == count:
            raise ValueError('Newton support-sum arrays must have equal length')
        output = np.empty((count, 3), dtype=np.float64)
        iterations = np.empty(count, dtype=np.int32)
        residuals = np.empty(count, dtype=np.float64)
        dp = ctypes.POINTER(ctypes.c_double)
        written = self._lib.ellipsoid_support_normals_sum_newton_warm(center.ctypes.data_as(dp), shape.ctypes.data_as(dp), centers.ctypes.data_as(dp), shapes.ctypes.data_as(dp), uncertainty.ctypes.data_as(dp), initial.ctypes.data_as(dp), count, int(max_iterations), output.ctypes.data_as(dp), iterations.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), residuals.ctypes.data_as(dp))
        if written != count:
            raise RuntimeError(f'native Newton support-sum failed with code {written}')
        return (output, iterations, residuals)

    def normals_sum_pairs_newton_warm(self, robot_centers: np.ndarray, robot_shapes: np.ndarray, obstacle_centers: np.ndarray, obstacle_shapes: np.ndarray, uncertainty_shapes: np.ndarray, initial_normals: np.ndarray, *, max_iterations: int=16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        robot_centers = np.ascontiguousarray(robot_centers, dtype=np.float64).reshape(-1, 3)
        robot_shapes = np.ascontiguousarray(robot_shapes, dtype=np.float64).reshape(-1, 3, 3)
        obstacle_centers = np.ascontiguousarray(obstacle_centers, dtype=np.float64).reshape(-1, 3)
        obstacle_shapes = np.ascontiguousarray(obstacle_shapes, dtype=np.float64).reshape(-1, 3, 3)
        uncertainty = np.ascontiguousarray(uncertainty_shapes, dtype=np.float64).reshape(-1, 3, 3)
        initial = np.ascontiguousarray(initial_normals, dtype=np.float64).reshape(-1, 3)
        count = len(robot_centers)
        if not len(robot_shapes) == len(obstacle_centers) == len(obstacle_shapes) == len(uncertainty) == len(initial) == count:
            raise ValueError('pairwise Newton support arrays must have equal length')
        output = np.empty((count, 3), dtype=np.float64)
        iterations = np.empty(count, dtype=np.int32)
        residuals = np.empty(count, dtype=np.float64)
        dp = ctypes.POINTER(ctypes.c_double)
        written = self._lib.ellipsoid_support_normals_sum_pairs_newton_warm(robot_centers.ctypes.data_as(dp), robot_shapes.ctypes.data_as(dp), obstacle_centers.ctypes.data_as(dp), obstacle_shapes.ctypes.data_as(dp), uncertainty.ctypes.data_as(dp), initial.ctypes.data_as(dp), count, int(max_iterations), output.ctypes.data_as(dp), iterations.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), residuals.ctypes.data_as(dp))
        if written != count:
            raise RuntimeError(f'native pairwise Newton support failed with code {written}')
        return (output, iterations, residuals)

    def prune_planes(self, obstacle_centers: np.ndarray, obstacle_shapes: np.ndarray, obstacle_offsets: np.ndarray, normals: np.ndarray, plane_offsets: np.ndarray) -> np.ndarray:
        centers = np.ascontiguousarray(obstacle_centers, dtype=np.float64).reshape(-1, 3)
        shapes = np.ascontiguousarray(obstacle_shapes, dtype=np.float64).reshape(-1, 3, 3)
        offsets = np.ascontiguousarray(obstacle_offsets, dtype=np.float64).reshape(-1)
        normals = np.ascontiguousarray(normals, dtype=np.float64).reshape(-1, 3)
        plane_offsets = np.ascontiguousarray(plane_offsets, dtype=np.float64).reshape(-1)
        count = len(centers)
        if not len(shapes) == len(offsets) == len(normals) == len(plane_offsets) == count:
            raise ValueError('native prune arrays must have equal length')
        output = np.empty(count, dtype=np.int32)
        dp = ctypes.POINTER(ctypes.c_double)
        written = self._lib.ellipsoid_prune_planes(centers.ctypes.data_as(dp), shapes.ctypes.data_as(dp), offsets.ctypes.data_as(dp), normals.ctypes.data_as(dp), plane_offsets.ctypes.data_as(dp), count, output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), count)
        if written < 0:
            raise RuntimeError(f'native ellipsoid pruning failed with code {written}')
        return output[:written].copy()

    def prune_planes_sum(self, obstacle_centers: np.ndarray, obstacle_shapes: np.ndarray, uncertainty_shapes: np.ndarray, obstacle_offsets: np.ndarray, normals: np.ndarray, plane_offsets: np.ndarray) -> np.ndarray:
        centers = np.ascontiguousarray(obstacle_centers, dtype=np.float64).reshape(-1, 3)
        shapes = np.ascontiguousarray(obstacle_shapes, dtype=np.float64).reshape(-1, 3, 3)
        uncertainty = np.ascontiguousarray(uncertainty_shapes, dtype=np.float64).reshape(-1, 3, 3)
        offsets = np.ascontiguousarray(obstacle_offsets, dtype=np.float64).reshape(-1)
        normals = np.ascontiguousarray(normals, dtype=np.float64).reshape(-1, 3)
        plane_offsets = np.ascontiguousarray(plane_offsets, dtype=np.float64).reshape(-1)
        count = len(centers)
        if not len(shapes) == len(uncertainty) == len(offsets) == len(normals) == len(plane_offsets) == count:
            raise ValueError('native support-sum prune arrays must have equal length')
        output = np.empty(count, dtype=np.int32)
        dp = ctypes.POINTER(ctypes.c_double)
        written = self._lib.ellipsoid_prune_planes_sum(centers.ctypes.data_as(dp), shapes.ctypes.data_as(dp), uncertainty.ctypes.data_as(dp), offsets.ctypes.data_as(dp), normals.ctypes.data_as(dp), plane_offsets.ctypes.data_as(dp), count, output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), count)
        if written < 0:
            raise RuntimeError(f'native support-sum ellipsoid pruning failed with code {written}')
        return output[:written].copy()
