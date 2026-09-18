"""Extracted original v4.3 ellipsoid dependency; see docs/source_manifest.json."""
from __future__ import annotations
import ctypes
from typing import Iterable
import numpy as np
from camera import DepthObservation, environment_endpoint_mask
from map_types import FREE, OCCUPIED, UNKNOWN, MapUpdateStats, SweptVolumeCertificate
from voxel_index import DLL

class NativeIncrementalOccupancyMap:

    def __init__(self, voxel_size: float=0.015, *, free_decrement: int=1, occupied_increment: int=4, minimum_score: int=-8, maximum_score: int=8, occupied_threshold: int=2, free_threshold: int=-1, track_state_deltas: bool=False) -> None:
        self._lib = ctypes.CDLL(str(DLL))
        vp = ctypes.c_void_p
        dp = ctypes.POINTER(ctypes.c_double)
        i32p = ctypes.POINTER(ctypes.c_int32)
        i8p = ctypes.POINTER(ctypes.c_int8)
        i64p = ctypes.POINTER(ctypes.c_int64)
        self._lib.occ_incremental_create.argtypes = [ctypes.c_double, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        self._lib.occ_incremental_create.restype = vp
        self._lib.occ_incremental_destroy.argtypes = [vp]
        self._lib.occ_incremental_set_state_journal_enabled.argtypes = [vp, ctypes.c_int32]
        self._lib.occ_incremental_set_state_journal_enabled.restype = ctypes.c_int32
        self._lib.occ_incremental_set_calibrated_free_aabb.argtypes = [vp, dp, dp]
        self._lib.occ_incremental_set_calibrated_free_aabb.restype = ctypes.c_int32
        self._lib.occ_incremental_integrate.argtypes = [vp, dp, dp, dp, i8p, ctypes.c_int32, i64p]
        self._lib.occ_incremental_integrate.restype = ctypes.c_int32
        self._lib.occ_incremental_integrate_ellipsoids.argtypes = [vp, dp, dp, dp, dp, i8p, ctypes.c_int32, i64p]
        self._lib.occ_incremental_integrate_ellipsoids.restype = ctypes.c_int32
        self._lib.occ_incremental_mark_free.argtypes = [vp, dp, dp, ctypes.c_int32, ctypes.c_double]
        self._lib.occ_incremental_mark_free.restype = ctypes.c_int32
        self._lib.occ_incremental_certify_swept_spheres.argtypes = [vp, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_float, i32p]
        self._lib.occ_incremental_certify_swept_spheres.restype = ctypes.c_int32
        self._lib.occ_incremental_size.argtypes = [vp]
        self._lib.occ_incremental_size.restype = ctypes.c_int32
        self._lib.occ_incremental_clone_snapshot.argtypes = [vp]
        self._lib.occ_incremental_clone_snapshot.restype = vp
        self._lib.occ_incremental_export.argtypes = [vp, i32p, i8p, ctypes.c_int32]
        self._lib.occ_incremental_export.restype = ctypes.c_int32
        self._lib.occ_incremental_dirty_size.argtypes = [vp]
        self._lib.occ_incremental_dirty_size.restype = ctypes.c_int32
        self._lib.occ_incremental_export_dirty.argtypes = [vp, i32p, i8p, ctypes.c_int32]
        self._lib.occ_incremental_export_dirty.restype = ctypes.c_int32
        self._handle = self._lib.occ_incremental_create(float(voxel_size), int(free_decrement), int(occupied_increment), int(minimum_score), int(maximum_score), int(occupied_threshold), int(free_threshold))
        if not self._handle:
            raise RuntimeError('native incremental occupancy construction failed')
        code = self._lib.occ_incremental_set_state_journal_enabled(self._handle, int(bool(track_state_deltas)))
        if code != 0:
            self.close()
            raise RuntimeError('native occupancy state journal setup failed')
        self.track_state_deltas = bool(track_state_deltas)
        self.voxel_size = float(voxel_size)
        self._unknown_queries = 0
        self._stats = np.zeros(6, dtype=np.int64)
        self._query_counts = np.zeros(3, dtype=np.int32)

    def set_calibrated_free_aabb(self, lower: np.ndarray, upper: np.ndarray) -> None:
        lower = np.ascontiguousarray(lower, dtype=np.float64).reshape(3)
        upper = np.ascontiguousarray(upper, dtype=np.float64).reshape(3)
        dp = ctypes.POINTER(ctypes.c_double)
        code = self._lib.occ_incremental_set_calibrated_free_aabb(self._handle, lower.ctypes.data_as(dp), upper.ctypes.data_as(dp))
        if code != 0:
            raise ValueError(f'invalid calibrated free AABB (code {code})')

    def integrate(self, observations: Iterable[DepthObservation]) -> MapUpdateStats:
        observations = list(observations)
        point_blocks = [np.asarray(item.points, dtype=np.float64).reshape(-1, 3) for item in observations]
        for item, points in zip(observations, point_blocks):
            if len(points) != len(item.sample_radii):
                raise ValueError('each depth point needs one conservative radius')
            if item.sample_uncertainty_shapes is not None and len(item.sample_uncertainty_shapes) != len(points):
                raise ValueError('each depth point needs one directional uncertainty shape')
        directional = bool(observations) and all((item.sample_uncertainty_shapes is not None for item in observations))
        ray_count = sum((len(points) for points in point_blocks))
        if ray_count:
            endpoints = np.ascontiguousarray(np.vstack(point_blocks), dtype=np.float64)
            radii = np.ascontiguousarray(np.concatenate([item.sample_radii for item in observations]), dtype=np.float64)
            endpoint_masks = np.ascontiguousarray(np.concatenate([environment_endpoint_mask(item) for item in observations]), dtype=np.int8)
            origins = np.ascontiguousarray(np.vstack([np.repeat(np.asarray(item.camera_position, dtype=np.float64)[None, :], len(points), axis=0) for item, points in zip(observations, point_blocks)]), dtype=np.float64)
            shapes = np.ascontiguousarray(np.vstack([item.sample_uncertainty_shapes for item in observations]), dtype=np.float64).reshape(-1, 3, 3) if directional else None
        else:
            origins = np.zeros((1, 3), dtype=np.float64)
            endpoints = np.zeros((1, 3), dtype=np.float64)
            radii = np.zeros(1, dtype=np.float64)
            endpoint_masks = np.zeros(1, dtype=np.int8)
            shapes = np.zeros((1, 3, 3), dtype=np.float64) if directional else None
        dp = ctypes.POINTER(ctypes.c_double)
        if directional:
            code = self._lib.occ_incremental_integrate_ellipsoids(self._handle, origins.ctypes.data_as(dp), endpoints.ctypes.data_as(dp), radii.ctypes.data_as(dp), shapes.ctypes.data_as(dp), endpoint_masks.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), ray_count, self._stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
        else:
            code = self._lib.occ_incremental_integrate(self._handle, origins.ctypes.data_as(dp), endpoints.ctypes.data_as(dp), radii.ctypes.data_as(dp), endpoint_masks.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), ray_count, self._stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
        if code != 0:
            raise RuntimeError(f'native incremental integrate failed with code {code}')
        return MapUpdateStats(frame_index=int(self._stats[0]), rays=int(self._stats[1]), free_updates=int(self._stats[2]), occupied_updates=int(self._stats[3]), free_voxels=int(self._stats[4]), occupied_voxels=int(self._stats[5]), unknown_queries=self._unknown_queries)

    def mark_current_robot_free(self, centers: np.ndarray, radii: np.ndarray, padding: float=0.0) -> None:
        centers = np.ascontiguousarray(centers, dtype=np.float64).reshape(-1, 3)
        radii = np.ascontiguousarray(radii, dtype=np.float64).reshape(-1)
        if len(centers) != len(radii):
            raise ValueError('centers and radii need equal length')
        dp = ctypes.POINTER(ctypes.c_double)
        code = self._lib.occ_incremental_mark_free(self._handle, centers.ctypes.data_as(dp), radii.ctypes.data_as(dp), len(radii), float(padding))
        if code != 0:
            raise RuntimeError(f'native mark-free failed with code {code}')

    def certify_swept_spheres(self, starts: np.ndarray, ends: np.ndarray, radii: np.ndarray, *, margin: float=0.0) -> SweptVolumeCertificate:
        starts = np.ascontiguousarray(starts, dtype=np.float32).reshape(-1, 3)
        ends = np.ascontiguousarray(ends, dtype=np.float32).reshape(-1, 3)
        radii = np.ascontiguousarray(radii, dtype=np.float32).reshape(-1)
        if starts.shape != ends.shape or len(starts) != len(radii):
            raise ValueError('swept sphere arrays need matching lengths')
        fp = ctypes.POINTER(ctypes.c_float)
        code = self._lib.occ_incremental_certify_swept_spheres(self._handle, starts.ctypes.data_as(fp), ends.ctypes.data_as(fp), radii.ctypes.data_as(fp), len(radii), float(margin), self._query_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if code < 0:
            raise RuntimeError(f'native incremental occupancy query failed with code {code}')
        self._unknown_queries += int(self._query_counts[1])
        return SweptVolumeCertificate(safe=bool(code), occupied_voxels=int(self._query_counts[0]), unknown_voxels=int(self._query_counts[1]), checked_voxels=int(self._query_counts[2]))

    def snapshot_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        count = int(self._lib.occ_incremental_size(self._handle))
        if count < 0:
            raise RuntimeError('native incremental occupancy size failed')
        keys = np.empty((count, 3), dtype=np.int32)
        states = np.empty(count, dtype=np.int8)
        written = self._lib.occ_incremental_export(self._handle, keys.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), states.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), count)
        if written != count:
            raise RuntimeError(f'native occupancy export wrote {written}, expected {count}')
        return (keys, states)

    def snapshot_delta_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.track_state_deltas:
            raise RuntimeError('state delta journal is disabled; construct with track_state_deltas=True')
        count = int(self._lib.occ_incremental_dirty_size(self._handle))
        if count < 0:
            raise RuntimeError('native incremental occupancy dirty size failed')
        keys = np.empty((count, 3), dtype=np.int32)
        states = np.empty(count, dtype=np.int8)
        written = self._lib.occ_incremental_export_dirty(self._handle, keys.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), states.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), count)
        if written != count:
            raise RuntimeError(f'native occupancy delta export wrote {written}, expected {count}')
        return (keys, states)

    def clone_snapshot(self) -> 'NativeOccupancySnapshot':
        handle = self._lib.occ_incremental_clone_snapshot(self._handle)
        if not handle:
            raise RuntimeError('native occupancy clone failed (calibrated AABB snapshots are unsupported)')
        return NativeOccupancySnapshot._from_native_handle(self._lib, handle, self.voxel_size)

    def state_counts(self) -> dict[int, int]:
        _, states = self.snapshot_arrays()
        return {UNKNOWN: int(np.count_nonzero(states == UNKNOWN)), FREE: int(np.count_nonzero(states == FREE)), OCCUPIED: int(np.count_nonzero(states == OCCUPIED))}

    def close(self) -> None:
        if getattr(self, '_handle', None):
            self._lib.occ_incremental_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

class NativeOccupancySnapshot:

    @staticmethod
    def _configure_library(lib) -> None:
        i32p = ctypes.POINTER(ctypes.c_int32)
        i8p = ctypes.POINTER(ctypes.c_int8)
        fp = ctypes.POINTER(ctypes.c_float)
        lib.occ_create.argtypes = [i32p, i8p, ctypes.c_int32, ctypes.c_float]
        lib.occ_create.restype = ctypes.c_void_p
        lib.occ_destroy.argtypes = [ctypes.c_void_p]
        lib.occ_certify_swept_spheres.argtypes = [ctypes.c_void_p, fp, fp, fp, ctypes.c_int32, ctypes.c_float, i32p]
        lib.occ_certify_swept_spheres.restype = ctypes.c_int32
        dp = ctypes.POINTER(ctypes.c_double)
        lib.occ_certify_ellipsoids.argtypes = [ctypes.c_void_p, dp, dp, ctypes.c_int32, i32p]
        lib.occ_certify_ellipsoids.restype = ctypes.c_int32

    @classmethod
    def _from_native_handle(cls, lib, handle, voxel_size: float) -> 'NativeOccupancySnapshot':
        instance = cls.__new__(cls)
        instance._lib = lib
        cls._configure_library(lib)
        instance._handle = handle
        instance.voxel_size = float(voxel_size)
        instance._counts = np.zeros(3, dtype=np.int32)
        return instance

    def __init__(self, keys: np.ndarray, states: np.ndarray, voxel_size: float):
        keys = np.ascontiguousarray(keys, dtype=np.int32).reshape(-1, 3)
        states = np.ascontiguousarray(states, dtype=np.int8).reshape(-1)
        if len(keys) != len(states):
            raise ValueError('keys and states need equal length')
        self._lib = ctypes.CDLL(str(DLL))
        self._configure_library(self._lib)
        i32p = ctypes.POINTER(ctypes.c_int32)
        i8p = ctypes.POINTER(ctypes.c_int8)
        self._handle = self._lib.occ_create(keys.ctypes.data_as(i32p), states.ctypes.data_as(i8p), len(keys), float(voxel_size))
        if not self._handle:
            raise RuntimeError('native occupancy construction failed')
        self.voxel_size = float(voxel_size)
        self._counts = np.zeros(3, dtype=np.int32)

    def certify_swept_spheres(self, starts: np.ndarray, ends: np.ndarray, radii: np.ndarray, *, margin: float=0.0) -> SweptVolumeCertificate:
        starts = np.ascontiguousarray(starts, dtype=np.float32).reshape(-1, 3)
        ends = np.ascontiguousarray(ends, dtype=np.float32).reshape(-1, 3)
        radii = np.ascontiguousarray(radii, dtype=np.float32).reshape(-1)
        if starts.shape != ends.shape or len(starts) != len(radii):
            raise ValueError('swept sphere arrays need matching lengths')
        fp = ctypes.POINTER(ctypes.c_float)
        code = self._lib.occ_certify_swept_spheres(self._handle, starts.ctypes.data_as(fp), ends.ctypes.data_as(fp), radii.ctypes.data_as(fp), len(radii), float(margin), self._counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if code < 0:
            raise RuntimeError(f'native occupancy query failed with code {code}')
        return SweptVolumeCertificate(safe=bool(code), occupied_voxels=int(self._counts[0]), unknown_voxels=int(self._counts[1]), checked_voxels=int(self._counts[2]))

    def certify_ellipsoids(self, centers: np.ndarray, shapes: np.ndarray) -> SweptVolumeCertificate:
        centers = np.ascontiguousarray(centers, dtype=np.float64).reshape(-1, 3)
        shapes = np.ascontiguousarray(shapes, dtype=np.float64).reshape(-1, 3, 3)
        if len(centers) != len(shapes):
            raise ValueError('centers and shapes need equal length')
        dp = ctypes.POINTER(ctypes.c_double)
        code = self._lib.occ_certify_ellipsoids(self._handle, centers.ctypes.data_as(dp), shapes.ctypes.data_as(dp), len(centers), self._counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if code < 0:
            raise RuntimeError(f'native ellipsoid occupancy query failed with code {code}')
        return SweptVolumeCertificate(safe=bool(code), occupied_voxels=int(self._counts[0]), unknown_voxels=int(self._counts[1]), checked_voxels=int(self._counts[2]))

    def close(self) -> None:
        if getattr(self, '_handle', None):
            self._lib.occ_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
