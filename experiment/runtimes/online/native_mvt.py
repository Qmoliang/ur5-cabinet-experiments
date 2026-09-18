"""ctypes wrappers for native AVX2 voxel-table broadphases.

``NativeMVT`` is the historical replicated single-level baseline.
``NativeMultilevelMVT`` is the formal VCC-style table: one proxy reference,
one size-selected level, and exactly 27 cell lookups per populated level.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DLL = ROOT / "native_mvt" / "native_mvt.dll"


@dataclass(frozen=True)
class NativeMVTStats:
    proxy_count: int
    occupied_cells: int
    index_references: int
    simd_width: int
    voxel_size: float


@dataclass(frozen=True)
class BruteForceAABBStats:
    """Metadata for the exact O(N) AABB-oracle baseline."""

    proxy_count: int
    level_count: int = 0
    occupied_cells: int = 0
    index_references: int = 0
    simd_width: int = 1
    cell_lookups_per_query: int = 0


class BruteForceAABBIndex:
    """Scan every proxy AABB using the same float32 predicates as native MVT.

    This is not a spatial acceleration structure.  It exists so an MVT
    ablation can hold the broadphase predicate, candidate IDs, and sorted
    order fixed while changing only how those candidates are found.
    """

    def __init__(
        self,
        centers: np.ndarray,
        half_extents: np.ndarray,
        *,
        query_padding: float = 0.0,
    ) -> None:
        centers = np.ascontiguousarray(centers, dtype=np.float32).reshape(-1, 3)
        half_extents = np.ascontiguousarray(
            half_extents, dtype=np.float32
        ).reshape(-1, 3)
        if centers.shape != half_extents.shape or not len(centers):
            raise ValueError(
                "centers and half_extents need matching non-empty (N,3) shapes"
            )
        # Native MVT stores these two arrays as float.  Precomputing them here
        # with float32 arithmetic makes boundary cases part of the equivalence
        # test instead of silently comparing a float64 oracle to float32 C++.
        self._lo = np.ascontiguousarray(centers - half_extents, dtype=np.float32)
        self._hi = np.ascontiguousarray(centers + half_extents, dtype=np.float32)
        self.query_padding = float(query_padding)
        self.stats = BruteForceAABBStats(proxy_count=len(centers))

    def query_aabb(
        self, center: np.ndarray, half_extent: np.ndarray
    ) -> np.ndarray:
        center32 = np.ascontiguousarray(center, dtype=np.float32).reshape(3)
        half32 = np.ascontiguousarray(half_extent, dtype=np.float32).reshape(3)
        qlo = center32 - half32
        qhi = center32 + half32
        return np.flatnonzero(
            np.all(self._lo <= qhi, axis=1)
            & np.all(self._hi >= qlo, axis=1)
        ).astype(np.int64, copy=False)

    def query_sphere(self, center: np.ndarray, radius: float) -> np.ndarray:
        center32 = np.ascontiguousarray(center, dtype=np.float32).reshape(3)
        radius32 = np.float32(float(radius) + self.query_padding)
        below = np.maximum(self._lo - center32, np.float32(0.0))
        above = np.maximum(center32 - self._hi, np.float32(0.0))
        delta = np.maximum(below, above)
        squared = np.sum(delta * delta, axis=1, dtype=np.float32)
        return np.flatnonzero(squared <= radius32 * radius32).astype(
            np.int64, copy=False
        )

    def query_spheres(
        self, centers: np.ndarray, radii: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        centers = np.asarray(centers, dtype=float).reshape(-1, 3)
        radii = np.asarray(radii, dtype=float).reshape(-1)
        if len(centers) != len(radii):
            raise ValueError("centers and radii must have matching lengths")
        return tuple(
            self.query_sphere(center, float(radius))
            for center, radius in zip(centers, radii)
        )

    def close(self) -> None:
        return None


class NativeMVT:
    def __init__(
        self,
        centers: np.ndarray,
        half_extents: np.ndarray,
        voxel_size: float,
        *,
        query_padding: float = 0.0,
        simd: bool = True,
    ):
        if not DLL.exists():
            raise FileNotFoundError(f"native MVT DLL is missing; run build_native_mvt.py: {DLL}")
        centers = np.ascontiguousarray(centers, dtype=np.float32).reshape(-1, 3)
        half_extents = np.ascontiguousarray(half_extents, dtype=np.float32).reshape(-1, 3)
        if centers.shape != half_extents.shape or not len(centers):
            raise ValueError("centers and half_extents need matching non-empty (N,3) shapes")
        self._lib = ctypes.CDLL(str(DLL))
        fp = ctypes.POINTER(ctypes.c_float)
        self._lib.mvt_create.argtypes = [fp, fp, ctypes.c_int32, ctypes.c_float]
        self._lib.mvt_create.restype = ctypes.c_void_p
        self._lib.mvt_destroy.argtypes = [ctypes.c_void_p]
        self._lib.mvt_query_aabb.argtypes = [ctypes.c_void_p, fp, fp, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
        self._lib.mvt_query_aabb.restype = ctypes.c_int32
        self._lib.mvt_query_aabb_scalar.argtypes = [ctypes.c_void_p, fp, fp, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
        self._lib.mvt_query_aabb_scalar.restype = ctypes.c_int32
        for name in ("mvt_proxy_count", "mvt_occupied_cells"):
            getattr(self._lib, name).argtypes = [ctypes.c_void_p]
            getattr(self._lib, name).restype = ctypes.c_int32
        self._lib.mvt_index_references.argtypes = [ctypes.c_void_p]
        self._lib.mvt_index_references.restype = ctypes.c_int64
        self._lib.mvt_simd_width.restype = ctypes.c_int32
        self._handle = self._lib.mvt_create(
            centers.ctypes.data_as(fp), half_extents.ctypes.data_as(fp), len(centers), float(voxel_size)
        )
        if not self._handle:
            raise RuntimeError("native MVT construction failed")
        self._capacity = len(centers)
        self._output = np.empty(self._capacity, dtype=np.int32)
        self.voxel_size = float(voxel_size)
        self.query_padding = float(query_padding)
        self.simd = bool(simd)
        self.stats = NativeMVTStats(
            proxy_count=int(self._lib.mvt_proxy_count(self._handle)),
            occupied_cells=int(self._lib.mvt_occupied_cells(self._handle)),
            index_references=int(self._lib.mvt_index_references(self._handle)),
            simd_width=int(self._lib.mvt_simd_width()),
            voxel_size=float(voxel_size),
        )

    @classmethod
    def from_spheres(cls, centers, radii, voxel_size, **kwargs):
        radii = np.asarray(radii, dtype=float).reshape(-1)
        return cls(centers, np.repeat(radii[:, None], 3, axis=1), voxel_size, **kwargs)

    @classmethod
    def from_ellipsoids(cls, centers, shapes, offsets, voxel_size, **kwargs):
        half = np.sqrt(np.maximum(np.diagonal(np.asarray(shapes), axis1=1, axis2=2), 0.0))
        half += np.asarray(offsets, dtype=float).reshape(-1, 1)
        return cls(centers, half, voxel_size, **kwargs)

    def query_aabb(self, center: np.ndarray, half_extent: np.ndarray) -> np.ndarray:
        center = np.ascontiguousarray(center, dtype=np.float32).reshape(3)
        half = np.ascontiguousarray(half_extent, dtype=np.float32).reshape(3)
        fp = ctypes.POINTER(ctypes.c_float)
        query = (
            self._lib.mvt_query_aabb
            if self.simd
            else self._lib.mvt_query_aabb_scalar
        )
        count = query(
            self._handle,
            center.ctypes.data_as(fp),
            half.ctypes.data_as(fp),
            self._output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._capacity,
        )
        if count < 0:
            raise RuntimeError(f"native MVT query failed with code {count}")
        return self._output[:count].copy()

    def query_sphere(self, center: np.ndarray, radius: float) -> np.ndarray:
        return self.query_aabb(
            center, np.full(3, float(radius) + self.query_padding)
        )

    def query_ellipsoid(self, center: np.ndarray, shape: np.ndarray) -> np.ndarray:
        return self.query_aabb(
            center,
            np.sqrt(np.maximum(np.diag(shape), 0.0)) + self.query_padding,
        )

    def close(self):
        if getattr(self, "_handle", None):
            self._lib.mvt_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()


@dataclass(frozen=True)
class NativeMultilevelMVTStats:
    proxy_count: int
    level_count: int
    occupied_cells: int
    index_references: int
    simd_width: int
    base_voxel_size: float
    maximum_query_half_extent: float
    level_voxel_sizes: tuple[float, ...]
    level_proxy_counts: tuple[int, ...]
    cell_lookups_per_query: int


class NativeMultilevelMVT:
    """True multilevel 27-neighbor MVT with AVX2 AABB filtering."""

    @property
    def native_handle(self) -> int:
        """Opaque handle for fused native consumers in the same process."""
        return int(self._handle or 0)

    def __init__(
        self,
        centers: np.ndarray,
        half_extents: np.ndarray,
        base_voxel_size: float,
        maximum_query_half_extent: float,
        *,
        query_padding: float = 0.0,
        simd: bool = True,
    ) -> None:
        if not DLL.exists():
            raise FileNotFoundError(
                f"native MVT DLL is missing; run build_native_mvt.py: {DLL}"
            )
        centers = np.ascontiguousarray(centers, dtype=np.float32).reshape(-1, 3)
        half_extents = np.ascontiguousarray(
            half_extents, dtype=np.float32
        ).reshape(-1, 3)
        if centers.shape != half_extents.shape or not len(centers):
            raise ValueError(
                "centers and half_extents need matching non-empty (N,3) shapes"
            )
        if base_voxel_size <= 0.0 or maximum_query_half_extent < 0.0:
            raise ValueError(
                "base_voxel_size must be positive and maximum query extent nonnegative"
            )
        self._lib = ctypes.CDLL(str(DLL))
        fp = ctypes.POINTER(ctypes.c_float)
        i32p = ctypes.POINTER(ctypes.c_int32)
        self._lib.mvt_multilevel_create.argtypes = [
            fp,
            fp,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self._lib.mvt_multilevel_create.restype = ctypes.c_void_p
        self._lib.mvt_multilevel_destroy.argtypes = [ctypes.c_void_p]
        for name in (
            "mvt_multilevel_query_aabb",
            "mvt_multilevel_query_aabb_scalar",
        ):
            getattr(self._lib, name).argtypes = [
                ctypes.c_void_p,
                fp,
                fp,
                i32p,
                ctypes.c_int32,
            ]
            getattr(self._lib, name).restype = ctypes.c_int32
        self._lib.mvt_multilevel_query_aabb_batch.argtypes = [
            ctypes.c_void_p,
            fp,
            fp,
            ctypes.c_int32,
            ctypes.c_int32,
            i32p,
            i32p,
            ctypes.c_int32,
        ]
        self._lib.mvt_multilevel_query_aabb_batch.restype = ctypes.c_int32
        self._lib.mvt_multilevel_query_sphere.argtypes = [
            ctypes.c_void_p,
            fp,
            ctypes.c_float,
            i32p,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        self._lib.mvt_multilevel_query_sphere.restype = ctypes.c_int32
        self._lib.mvt_multilevel_query_sphere_batch.argtypes = [
            ctypes.c_void_p,
            fp,
            fp,
            ctypes.c_int32,
            ctypes.c_int32,
            i32p,
            i32p,
            ctypes.c_int32,
        ]
        self._lib.mvt_multilevel_query_sphere_batch.restype = ctypes.c_int32
        self._lib.mvt_multilevel_query_sphere_batch_unordered.argtypes = [
            ctypes.c_void_p,
            fp,
            fp,
            ctypes.c_int32,
            ctypes.c_int32,
            i32p,
            i32p,
            ctypes.c_int32,
        ]
        self._lib.mvt_multilevel_query_sphere_batch_unordered.restype = (
            ctypes.c_int32
        )
        for name in (
            "mvt_multilevel_proxy_count",
            "mvt_multilevel_level_count",
            "mvt_multilevel_occupied_cells",
        ):
            getattr(self._lib, name).argtypes = [ctypes.c_void_p]
            getattr(self._lib, name).restype = ctypes.c_int32
        self._lib.mvt_multilevel_index_references.argtypes = [ctypes.c_void_p]
        self._lib.mvt_multilevel_index_references.restype = ctypes.c_int64
        self._lib.mvt_multilevel_level_proxy_count.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
        ]
        self._lib.mvt_multilevel_level_proxy_count.restype = ctypes.c_int32
        self._lib.mvt_multilevel_level_voxel_size.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
        ]
        self._lib.mvt_multilevel_level_voxel_size.restype = ctypes.c_float
        self._lib.mvt_simd_width.restype = ctypes.c_int32
        self._handle = self._lib.mvt_multilevel_create(
            centers.ctypes.data_as(fp),
            half_extents.ctypes.data_as(fp),
            len(centers),
            float(base_voxel_size),
            float(maximum_query_half_extent),
        )
        if not self._handle:
            raise RuntimeError("native multilevel MVT construction failed")
        self._capacity = len(centers)
        self._output = np.empty(self._capacity, dtype=np.int32)
        self.query_padding = float(query_padding)
        self.simd = bool(simd)
        levels = int(self._lib.mvt_multilevel_level_count(self._handle))
        level_sizes = tuple(
            float(self._lib.mvt_multilevel_level_voxel_size(self._handle, index))
            for index in range(levels)
        )
        level_counts = tuple(
            int(self._lib.mvt_multilevel_level_proxy_count(self._handle, index))
            for index in range(levels)
        )
        self.stats = NativeMultilevelMVTStats(
            proxy_count=int(self._lib.mvt_multilevel_proxy_count(self._handle)),
            level_count=levels,
            occupied_cells=int(
                self._lib.mvt_multilevel_occupied_cells(self._handle)
            ),
            index_references=int(
                self._lib.mvt_multilevel_index_references(self._handle)
            ),
            simd_width=int(self._lib.mvt_simd_width()),
            base_voxel_size=float(base_voxel_size),
            maximum_query_half_extent=float(maximum_query_half_extent),
            level_voxel_sizes=level_sizes,
            level_proxy_counts=level_counts,
            cell_lookups_per_query=27 * levels,
        )

    @classmethod
    def from_spheres(
        cls,
        centers,
        radii,
        base_voxel_size,
        maximum_query_half_extent,
        **kwargs,
    ):
        radii = np.asarray(radii, dtype=float).reshape(-1)
        return cls(
            centers,
            np.repeat(radii[:, None], 3, axis=1),
            base_voxel_size,
            maximum_query_half_extent,
            **kwargs,
        )

    @classmethod
    def from_ellipsoids(
        cls,
        centers,
        shapes,
        offsets,
        base_voxel_size,
        maximum_query_half_extent,
        **kwargs,
    ):
        half = np.sqrt(
            np.maximum(np.diagonal(np.asarray(shapes), axis1=1, axis2=2), 0.0)
        )
        half += np.asarray(offsets, dtype=float).reshape(-1, 1)
        return cls(
            centers,
            half,
            base_voxel_size,
            maximum_query_half_extent,
            **kwargs,
        )

    def query_aabb(self, center: np.ndarray, half_extent: np.ndarray) -> np.ndarray:
        center = np.ascontiguousarray(center, dtype=np.float32).reshape(3)
        half = np.ascontiguousarray(half_extent, dtype=np.float32).reshape(3)
        fp = ctypes.POINTER(ctypes.c_float)
        query = (
            self._lib.mvt_multilevel_query_aabb
            if self.simd
            else self._lib.mvt_multilevel_query_aabb_scalar
        )
        count = query(
            self._handle,
            center.ctypes.data_as(fp),
            half.ctypes.data_as(fp),
            self._output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._capacity,
        )
        if count == -3:
            raise ValueError(
                "query exceeds the construction-time maximum half extent"
            )
        if count < 0:
            raise RuntimeError(f"native multilevel MVT query failed with code {count}")
        return self._output[:count].copy()

    def query_sphere(self, center: np.ndarray, radius: float) -> np.ndarray:
        center = np.ascontiguousarray(center, dtype=np.float32).reshape(3)
        effective_radius = np.float32(float(radius) + self.query_padding)
        fp = ctypes.POINTER(ctypes.c_float)
        count = self._lib.mvt_multilevel_query_sphere(
            self._handle,
            center.ctypes.data_as(fp),
            effective_radius,
            self._output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._capacity,
            int(self.simd),
        )
        if count == -3:
            raise ValueError(
                "query exceeds the construction-time maximum half extent"
            )
        if count < 0:
            raise RuntimeError(
                f"native multilevel MVT sphere query failed with code {count}"
            )
        return self._output[:count].copy()

    def query_aabbs(
        self, centers: np.ndarray, half_extents: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        centers = np.ascontiguousarray(centers, dtype=np.float32).reshape(-1, 3)
        half_extents = np.ascontiguousarray(
            half_extents, dtype=np.float32
        ).reshape(-1, 3)
        if centers.shape != half_extents.shape:
            raise ValueError("centers and half_extents must have matching (N,3) shapes")
        if not len(centers):
            return ()
        offsets = np.empty(len(centers) + 1, dtype=np.int32)
        output = np.empty(len(centers) * self._capacity, dtype=np.int32)
        fp = ctypes.POINTER(ctypes.c_float)
        i32p = ctypes.POINTER(ctypes.c_int32)
        written = self._lib.mvt_multilevel_query_aabb_batch(
            self._handle,
            centers.ctypes.data_as(fp),
            half_extents.ctypes.data_as(fp),
            len(centers),
            int(self.simd),
            offsets.ctypes.data_as(i32p),
            output.ctypes.data_as(i32p),
            len(output),
        )
        if written == -3:
            raise ValueError(
                "query exceeds the construction-time maximum half extent"
            )
        if written < 0:
            raise RuntimeError(
                f"native multilevel MVT batch query failed with code {written}"
            )
        if int(offsets[-1]) != written:
            raise RuntimeError("native multilevel MVT returned invalid offsets")
        return tuple(
            output[int(offsets[index]) : int(offsets[index + 1])].copy()
            for index in range(len(centers))
        )

    def query_spheres(
        self, centers: np.ndarray, radii: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        return self._query_spheres_batch(centers, radii, ordered=True)

    def query_spheres_unordered(
        self, centers: np.ndarray, radii: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """Return the same unique candidate sets without proxy-ID sorting.

        This is intended only for a consumer that immediately imposes its own
        deterministic order, such as the exact closest-point kernel.
        """
        return self._query_spheres_batch(centers, radii, ordered=False)

    def _query_spheres_batch(
        self, centers: np.ndarray, radii: np.ndarray, *, ordered: bool
    ) -> tuple[np.ndarray, ...]:
        centers = np.ascontiguousarray(centers, dtype=np.float32).reshape(-1, 3)
        radii = np.ascontiguousarray(
            np.asarray(radii, dtype=float).reshape(-1) + self.query_padding,
            dtype=np.float32,
        )
        if len(centers) != len(radii):
            raise ValueError("centers and radii must have matching lengths")
        if not len(centers):
            return ()
        offsets = np.empty(len(centers) + 1, dtype=np.int32)
        output = np.empty(len(centers) * self._capacity, dtype=np.int32)
        fp = ctypes.POINTER(ctypes.c_float)
        i32p = ctypes.POINTER(ctypes.c_int32)
        query_function = (
            self._lib.mvt_multilevel_query_sphere_batch
            if ordered
            else self._lib.mvt_multilevel_query_sphere_batch_unordered
        )
        written = query_function(
            self._handle,
            centers.ctypes.data_as(fp),
            radii.ctypes.data_as(fp),
            len(centers),
            int(self.simd),
            offsets.ctypes.data_as(i32p),
            output.ctypes.data_as(i32p),
            len(output),
        )
        if written == -3:
            raise ValueError(
                "query exceeds the construction-time maximum half extent"
            )
        if written < 0:
            raise RuntimeError(
                f"native multilevel MVT sphere batch query failed with code {written}"
            )
        if int(offsets[-1]) != written:
            raise RuntimeError("native multilevel MVT returned invalid sphere offsets")
        return tuple(
            output[int(offsets[index]) : int(offsets[index + 1])].copy()
            for index in range(len(centers))
        )

    def query_ellipsoid(self, center: np.ndarray, shape: np.ndarray) -> np.ndarray:
        return self.query_aabb(
            center,
            np.sqrt(np.maximum(np.diag(shape), 0.0)) + self.query_padding,
        )

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.mvt_multilevel_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
