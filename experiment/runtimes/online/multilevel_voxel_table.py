"""Sparse three-level voxel table for proxy broad-phase queries.

This is a deterministic Python reference structure inspired by VCC's MVT.
It deliberately uses one uniform resolution: the three levels are sparse
X/Y/Z indexing, not a multiresolution octree.  SIMD pools remain future native
optimization; the correctness role here is to return a conservative superset
of proxies whose AABBs overlap a robot query AABB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VoxelTableStats:
    voxel_size: float
    proxy_count: int
    occupied_cells: int
    x_tables: int
    xy_tables: int
    proxy_cell_references: int


class MultilevelVoxelTable:
    def __init__(
        self,
        centers: np.ndarray,
        half_extents: np.ndarray,
        voxel_size: float,
        query_padding: float = 0.16,
    ) -> None:
        self.centers = np.asarray(centers, dtype=float)
        self.half_extents = np.asarray(half_extents, dtype=float)
        self.voxel_size = float(voxel_size)
        self.query_padding = float(query_padding)
        if self.centers.shape != self.half_extents.shape or self.centers.shape[1:] != (3,):
            raise ValueError("centers and half_extents must both have shape (N, 3)")
        if self.voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")
        lower = self.centers - self.half_extents
        self.origin = np.min(lower, axis=0) - self.voxel_size
        self._x: dict[int, dict[int, dict[int, list[int]]]] = {}
        references = 0
        for index, (center, half) in enumerate(zip(self.centers, self.half_extents)):
            lo = self._index(center - half)
            hi = self._index(center + half)
            for x in range(lo[0], hi[0] + 1):
                y_table = self._x.setdefault(x, {})
                for y in range(lo[1], hi[1] + 1):
                    z_table = y_table.setdefault(y, {})
                    for z in range(lo[2], hi[2] + 1):
                        z_table.setdefault(z, []).append(index)
                        references += 1
        occupied = sum(
            len(z_table)
            for y_table in self._x.values()
            for z_table in y_table.values()
        )
        xy_tables = sum(len(y_table) for y_table in self._x.values())
        self.stats = VoxelTableStats(
            voxel_size=self.voxel_size,
            proxy_count=len(self.centers),
            occupied_cells=occupied,
            x_tables=len(self._x),
            xy_tables=xy_tables,
            proxy_cell_references=references,
        )

    @classmethod
    def from_spheres(
        cls,
        centers: np.ndarray,
        radii: np.ndarray,
        voxel_size: float,
        query_padding: float = 0.16,
    ) -> "MultilevelVoxelTable":
        radii = np.asarray(radii, dtype=float)
        return cls(
            centers,
            np.repeat(radii[:, None], 3, axis=1),
            voxel_size,
            query_padding,
        )

    @classmethod
    def from_ellipsoids(
        cls,
        centers: np.ndarray,
        shapes: np.ndarray,
        voxel_size: float,
        query_padding: float = 0.16,
        offset_radii: np.ndarray | float = 0.0,
    ) -> "MultilevelVoxelTable":
        # Axis-aligned support of E={c+x | x^T Q^-1 x<=1} is sqrt(diag(Q)).
        half_extents = np.sqrt(np.maximum(np.diagonal(shapes, axis1=1, axis2=2), 0.0))
        offsets = np.broadcast_to(np.asarray(offset_radii, dtype=float), (len(centers),))
        half_extents = half_extents + offsets[:, None]
        return cls(centers, half_extents, voxel_size, query_padding)

    def _index(self, point: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(point) - self.origin) / self.voxel_size).astype(int)

    def query_aabb(self, center: np.ndarray, half_extent: np.ndarray) -> np.ndarray:
        half = np.asarray(half_extent, dtype=float) + self.query_padding
        lo = self._index(np.asarray(center) - half)
        hi = self._index(np.asarray(center) + half)
        found: set[int] = set()
        for x in range(lo[0], hi[0] + 1):
            y_table = self._x.get(x)
            if y_table is None:
                continue
            for y in range(lo[1], hi[1] + 1):
                z_table = y_table.get(y)
                if z_table is None:
                    continue
                for z in range(lo[2], hi[2] + 1):
                    found.update(z_table.get(z, ()))
        return np.fromiter(sorted(found), dtype=int)

    def query_sphere(self, center: np.ndarray, radius: float) -> np.ndarray:
        return self.query_aabb(center, np.full(3, float(radius)))

    def query_ellipsoid(self, center: np.ndarray, shape: np.ndarray) -> np.ndarray:
        half = np.sqrt(np.maximum(np.diag(shape), 0.0))
        return self.query_aabb(center, half)
