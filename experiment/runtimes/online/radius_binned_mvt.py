"""Query-radius-binned prototype built from existing exact NativeMultilevelMVT tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from native_mvt import NativeMultilevelMVT


@dataclass(frozen=True)
class RadiusBinnedMVTStats:
    class_names: tuple[str, ...]
    class_robot_radius_limits: tuple[float, ...]
    class_level_proxy_counts: tuple[tuple[int, ...], ...]
    class_level_voxel_sizes: tuple[tuple[float, ...], ...]
    proxy_count: int
    index_references: int
    table_count: int
    simd_width: int


class RadiusBinnedMVT:
    """Prototype that routes each robot sphere to a radius-specific MVT.

    Each native table still stores one proxy at one level and uses 27 cells per
    populated level.  The complete proxy set is replicated across radius classes,
    so this prototype trades memory for better level assignment.  It deliberately
    exposes no ``native_handle``; current fused kernels accept only one table.
    """

    def __init__(
        self,
        centers,
        half_extents,
        base_voxel_size,
        query_padding,
        class_limits=(0.025, 0.045, np.inf),
        class_names=("small", "medium", "large"),
        simd=True,
    ):
        if len(class_limits) != len(class_names):
            raise ValueError("class limits and names must match")
        finite_limits = np.asarray(class_limits[:-1], dtype=float)
        if np.any(np.diff(finite_limits) <= 0.0) or not np.isinf(class_limits[-1]):
            raise ValueError("class limits must increase and end at infinity")
        self.class_limits = tuple(float(value) for value in class_limits)
        self.class_names = tuple(class_names)
        self.query_padding = float(query_padding)
        self.simd = bool(simd)
        # The last class uses the largest radius present in the frozen UR5
        # certificate; callers pass it explicitly after construction.
        self._centers = np.asarray(centers, dtype=float)
        self._half = np.asarray(half_extents, dtype=float)
        self._base = float(base_voxel_size)
        self._tables = None
        self.stats = None

    def build(self, robot_radii):
        robot_radii = np.asarray(robot_radii, dtype=float).reshape(-1)
        tables = []
        actual_limits = []
        lower = -np.inf
        for upper in self.class_limits:
            mask = (robot_radii > lower) & (robot_radii <= upper)
            if not np.any(mask):
                raise ValueError(f"empty radius class ({lower}, {upper}]")
            maximum = float(np.max(robot_radii[mask]))
            tables.append(
                NativeMultilevelMVT(
                    self._centers,
                    self._half,
                    base_voxel_size=self._base,
                    maximum_query_half_extent=maximum + self.query_padding,
                    query_padding=self.query_padding,
                    simd=self.simd,
                )
            )
            actual_limits.append(maximum)
            lower = upper
        self._tables = tuple(tables)
        self.stats = RadiusBinnedMVTStats(
            class_names=self.class_names,
            class_robot_radius_limits=tuple(actual_limits),
            class_level_proxy_counts=tuple(
                table.stats.level_proxy_counts for table in self._tables
            ),
            class_level_voxel_sizes=tuple(
                table.stats.level_voxel_sizes for table in self._tables
            ),
            proxy_count=len(self._centers),
            index_references=sum(table.stats.index_references for table in self._tables),
            table_count=len(self._tables),
            simd_width=self._tables[0].stats.simd_width,
        )
        return self

    def _class_index(self, radius):
        for index, upper in enumerate(self.class_limits):
            if radius <= upper + 1.0e-12:
                return index
        raise AssertionError("infinite final class did not accept radius")

    def query_sphere(self, center, radius):
        if self._tables is None:
            raise RuntimeError("build(robot_radii) must be called first")
        return self._tables[self._class_index(float(radius))].query_sphere(center, radius)

    def query_spheres(self, centers, radii):
        if self._tables is None:
            raise RuntimeError("build(robot_radii) must be called first")
        centers = np.asarray(centers, dtype=float).reshape(-1, 3)
        radii = np.asarray(radii, dtype=float).reshape(-1)
        if len(centers) != len(radii):
            raise ValueError("centers and radii must have matching lengths")
        class_indices = np.asarray([self._class_index(float(radius)) for radius in radii])
        result = [None] * len(radii)
        for class_index, table in enumerate(self._tables):
            indices = np.flatnonzero(class_indices == class_index)
            values = table.query_spheres(centers[indices], radii[indices])
            for original, candidates in zip(indices, values):
                result[int(original)] = candidates
        return tuple(result)

    def close(self):
        if self._tables is not None:
            for table in self._tables:
                table.close()
            self._tables = None

    def __del__(self):
        self.close()
