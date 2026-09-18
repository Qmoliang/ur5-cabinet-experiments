"""Matched known-environment sphere/ellipsoid trees for protocol v1.

This module deliberately does not use CenterVox.  Each known obstacle box is
subdivided into fixed cells.  One circumscribed sphere and one circumscribed
axis-aligned ellipsoid are generated from every identical cell, so proxy IDs,
centers, owners, and counts are matched one-to-one.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from model import BoxObstacle


@dataclass(frozen=True)
class KnownMatchedProxySet:
    proxy_ids: np.ndarray
    centers: np.ndarray
    sphere_radii: np.ndarray
    ellipsoid_shapes: np.ndarray
    cell_half_extents: np.ndarray
    owner_indices: np.ndarray
    cell_indices: np.ndarray
    cell_size: float
    snapshot_sha256: str


def build_known_matched_proxy_tree(
    boxes: tuple[BoxObstacle, ...],
    *,
    cell_size: float,
) -> KnownMatchedProxySet:
    """Conservatively enclose every fixed box cell with matched proxies.

    A box cell with half extents ``h`` is enclosed by a sphere of radius
    ``||h||``.  It is also enclosed by the ellipsoid with shape
    ``Q = 3 diag(h^2)`` because every corner satisfies
    ``sum(h_i^2 / (3 h_i^2)) = 1``.  The ellipsoid is therefore thin in the
    wall-normal direction while the matched sphere uses one isotropic radius.
    """

    if cell_size <= 0.0:
        raise ValueError("cell_size must be positive")
    centers: list[np.ndarray] = []
    radii: list[float] = []
    shapes: list[np.ndarray] = []
    half_extents: list[np.ndarray] = []
    owners: list[int] = []
    cell_indices: list[tuple[int, int, int]] = []
    for owner, box in enumerate(boxes):
        box_half = np.asarray(box.half_size, dtype=float)
        counts = np.maximum(1, np.ceil((2.0 * box_half) / cell_size).astype(int))
        step = 2.0 * box_half / counts
        cell_half = 0.5 * step
        axes = [
            np.linspace(
                -box_half[axis] + cell_half[axis],
                box_half[axis] - cell_half[axis],
                counts[axis],
            )
            for axis in range(3)
        ]
        for ix, x in enumerate(axes[0]):
            for iy, y in enumerate(axes[1]):
                for iz, z in enumerate(axes[2]):
                    centers.append(np.asarray(box.center) + np.array([x, y, z]))
                    half_extents.append(cell_half.copy())
                    radii.append(float(np.linalg.norm(cell_half)))
                    shapes.append(np.diag(3.0 * np.maximum(cell_half, 1.0e-9) ** 2))
                    owners.append(owner)
                    cell_indices.append((ix, iy, iz))
    center_array = np.asarray(centers, dtype=float)
    radius_array = np.asarray(radii, dtype=float)
    shape_array = np.asarray(shapes, dtype=float)
    half_array = np.asarray(half_extents, dtype=float)
    owner_array = np.asarray(owners, dtype=np.int32)
    cell_array = np.asarray(cell_indices, dtype=np.int32)
    proxy_ids = np.arange(len(center_array), dtype=np.int64)
    payload = np.ascontiguousarray(center_array).tobytes()
    payload += np.ascontiguousarray(radius_array).tobytes()
    payload += np.ascontiguousarray(shape_array).tobytes()
    payload += np.ascontiguousarray(owner_array).tobytes()
    payload += np.ascontiguousarray(cell_array).tobytes()
    return KnownMatchedProxySet(
        proxy_ids=proxy_ids,
        centers=center_array,
        sphere_radii=radius_array,
        ellipsoid_shapes=shape_array,
        cell_half_extents=half_array,
        owner_indices=owner_array,
        cell_indices=cell_array,
        cell_size=float(cell_size),
        snapshot_sha256=hashlib.sha256(payload).hexdigest(),
    )


def audit_known_proxy_cell_coverage(proxies: KnownMatchedProxySet) -> dict[str, float | int | bool]:
    """Check all eight corners of every source cell against both proxies."""

    signs = np.asarray(
        [
            [sx, sy, sz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    )
    maximum_sphere_violation = -np.inf
    maximum_ellipsoid_violation = -np.inf
    for center, radius, shape, half in zip(
        proxies.centers,
        proxies.sphere_radii,
        proxies.ellipsoid_shapes,
        proxies.cell_half_extents,
    ):
        del center
        corners = signs * half[None, :]
        maximum_sphere_violation = max(
            maximum_sphere_violation,
            float(np.max(np.linalg.norm(corners, axis=1) - radius)),
        )
        inverse = np.linalg.inv(shape)
        values = np.einsum("ni,ij,nj->n", corners, inverse, corners)
        maximum_ellipsoid_violation = max(
            maximum_ellipsoid_violation,
            float(np.max(values - 1.0)),
        )
    return {
        "proxy_count": int(len(proxies.centers)),
        "matched_centers_and_count": bool(
            len(proxies.centers)
            == len(proxies.sphere_radii)
            == len(proxies.ellipsoid_shapes)
        ),
        "maximum_sphere_corner_violation_m": float(maximum_sphere_violation),
        "maximum_ellipsoid_corner_quadratic_violation": float(
            maximum_ellipsoid_violation
        ),
        "sphere_covers_all_cells": bool(maximum_sphere_violation <= 1.0e-12),
        "ellipsoid_covers_all_cells": bool(
            maximum_ellipsoid_violation <= 1.0e-12
        ),
    }

