"""Deterministic full-volume sphere and ellipsoid covers of the known drawer boxes."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import numpy as np


@dataclass(frozen=True)
class VolumeCover:
    representation: str
    centers: np.ndarray
    cell_half_extents: np.ndarray
    box_indices: np.ndarray
    proxy_ids: np.ndarray
    sphere_radii: np.ndarray
    ellipsoid_shapes: np.ndarray
    aabb_half_extents: np.ndarray


def subdivide_known_boxes(boxes, maximum_cell_span: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition every closed box into closed 3-D sub-boxes without gaps."""
    if maximum_cell_span <= 0.0:
        raise ValueError("maximum_cell_span must be positive")
    centers: list[np.ndarray] = []
    halves: list[np.ndarray] = []
    owners: list[int] = []
    for box_index, box in enumerate(boxes):
        center = np.asarray(box.center, dtype=float)
        half = np.asarray(box.half_size, dtype=float)
        counts = np.maximum(1, np.ceil((2.0 * half) / maximum_cell_span).astype(int))
        cell_half = half / counts
        axes = [
            center[axis] - half[axis] + cell_half[axis]
            + 2.0 * cell_half[axis] * np.arange(counts[axis])
            for axis in range(3)
        ]
        for xyz in itertools.product(*axes):
            centers.append(np.asarray(xyz, dtype=float))
            halves.append(cell_half.copy())
            owners.append(box_index)
    return np.asarray(centers), np.asarray(halves), np.asarray(owners, dtype=np.int64)


def build_volume_cover(boxes, representation: str, maximum_cell_span: float = 0.075) -> VolumeCover:
    """Build one outer primitive per identical solid sub-box.

    Sphere: its radius is the sub-box half diagonal.
    Ellipsoid: the Loewner outer ellipsoid of a centered 3-D box, with axes sqrt(3) h.
    """
    if representation not in {"sphere", "ellipsoid"}:
        raise ValueError("representation must be sphere or ellipsoid")
    centers, halves, owners = subdivide_known_boxes(boxes, maximum_cell_span)
    radii = np.linalg.norm(halves, axis=1)
    shapes = np.asarray([np.diag(3.0 * half * half) for half in halves])
    if representation == "sphere":
        aabb_half = np.repeat(radii[:, None], 3, axis=1)
    else:
        aabb_half = math.sqrt(3.0) * halves
    return VolumeCover(
        representation=representation,
        centers=centers,
        cell_half_extents=halves,
        box_indices=owners,
        proxy_ids=np.arange(len(centers), dtype=np.int64),
        sphere_radii=radii,
        ellipsoid_shapes=shapes,
        aabb_half_extents=aabb_half,
    )


def coverage_audit(cover: VolumeCover) -> dict:
    """Prove each complete solid sub-box is inside its assigned convex primitive."""
    signs = np.asarray(tuple(itertools.product((-1.0, 1.0), repeat=3)), dtype=float)
    maximum_measure = 0.0
    failed: list[int] = []
    for index, (half, radius, shape) in enumerate(
        zip(cover.cell_half_extents, cover.sphere_radii, cover.ellipsoid_shapes)
    ):
        offsets = signs * half[None, :]
        if cover.representation == "sphere":
            values = np.linalg.norm(offsets, axis=1) / radius
        else:
            inverse = np.linalg.inv(shape)
            values = np.einsum("ni,ij,nj->n", offsets, inverse, offsets)
        local_maximum = float(np.max(values))
        maximum_measure = max(maximum_measure, local_maximum)
        if local_maximum > 1.0 + 1e-10:
            failed.append(index)

    box_volume = float(np.sum(8.0 * np.prod(cover.cell_half_extents, axis=1)))
    if cover.representation == "sphere":
        primitive_volume = float(np.sum((4.0 * math.pi / 3.0) * cover.sphere_radii**3))
        measure_name = "corner_distance_divided_by_radius"
    else:
        axes = math.sqrt(3.0) * cover.cell_half_extents
        primitive_volume = float(np.sum((4.0 * math.pi / 3.0) * np.prod(axes, axis=1)))
        measure_name = "corner_mahalanobis_squared"
    return {
        "representation": cover.representation,
        "proof": "all 8 vertices of every solid sub-box lie in its assigned convex primitive",
        "measure_name": measure_name,
        "maximum_corner_measure": maximum_measure,
        "tolerance": 1e-10,
        "failed_cell_count": len(failed),
        "failed_cell_indices": failed,
        "complete_solid_cell_volume_covered": len(failed) == 0,
        "cell_count": len(cover.centers),
        "known_box_volume_sum_m3": box_volume,
        "primitive_volume_sum_m3": primitive_volume,
        "primitive_to_box_volume_sum_ratio": primitive_volume / box_volume,
        "minimum_aabb_half_extent_m": np.min(cover.aabb_half_extents, axis=0).tolist(),
        "median_aabb_half_extent_m": np.median(cover.aabb_half_extents, axis=0).tolist(),
        "maximum_aabb_half_extent_m": np.max(cover.aabb_half_extents, axis=0).tolist(),
    }
