"""Deterministic inclusion-minimal covers for fixed proxy candidate families.

This module has no robot, target, task or path inputs.  It reduces only a
declared geometric candidate family over a complete CenterVox universe.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import heapq

import numpy as np
from scipy.spatial import cKDTree

from pointcloud_proxy import PointCloudProxySet


@dataclass(frozen=True)
class InclusionMinimalCover:
    selected_candidates: np.ndarray
    owner: np.ndarray
    unique_witness_counts: np.ndarray
    greedy_selected_count: int
    reverse_deleted_count: int


def inclusion_minimal_set_cover(
    coverage: list[np.ndarray], universe_size: int
) -> InclusionMinimalCover:
    """Lazy-greedy cover followed by deterministic reverse deletion."""

    if universe_size <= 0:
        raise ValueError("cover universe must be non-empty")
    normalized = [
        np.unique(np.asarray(indices, dtype=np.int64)) for indices in coverage
    ]
    if any(np.any((indices < 0) | (indices >= universe_size)) for indices in normalized):
        raise ValueError("candidate coverage index is outside the universe")
    uncovered = np.ones(universe_size, dtype=bool)
    heap = [(-len(indices), index) for index, indices in enumerate(normalized)]
    heapq.heapify(heap)
    selected: list[int] = []
    remaining = universe_size
    while remaining:
        if not heap:
            missing = np.flatnonzero(uncovered)
            raise ValueError(
                f"candidate family does not cover {len(missing)} universe items"
            )
        negative_cached_gain, candidate = heapq.heappop(heap)
        gain = int(np.count_nonzero(uncovered[normalized[candidate]]))
        if gain != -negative_cached_gain:
            if gain:
                heapq.heappush(heap, (-gain, candidate))
            continue
        if gain == 0:
            continue
        selected.append(candidate)
        newly_covered = normalized[candidate][uncovered[normalized[candidate]]]
        uncovered[newly_covered] = False
        remaining -= len(newly_covered)

    cover_counts = np.zeros(universe_size, dtype=np.int64)
    for candidate in selected:
        cover_counts[normalized[candidate]] += 1
    retained = np.ones(len(selected), dtype=bool)
    reverse_deleted = 0
    for position in range(len(selected) - 1, -1, -1):
        candidate = selected[position]
        indices = normalized[candidate]
        if len(indices) and np.all(cover_counts[indices] >= 2):
            cover_counts[indices] -= 1
            retained[position] = False
            reverse_deleted += 1
    retained_candidates = np.asarray(
        sorted(
            selected[position]
            for position in range(len(selected))
            if retained[position]
        ),
        dtype=np.int64,
    )
    if len(retained_candidates) == 0 or np.any(cover_counts < 1):
        raise AssertionError("reverse deletion opened the cover")

    owner = np.full(universe_size, -1, dtype=np.int64)
    for proxy_index, candidate in enumerate(retained_candidates):
        indices = normalized[int(candidate)]
        unowned = indices[owner[indices] < 0]
        owner[unowned] = proxy_index
    if np.any(owner < 0):
        raise AssertionError("retained cover failed to assign every universe item")

    final_counts = np.zeros(universe_size, dtype=np.int64)
    for candidate in retained_candidates:
        final_counts[normalized[int(candidate)]] += 1
    unique_witness_counts = np.asarray(
        [
            np.count_nonzero(final_counts[normalized[int(candidate)]] == 1)
            for candidate in retained_candidates
        ],
        dtype=np.int64,
    )
    if np.any(unique_witness_counts <= 0):
        raise AssertionError("retained candidate has no unique witness")
    return InclusionMinimalCover(
        selected_candidates=retained_candidates,
        owner=owner,
        unique_witness_counts=unique_witness_counts,
        greedy_selected_count=len(selected),
        reverse_deleted_count=reverse_deleted,
    )


def ellipsoid_candidate_coverage(
    candidates: PointCloudProxySet,
    *,
    cell_points: np.ndarray,
    cell_uncertainty_shapes: np.ndarray,
    cell_offsets: np.ndarray,
    tolerance: float = 1.0e-9,
) -> list[np.ndarray]:
    """Return a conservative CenterVox inclusion set per ellipsoid candidate.

    A cell certificate p_i + E(U_i) + B(delta_i) is accepted only if

    * p_i is inside the candidate core E(c_j,Q_j),
    * U_j - U_i is positive semidefinite, and
    * delta_j >= delta_i.

    These three conditions are sufficient for complete Minkowski-set
    inclusion and avoid a sampled-direction approximation.
    """

    points = np.asarray(cell_points, dtype=float).reshape(-1, 3)
    cell_u = np.asarray(cell_uncertainty_shapes, dtype=float).reshape(-1, 3, 3)
    cell_delta = np.asarray(cell_offsets, dtype=float).reshape(-1)
    if len(points) == 0 or len(cell_u) != len(points) or len(cell_delta) != len(points):
        raise ValueError("invalid CenterVox certificate arrays")
    centers = np.asarray(candidates.centers, dtype=float).reshape(-1, 3)
    core_shapes = np.asarray(
        candidates.base_ellipsoid_shapes
        if candidates.base_ellipsoid_shapes is not None
        else candidates.ellipsoid_shapes,
        dtype=float,
    ).reshape(-1, 3, 3)
    proxy_u = (
        np.zeros_like(core_shapes)
        if candidates.proxy_uncertainty_shapes is None
        else np.asarray(candidates.proxy_uncertainty_shapes, dtype=float).reshape(-1, 3, 3)
    )
    proxy_delta = (
        np.zeros(len(centers), dtype=float)
        if candidates.proxy_offset_radii is None
        else np.asarray(candidates.proxy_offset_radii, dtype=float).reshape(-1)
    )
    if not (
        len(core_shapes) == len(centers)
        and len(proxy_u) == len(centers)
        and len(proxy_delta) == len(centers)
    ):
        raise ValueError("invalid ellipsoid candidate arrays")

    cell_support = np.sqrt(
        np.maximum(np.linalg.eigvalsh(cell_u)[:, -1], 0.0)
    ) + cell_delta
    maximum_cell_support = float(np.max(cell_support))
    tree = cKDTree(points)
    coverage: list[np.ndarray] = []
    for center, shape, uncertainty, offset in zip(
        centers, core_shapes, proxy_u, proxy_delta
    ):
        values, rotation = np.linalg.eigh(0.5 * (shape + shape.T))
        values = np.maximum(values, 1.0e-18)
        candidate_radius = (
            float(np.sqrt(values[-1]))
            + float(np.sqrt(max(np.linalg.eigvalsh(uncertainty)[-1], 0.0)))
            + float(offset)
        )
        neighbours = np.asarray(
            sorted(
                tree.query_ball_point(
                    center, candidate_radius + maximum_cell_support + tolerance
                )
            ),
            dtype=np.int64,
        )
        if not len(neighbours):
            coverage.append(neighbours)
            continue
        local = (points[neighbours] - center) @ rotation
        core_values = np.sum(np.square(local) / values[None, :], axis=1)
        inside_core = core_values <= 1.0 + tolerance
        offset_ok = cell_delta[neighbours] <= float(offset) + tolerance
        uncertainty_values, uncertainty_rotation = np.linalg.eigh(
            0.5 * (uncertainty + uncertainty.T)
        )
        uncertainty_values = np.maximum(uncertainty_values, 1.0e-18)
        inverse_root = (
            uncertainty_rotation
            @ np.diag(1.0 / np.sqrt(uncertainty_values))
            @ uncertainty_rotation.T
        )
        normalized_cell_u = np.einsum(
            "ij,njk,kl->nil",
            inverse_root,
            cell_u[neighbours],
            inverse_root,
        )
        normalized_cell_u = 0.5 * (
            normalized_cell_u + normalized_cell_u.swapaxes(1, 2)
        )
        uncertainty_ok = (
            np.linalg.eigvalsh(normalized_cell_u)[:, -1]
            <= 1.0 + tolerance
        )
        coverage.append(neighbours[inside_core & offset_ok & uncertainty_ok])
    return coverage


def reduce_ellipsoid_candidates(
    candidates: PointCloudProxySet,
    *,
    cell_points: np.ndarray | None = None,
    cell_uncertainty_shapes: np.ndarray | None = None,
    cell_offsets: np.ndarray | None = None,
    tolerance: float = 1.0e-9,
) -> tuple[PointCloudProxySet, InclusionMinimalCover]:
    """Publish an inclusion-minimal subset of fixed ellipsoid candidates."""

    points = np.asarray(
        candidates.filtered_points if cell_points is None else cell_points,
        dtype=float,
    ).reshape(-1, 3)
    uncertainty = (
        np.zeros((len(points), 3, 3), dtype=float)
        if cell_uncertainty_shapes is None
        and candidates.filtered_uncertainty_shapes is None
        else np.asarray(
            candidates.filtered_uncertainty_shapes
            if cell_uncertainty_shapes is None
            else cell_uncertainty_shapes,
            dtype=float,
        ).reshape(-1, 3, 3)
    )
    offsets = (
        np.zeros(len(points), dtype=float)
        if cell_offsets is None and candidates.filtered_point_offsets is None
        else np.asarray(
            candidates.filtered_point_offsets if cell_offsets is None else cell_offsets,
            dtype=float,
        ).reshape(-1)
    )
    coverage = ellipsoid_candidate_coverage(
        candidates,
        cell_points=points,
        cell_uncertainty_shapes=uncertainty,
        cell_offsets=offsets,
        tolerance=tolerance,
    )
    cover = inclusion_minimal_set_cover(coverage, len(points))
    selected = cover.selected_candidates

    def subset(value):
        return None if value is None else np.asarray(value)[selected].copy()

    reduced = replace(
        candidates,
        centers=subset(candidates.centers),
        sphere_radii=subset(candidates.sphere_radii),
        ellipsoid_shapes=subset(candidates.ellipsoid_shapes),
        rotations=subset(candidates.rotations),
        cluster_keys=subset(candidates.cluster_keys),
        filtered_cluster_indices=cover.owner.copy(),
        proxy_offset_radii=subset(candidates.proxy_offset_radii),
        proxy_uncertainty_shapes=subset(candidates.proxy_uncertainty_shapes),
        base_sphere_radii=subset(candidates.base_sphere_radii),
        base_ellipsoid_shapes=subset(candidates.base_ellipsoid_shapes),
        ellipsoid_outer_shapes=subset(candidates.ellipsoid_outer_shapes),
        proxy_ids=subset(candidates.proxy_ids),
        certificate_cover_mode="adaptive_irredundant_ellipsoid",
        certificate_cover_candidate_count=len(candidates.centers),
        certificate_cover_selected_count=len(selected),
        certificate_cover_reverse_deleted=cover.reverse_deleted_count,
        certificate_cover_minimum_unique_witnesses=int(
            np.min(cover.unique_witness_counts)
        ),
    )
    return reduced, cover
