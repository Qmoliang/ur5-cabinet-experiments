"""Causal online CenterVox and matched sphere/ellipsoid proxy snapshots."""

from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import hashlib
import time

import numpy as np
from scipy.spatial import cKDTree

from irredundant_proxy_cover import reduce_ellipsoid_candidates

from pointcloud_proxy import (
    PointCloudProxySet,
    certified_translated_ellipsoid_outer,
    fit_matched_voxel_proxies,
    loewner_union_outer_shape,
    loewner_union_outer_shapes_grouped,
    loewner_union_outer_shapes_grouped_common_frame,
    minkowski_outer_shape,
    minkowski_outer_shapes,
)


@dataclass(frozen=True)
class ProxyUpdateStats:
    generation: int
    inserted_points: int
    center_voxels: int
    proxy_count: int
    changed_center_voxels: int
    update_ms: float
    snapshot_sha256: str
    frame_centervox_ms: float = 0.0
    history_union_ms: float = 0.0
    cell_merge_ms: float = 0.0
    bucket_fit_ms: float = 0.0
    publish_ids_hash_ms: float = 0.0
    sphere_cover_candidate_count: int = 0
    sphere_cover_selected_count: int = 0
    sphere_cover_reverse_deleted: int = 0
    certificate_cover_candidate_count: int = 0
    certificate_cover_selected_count: int = 0
    certificate_cover_reverse_deleted: int = 0
    certificate_cover_minimum_unique_witnesses: int = 0
    contained_cell_reuses: int = 0


@dataclass(frozen=True)
class ProxyCoverageAudit:
    raw_sample_bins: int
    center_voxels: int
    proxy_count: int
    minimum_centervox_cover_slack: float
    minimum_proxy_offset_slack: float
    maximum_base_ellipsoid_value: float
    minimum_sphere_certificate_slack: float
    all_raw_sample_balls_certified: bool
    minimum_measurement_uncertainty_slack: float = np.inf
    minimum_proxy_uncertainty_slack: float = np.inf
    sphere_cover_inclusion_minimal: bool = True
    sphere_cover_candidate_count: int = 0
    sphere_cover_selected_count: int = 0
    sphere_cover_reverse_deleted: int = 0
    certificate_cover_inclusion_minimal: bool = True
    certificate_cover_candidate_count: int = 0
    certificate_cover_selected_count: int = 0
    certificate_cover_reverse_deleted: int = 0
    certificate_cover_minimum_unique_witnesses: int = 0


@dataclass
class _CenterSample:
    point: np.ndarray
    cover_radius: float
    uncertainty_shape: np.ndarray | None
    source_count: int = 1
    source_generations: frozenset[int] = frozenset()


@dataclass
class _CenterVoxel:
    representative: np.ndarray
    representative_distance: float
    cover_radius: float
    uncertainty_shape: np.ndarray | None
    samples: dict[tuple[int, int, int], _CenterSample]
    fusion_rotation: np.ndarray | None = None
    fusion_lower: np.ndarray | None = None
    fusion_upper: np.ndarray | None = None
    fusion_narrow_axis: int | None = None


@dataclass(frozen=True)
class _BucketFit:
    """Cached certified fit for one fixed world-aligned proxy bucket."""

    keys: tuple[tuple[int, int, int], ...]
    proxies: PointCloudProxySet


class IncrementalMatchedProxyManager:
    """Maintain stable CenterVox cells and publish matched proxy snapshots.

    The same ``PointCloudProxySet`` contains both sphere radii and ellipsoid
    shapes, so a generation can never accidentally use different clusters or
    centers for the two controller variants.
    """

    def __init__(
        self,
        *,
        filter_size: float = 0.006,
        cluster_size: float = 0.10,
        maximum_aabb_overshoot: float = 0.025,
        maximum_uncertainty_union_inflation: float | None = None,
        normal_connection_distance: float | None = None,
        certificate_radius_limit: float | None = None,
        radius_limit_representation: str = "both",
        uncertainty_fusion_mode: str = "separate_uncertainty",
        direct_thin_axis_inflation: float = np.sqrt(2.0),
        direct_tangent_subdivisions: int = 1,
        direct_partition_mode: str = "grid",
        sphere_cover_mode: str = "matched",
        ellipsoid_cover_mode: str = "matched",
        minimum_core_semi_axis: float = 0.0,
        origin: np.ndarray = np.array([-1.2, -1.2, 0.0]),
    ) -> None:
        if filter_size <= 0.0 or cluster_size <= 0.0:
            raise ValueError("filter_size and cluster_size must be positive")
        self.filter_size = float(filter_size)
        self.cluster_size = float(cluster_size)
        self.maximum_aabb_overshoot = float(maximum_aabb_overshoot)
        if minimum_core_semi_axis < 0.0:
            raise ValueError("minimum core semi-axis must be non-negative")
        self.minimum_core_semi_axis = float(minimum_core_semi_axis)
        self.maximum_uncertainty_union_inflation = (
            None
            if maximum_uncertainty_union_inflation is None
            else float(maximum_uncertainty_union_inflation)
        )
        if (
            self.maximum_uncertainty_union_inflation is not None
            and self.maximum_uncertainty_union_inflation < 1.0
        ):
            raise ValueError("maximum uncertainty union inflation must be >= 1")
        self.normal_connection_distance = (
            1.75 * self.filter_size
            if normal_connection_distance is None
            else float(normal_connection_distance)
        )
        if certificate_radius_limit is not None and certificate_radius_limit <= 0.0:
            raise ValueError("certificate radius limit must be positive")
        self.certificate_radius_limit = (
            None
            if certificate_radius_limit is None
            else float(certificate_radius_limit)
        )
        if radius_limit_representation not in {"both", "sphere", "ellipsoid"}:
            raise ValueError("unknown radius-limit representation")
        self.radius_limit_representation = radius_limit_representation
        if uncertainty_fusion_mode not in {
            "separate_uncertainty",
            "support_interval_uncertainty",
            "raw_join_uncertainty",
            "partitioned_separate_uncertainty",
            "raw_directional_uncertainty",
            "fused_certified_ellipsoid",
            "fused_certified_centervox",
        }:
            raise ValueError("unknown uncertainty_fusion_mode")
        self.uncertainty_fusion_mode = uncertainty_fusion_mode
        self.direct_thin_axis_inflation = float(direct_thin_axis_inflation)
        self.direct_tangent_subdivisions = int(direct_tangent_subdivisions)
        if self.direct_tangent_subdivisions < 1:
            raise ValueError("direct tangent subdivisions must be positive")
        if direct_partition_mode not in {"grid", "longest_tangent_binary"}:
            raise ValueError("unknown direct partition mode")
        self.direct_partition_mode = direct_partition_mode
        if sphere_cover_mode not in {
            "matched",
            "adaptive_irredundant",
            "cap_irredundant",
        }:
            raise ValueError("unknown sphere cover mode")
        self.sphere_cover_mode = sphere_cover_mode
        if ellipsoid_cover_mode not in {"matched", "adaptive_irredundant"}:
            raise ValueError("unknown ellipsoid cover mode")
        self.ellipsoid_cover_mode = ellipsoid_cover_mode
        if (
            self.sphere_cover_mode != "matched"
            and self.ellipsoid_cover_mode != "matched"
        ):
            raise ValueError("one publication cannot reduce both representations")
        if self.sphere_cover_mode in {
            "adaptive_irredundant",
            "cap_irredundant",
        }:
            if self.certificate_radius_limit is None:
                raise ValueError(
                    "adaptive irredundant spheres require a candidate radius limit"
                )
        if (
            self.uncertainty_fusion_mode == "fused_certified_centervox"
            and self.sphere_cover_mode == "matched"
        ):
            if self.direct_thin_axis_inflation <= 1.0:
                raise ValueError("direct thin-axis inflation must exceed one")
        elif (
            min(
                abs(self.direct_thin_axis_inflation - 1.0),
                abs(self.direct_thin_axis_inflation - np.sqrt(2.0)),
            )
            > 1.0e-12
            or self.direct_tangent_subdivisions != 1
            or self.direct_partition_mode != "grid"
        ):
            raise ValueError(
                "inactive direct thin-axis controls must use identity 1.0 "
                "or the legacy default sqrt(2)"
            )
        self.origin = np.asarray(origin, dtype=float).reshape(3)
        self._cells: dict[tuple[int, int, int], _CenterVoxel] = {}
        # v5.3 keeps the spatial CenterVox dictionary separate from the
        # flattened directional subcertificates consumed by proxy fitting.
        # Other modes continue to use ``_cells`` exactly as before.
        self._spatial_cells: dict[tuple[int, int, int], _CenterVoxel] = {}
        from raw_directional_uncertainty import RawDirectionalJoin
        self.raw_directional_join = RawDirectionalJoin()
        self._next_directional_leaf_id = 0
        from raw_support_intervals import RawSupportIntervals, RawLoewnerJoin
        self.raw_support_intervals = RawLoewnerJoin() if self.uncertainty_fusion_mode == "raw_join_uncertainty" else RawSupportIntervals()
        self._generation = -1
        self._snapshot: PointCloudProxySet | None = None
        self._snapshot_hash = ""
        self._directional: bool | None = None
        self._sample_bin = max(0.00025, self.filter_size / 64.0)
        self._next_proxy_id = 0
        self._previous_memberships: dict[int, frozenset[tuple[int, int, int]]] = {}
        self._last_ordered_keys: tuple[tuple[int, int, int], ...] = ()
        self._bucket_fits: dict[tuple[int, int, int], _BucketFit] = {}
        self._direct_subproxy_ids: dict[
            tuple[tuple[int, int, int], int], int
        ] = {}

    @property
    def _direct_subproxy_count(self) -> int:
        if self.direct_partition_mode == "longest_tangent_binary":
            return 2
        return self.direct_tangent_subdivisions**2

    @property
    def _uses_fused_uncertainty(self) -> bool:
        return self.uncertainty_fusion_mode in {
            "fused_certified_ellipsoid",
            "fused_certified_centervox",
        }

    @property
    def _uses_partitioned_uncertainty(self) -> bool:
        return self.uncertainty_fusion_mode in {"partitioned_separate_uncertainty", "raw_directional_uncertainty"}

    def _publish_fused_centervox_snapshot(
        self, ordered_keys: list[tuple[int, int, int]]
    ) -> PointCloudProxySet:
        """Publish a certified thin-ellipsoid cover per occupied CenterVox.

        The retained support box is partitioned only in its two tangent
        coordinates. Every sub-box is enclosed by one ellipsoid whose normal
        semi-axis uses the frozen direct thin-axis factor. Coverage is paid for
        with more tangent-shifted proxies instead of repeated core thickening.
        """

        cell_centers = np.asarray(
            [self._cells[key].representative for key in ordered_keys], dtype=float
        )
        cell_shapes = np.asarray(
            [self._cells[key].uncertainty_shape for key in ordered_keys], dtype=float
        )
        rotations_box = np.asarray(
            [self._cells[key].fusion_rotation for key in ordered_keys], dtype=float
        )
        lowers = np.asarray(
            [self._cells[key].fusion_lower for key in ordered_keys], dtype=float
        )
        uppers = np.asarray(
            [self._cells[key].fusion_upper for key in ordered_keys], dtype=float
        )
        narrow = np.asarray(
            [self._cells[key].fusion_narrow_axis for key in ordered_keys],
            dtype=np.int64,
        )
        if (
            rotations_box.shape != (len(ordered_keys), 3, 3)
            or lowers.shape != (len(ordered_keys), 3)
            or uppers.shape != (len(ordered_keys), 3)
        ):
            raise AssertionError("direct CenterVox lost its support-box state")

        subdivisions = self.direct_tangent_subdivisions
        subproxy_count = self._direct_subproxy_count
        owner = np.repeat(
            np.arange(len(ordered_keys), dtype=np.int64), subproxy_count
        )
        local_subindex = np.tile(
            np.arange(subproxy_count, dtype=np.int64), len(ordered_keys)
        )
        local_centers = np.repeat(0.5 * (lowers + uppers), subproxy_count, axis=0)
        cell_half = np.maximum(0.5 * (uppers - lowers), 1.0e-7)
        half = np.repeat(
            cell_half,
            subproxy_count,
            axis=0,
        )
        tangent = np.asarray(
            [[axis for axis in range(3) if axis != int(n)] for n in narrow],
            dtype=np.int64,
        )
        rows = np.arange(len(owner), dtype=np.int64)
        if self.direct_partition_mode == "longest_tangent_binary":
            tangent_half = cell_half[
                np.arange(len(ordered_keys))[:, None], tangent
            ]
            split_slot = np.argmax(tangent_half, axis=1)
            axes_index = tangent[
                owner, split_slot[owner]
            ]
            span = uppers[owner, axes_index] - lowers[owner, axes_index]
            local_centers[rows, axes_index] = (
                lowers[owner, axes_index]
                + (local_subindex + 0.5) * span / 2.0
            )
            half[rows, axes_index] = np.maximum(
                0.25 * span, 1.0e-7
            )
        else:
            tangent_bins = np.column_stack(
                (local_subindex // subdivisions, local_subindex % subdivisions)
            )
            for tangent_slot in range(2):
                axes_index = tangent[owner, tangent_slot]
                span = uppers[owner, axes_index] - lowers[owner, axes_index]
                local_centers[rows, axes_index] = (
                    lowers[owner, axes_index]
                    + (tangent_bins[:, tangent_slot] + 0.5)
                    * span
                    / subdivisions
                )
                half[rows, axes_index] = np.maximum(
                    0.5 * span / subdivisions, 1.0e-7
                )

        narrow_weight = 1.0 / self.direct_thin_axis_inflation**2
        tangent_budget = 1.0 - narrow_weight
        weights = np.zeros_like(half)
        weights[rows, narrow[owner]] = narrow_weight
        if self.direct_partition_mode == "longest_tangent_binary":
            tangent_axes = tangent[owner]
            tangent_half_squared = np.square(
                half[rows[:, None], tangent_axes]
            )
            tangent_denominator = np.sum(tangent_half_squared, axis=1)
            tangent_weights = (
                tangent_budget
                * tangent_half_squared
                / np.maximum(tangent_denominator[:, None], 1.0e-30)
            )
            weights[rows[:, None], tangent_axes] = tangent_weights
        else:
            tangent_weight = 0.5 * tangent_budget
            weights[rows[:, None], tangent[owner]] = tangent_weight
        axes = half / np.sqrt(weights)
        centers = np.einsum(
            "nij,nj->ni", rotations_box[owner], local_centers
        )
        shapes = np.einsum(
            "nik,nk,njk->nij",
            rotations_box[owner],
            np.square(axes),
            rotations_box[owner],
        )
        shapes = 0.5 * (shapes + np.swapaxes(shapes, 1, 2))
        values, rotations = np.linalg.eigh(shapes)
        values = np.maximum(values, 0.0)
        radii = np.sqrt(values[:, -1])
        count = len(centers)
        zeros = np.zeros_like(shapes)
        cluster_keys = np.column_stack(
            (
                np.asarray(ordered_keys, dtype=np.int64)[owner],
                local_subindex,
            )
        )
        return PointCloudProxySet(
            raw_points=cell_centers.copy(),
            filtered_points=cell_centers.copy(),
            centers=centers,
            sphere_radii=radii.copy(),
            ellipsoid_shapes=shapes.copy(),
            rotations=rotations,
            cluster_keys=cluster_keys,
            filtered_cluster_indices=(
                np.arange(len(ordered_keys), dtype=np.int64) * subproxy_count
            ),
            maximum_ellipsoid_overshoot=0.0,
            surface_cover_radius=float(np.max(radii)),
            proxy_offset_radii=np.zeros(count, dtype=float),
            filtered_uncertainty_shapes=cell_shapes.copy(),
            filtered_point_offsets=np.asarray(
                [self._cells[key].cover_radius for key in ordered_keys],
                dtype=float,
            ),
            proxy_uncertainty_shapes=zeros,
            base_sphere_radii=radii.copy(),
            base_ellipsoid_shapes=shapes.copy(),
            ellipsoid_outer_shapes=shapes.copy(),
            maximum_uncertainty_union_inflation=1.0,
            certificate_radius_limit=None,
            uncertainty_fusion_mode=self.uncertainty_fusion_mode,
        )

    def _publish_adaptive_irredundant_sphere_snapshot(
        self,
        ordered_keys: list[tuple[int, int, int]],
        candidate_centers: np.ndarray,
        candidate_radii: np.ndarray,
    ) -> PointCloudProxySet:
        """Cover CenterVox certificates with adaptive irredundant balls.

        For the formal radius-cap family, the configured certificate radius is
        the preregistered fixed candidate radius. Candidate centres come from
        deterministic fixed-world-bucket subdivision. After set cover and
        reverse deletion, each retained ball is shrunk to the smallest support
        required by the cells assigned to that centre.
        The coverage universe is the full persistent CenterVox map. For a
        CenterVox certificate

            E(p_i, U_i) (+) B(0, delta_i),

        a candidate ball B(c_j, r_j) is accepted only when

            ||p_i-c_j|| + sqrt(lambda_max(U_i)) + delta_i <= r_j.

        Thus coverage is proved for the complete directional measurement set,
        not only its representative point. Lazy greedy set cover selects an
        approximate minimum-cardinality cover, a reverse pass removes every
        deletable selection, and retained balls are finally shrunk to the
        exact support required by their assigned cells. This is map-level
        certificate reduction, independent of LiuQP ordered erase-remove.
        """

        if self.certificate_radius_limit is None:
            raise AssertionError("sphere set cover lost its candidate radius")
        cap = float(self.certificate_radius_limit)
        cells = [self._cells[key] for key in ordered_keys]
        cell_centers = np.asarray(
            [cell.representative for cell in cells], dtype=float
        )
        if any(cell.uncertainty_shape is None for cell in cells):
            raise AssertionError(
                "adaptive directional sphere cover lost CenterVox U"
            )
        cell_shapes = np.asarray(
            [cell.uncertainty_shape for cell in cells], dtype=float
        )
        cell_support_radii = np.sqrt(
            np.maximum(np.linalg.eigvalsh(cell_shapes)[:, -1], 0.0)
        ) + np.asarray([cell.cover_radius for cell in cells], dtype=float)
        if float(np.max(cell_support_radii)) > cap + 1.0e-12:
            index = int(np.argmax(cell_support_radii))
            raise ValueError(
                "one indivisible CenterVox certificate exceeds the sphere "
                f"candidate radius: key={ordered_keys[index]}, "
                f"required={cell_support_radii[index]:.9g}, limit={cap:.9g}"
            )

        candidate_centers = np.asarray(candidate_centers, dtype=float).reshape(-1, 3)
        candidate_radii = np.asarray(candidate_radii, dtype=float).reshape(-1)
        if len(candidate_centers) == 0:
            raise AssertionError("sphere fitter produced no candidate centres")
        if len(candidate_radii) != len(candidate_centers):
            raise ValueError("one adaptive radius is required per candidate centre")
        if np.any(candidate_radii <= 0.0) or np.any(
            candidate_radii > cap + 1.0e-9
        ):
            raise AssertionError("local sphere candidate exceeded its split cap")
        cell_count = len(cells)
        candidate_count = len(candidate_centers)
        tree = cKDTree(cell_centers)
        coverage: list[np.ndarray] = []
        required_radii: list[np.ndarray] = []
        for candidate_center, candidate_radius in zip(
            candidate_centers, candidate_radii
        ):
            neighbours = np.asarray(
                sorted(tree.query_ball_point(candidate_center, candidate_radius)),
                dtype=np.int64,
            )
            required = (
                np.linalg.norm(
                    candidate_center[None, :] - cell_centers[neighbours], axis=1
                )
                + cell_support_radii[neighbours]
            )
            keep = required <= candidate_radius + 1.0e-12
            covered = neighbours[keep]
            covered_required = required[keep]
            coverage.append(covered)
            required_radii.append(covered_required)

        # Deterministic lazy greedy set cover.  Equal gains are resolved by
        # the stable CenterVox key order represented by the candidate index.
        uncovered = np.ones(cell_count, dtype=bool)
        heap = [(-len(indices), index) for index, indices in enumerate(coverage)]
        heapq.heapify(heap)
        selected: list[int] = []
        remaining = cell_count
        while remaining:
            if not heap:
                raise AssertionError("sphere candidate family did not cover all cells")
            negative_cached_gain, candidate = heapq.heappop(heap)
            gain = int(np.count_nonzero(uncovered[coverage[candidate]]))
            if gain != -negative_cached_gain:
                if gain:
                    heapq.heappush(heap, (-gain, candidate))
                continue
            if gain == 0:
                continue
            selected.append(candidate)
            newly_covered = coverage[candidate][uncovered[coverage[candidate]]]
            uncovered[newly_covered] = False
            remaining -= len(newly_covered)

        # Reverse deletion produces an inclusion-minimal selected cover: each
        # retained candidate has at least one support box that no other
        # retained fixed-radius candidate covers.
        cover_counts = np.zeros(cell_count, dtype=np.int64)
        for candidate in selected:
            cover_counts[coverage[candidate]] += 1
        retained_mask = np.ones(len(selected), dtype=bool)
        reverse_deleted = 0
        for position in range(len(selected) - 1, -1, -1):
            candidate = selected[position]
            if np.all(cover_counts[coverage[candidate]] >= 2):
                cover_counts[coverage[candidate]] -= 1
                retained_mask[position] = False
                reverse_deleted += 1
        retained_candidates = np.asarray(
            sorted(
                selected[position]
                for position in range(len(selected))
                if retained_mask[position]
            ),
            dtype=np.int64,
        )
        if np.any(cover_counts < 1):
            raise AssertionError("sphere reverse deletion opened a coverage hole")

        # Assign every box to the retained centre that needs the smallest
        # containing radius.  The final radius is therefore adaptive and is
        # the exact minimum required for that centre's assigned boxes.
        owner = np.full(cell_count, -1, dtype=np.int64)
        owner_required = np.full(cell_count, np.inf, dtype=float)
        for proxy_index, candidate in enumerate(retained_candidates):
            indices = coverage[int(candidate)]
            required = required_radii[int(candidate)]
            better = required < owner_required[indices] - 1.0e-15
            ties = np.abs(required - owner_required[indices]) <= 1.0e-15
            better |= ties & ((owner[indices] < 0) | (proxy_index < owner[indices]))
            update_indices = indices[better]
            owner[update_indices] = proxy_index
            owner_required[update_indices] = required[better]
        if np.any(owner < 0) or not np.all(np.isfinite(owner_required)):
            raise AssertionError("sphere set cover failed to assign every cell")

        final_radii = np.zeros(len(retained_candidates), dtype=float)
        np.maximum.at(final_radii, owner, owner_required)
        if np.any(final_radii <= 0.0) or np.any(final_radii > cap + 1.0e-12):
            raise AssertionError("adaptive sphere shrink produced an invalid radius")
        centers = candidate_centers[retained_candidates]
        identity = np.repeat(np.eye(3)[None, :, :], len(centers), axis=0)
        shapes = identity * np.square(final_radii)[:, None, None]
        zeros = np.zeros_like(shapes)
        cluster_keys = retained_candidates[:, None]
        return PointCloudProxySet(
            raw_points=cell_centers.copy(),
            filtered_points=cell_centers.copy(),
            centers=centers,
            sphere_radii=final_radii,
            ellipsoid_shapes=shapes.copy(),
            rotations=identity,
            cluster_keys=cluster_keys,
            filtered_cluster_indices=owner,
            maximum_ellipsoid_overshoot=0.0,
            surface_cover_radius=float(np.max(final_radii)),
            proxy_offset_radii=np.zeros(len(centers), dtype=float),
            filtered_uncertainty_shapes=cell_shapes,
            filtered_point_offsets=np.asarray(
                [cell.cover_radius for cell in cells], dtype=float
            ),
            proxy_uncertainty_shapes=zeros,
            base_sphere_radii=final_radii.copy(),
            base_ellipsoid_shapes=shapes.copy(),
            ellipsoid_outer_shapes=shapes.copy(),
            maximum_uncertainty_union_inflation=1.0,
            certificate_radius_limit=cap,
            uncertainty_fusion_mode=self.uncertainty_fusion_mode,
            sphere_cover_mode=self.sphere_cover_mode,
            sphere_cover_candidate_count=candidate_count,
            sphere_cover_selected_count=len(centers),
            sphere_cover_reverse_deleted=reverse_deleted,
        )

    def _build_resolution_bounded_sphere_candidates(
        self,
        ordered_keys: list[tuple[int, int, int]],
        fitted: PointCloudProxySet,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Refine local ball candidates to a CenterVox error budget.

        Minimising ball count without limiting isotropic over-approximation
        always favours large balls.  The representation error budget here is
        fixed independently of the task and robot pose: beyond the certified
        per-cell measurement support, a candidate may add at most one
        CenterVox diagonal ``sqrt(3) * h``.  Ellipsoids can retain broad
        tangential support while balls must subdivide, exposing the intended
        fidelity-versus-proxy-count trade-off without selecting a goal-specific
        radius.
        """

        cells = [self._cells[key] for key in ordered_keys]
        centers = np.asarray(
            [cell.representative for cell in cells], dtype=float
        )
        shapes = np.asarray(
            [cell.uncertainty_shape for cell in cells], dtype=float
        )
        support = np.sqrt(
            np.maximum(np.linalg.eigvalsh(shapes)[:, -1], 0.0)
        ) + np.asarray([cell.cover_radius for cell in cells], dtype=float)
        assignments = np.asarray(
            fitted.filtered_cluster_indices, dtype=np.int64
        )
        order = np.argsort(assignments, kind="stable")
        boundaries = np.flatnonzero(np.diff(assignments[order])) + 1
        groups = np.split(order, boundaries)
        candidate_centers: list[np.ndarray] = []
        candidate_radii: list[float] = []
        geometric_budget = np.sqrt(3.0) * self.filter_size
        global_cap = float(self.certificate_radius_limit)

        def refine(indices: np.ndarray, depth: int = 0) -> None:
            points = centers[indices]
            center = 0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
            required = float(
                np.max(np.linalg.norm(points - center, axis=1) + support[indices])
            )
            local_cap = min(
                global_cap,
                float(np.max(support[indices])) + geometric_budget,
            )
            if required > local_cap + 1.0e-12 and len(indices) > 1:
                spread = np.ptp(points, axis=0)
                if float(np.max(spread)) <= 1.0e-15:
                    split_order = np.sort(indices)
                else:
                    axis = int(np.argmax(spread))
                    split_order = indices[
                        np.argsort(points[:, axis], kind="stable")
                    ]
                midpoint = len(split_order) // 2
                if 0 < midpoint < len(split_order) and depth < 32:
                    refine(split_order[:midpoint], depth + 1)
                    refine(split_order[midpoint:], depth + 1)
                    return
            if required > global_cap + 1.0e-9:
                raise ValueError(
                    "resolution-bounded sphere candidate exceeded global cap"
                )
            candidate_centers.append(center)
            candidate_radii.append(required)

        for group in groups:
            refine(np.asarray(group, dtype=np.int64))
        return (
            np.asarray(candidate_centers, dtype=float),
            np.asarray(candidate_radii, dtype=float),
        )

    def _proxy_bucket_key(
        self, representative: np.ndarray
    ) -> tuple[int, int, int]:
        key = np.floor(
            (np.asarray(representative, dtype=float) - self.origin)
            / self.cluster_size
        ).astype(np.int64)
        return int(key[0]), int(key[1]), int(key[2])

    def _build_radius_cap_sphere_candidates(
        self, bucket_keys: list[tuple[int, int, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the formal task-independent fixed-R ball candidate family.

        The input is exactly one fixed-world ``cluster_size`` bucket.  AABB
        midpoint subdivision along the longest point spread is deterministic
        because ties retain the already sorted CenterVox key order.  A node is
        a valid candidate centre once one radius-R ball covers every complete
        CenterVox certificate in that node.  Candidate membership is still
        evaluated against every cell in the bucket by the set-cover routine;
        the recursive groups only guarantee that the declared family covers.
        """

        if self.certificate_radius_limit is None:
            raise AssertionError("radius-cap sphere family requires R")
        cap = float(self.certificate_radius_limit)
        cells = [self._cells[key] for key in bucket_keys]
        if any(cell.uncertainty_shape is None for cell in cells):
            raise AssertionError("radius-cap sphere family lost CenterVox U")
        centers = np.asarray(
            [cell.representative for cell in cells], dtype=float
        )
        shapes = np.asarray(
            [cell.uncertainty_shape for cell in cells], dtype=float
        )
        support = np.sqrt(
            np.maximum(np.linalg.eigvalsh(shapes)[:, -1], 0.0)
        ) + np.asarray([cell.cover_radius for cell in cells], dtype=float)
        if float(np.max(support)) > cap + 1.0e-12:
            raise ValueError("one CenterVox certificate exceeds sphere cap")

        candidate_centers: list[np.ndarray] = []

        def refine(indices: np.ndarray, depth: int = 0) -> None:
            points = centers[indices]
            center = 0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
            required = float(
                np.max(np.linalg.norm(points - center, axis=1) + support[indices])
            )
            if required <= cap + 1.0e-12:
                candidate_centers.append(center)
                return
            if len(indices) <= 1 or depth >= 32:
                raise ValueError("fixed-R sphere subdivision could not cover a cell")
            spread = np.ptp(points, axis=0)
            axis = int(np.argmax(spread))
            split_order = indices[
                np.argsort(points[:, axis], kind="stable")
            ]
            midpoint = len(split_order) // 2
            if midpoint <= 0 or midpoint >= len(split_order):
                raise AssertionError("invalid deterministic sphere subdivision")
            refine(split_order[:midpoint], depth + 1)
            refine(split_order[midpoint:], depth + 1)

        refine(np.arange(len(bucket_keys), dtype=np.int64))
        candidates = np.asarray(candidate_centers, dtype=float)
        return candidates, np.full(len(candidates), cap, dtype=float)

    @staticmethod
    def _concatenate_proxy_field(
        bucket_fits: list[_BucketFit], field: str
    ) -> np.ndarray:
        return np.concatenate(
            [np.asarray(getattr(item.proxies, field)) for item in bucket_fits],
            axis=0,
        )

    def _fit_incremental_snapshot(
        self,
        ordered_keys: list[tuple[int, ...]],
        changed_keys: set[tuple[int, ...]],
        directional: bool,
    ) -> PointCloudProxySet:
        """Refit only changed fixed buckets, then publish one global snapshot.

        No proxy spans two ``cluster_size`` buckets in the reference fitter.
        Therefore cached fits for unchanged buckets remain valid certificates;
        fitting a changed bucket from all of its current CenterVox cells and
        concatenating the results is conservative and avoids rebuilding the
        complete historical point map every camera frame.
        """

        members: dict[
            tuple[int, int, int], list[tuple[int, ...]]
        ] = {}
        for key in ordered_keys:
            bucket = self._proxy_bucket_key(self._cells[key].representative)
            members.setdefault(bucket, []).append(key)
        changed_buckets = {
            self._proxy_bucket_key(self._cells[key].representative)
            for key in changed_keys
            if key in self._cells
        }
        # A fused representative may cross a fixed proxy-bucket boundary as
        # new frames arrive.  Refit both its new bucket and the old cached
        # bucket that lost the cell.  Removing a now-empty cache entry never
        # removes a CenterVox cell: ``members`` above is rebuilt from the
        # complete persistent cell map on every publication.
        previous_changed_buckets = {
            bucket
            for bucket, item in self._bucket_fits.items()
            if not changed_keys.isdisjoint(item.keys)
        }
        changed_buckets.update(
            bucket for bucket in previous_changed_buckets if bucket in members
        )
        changed_buckets.update(
            bucket for bucket in members if bucket not in self._bucket_fits
        )
        for bucket in set(self._bucket_fits).difference(members):
            del self._bucket_fits[bucket]
        if changed_buckets and self.sphere_cover_mode == "cap_irredundant":
            # Formal sphere candidates and covers are bucket-local, just like
            # the declared ellipsoid family.  No ellipsoid/PCA/Loewner fit is
            # needed to construct balls.  Cached unchanged buckets remain
            # byte-identical and only dirty buckets are regenerated.
            for bucket in sorted(changed_buckets):
                local_keys = list(members[bucket])
                candidate_centers, candidate_radii = (
                    self._build_radius_cap_sphere_candidates(local_keys)
                )
                local = self._publish_adaptive_irredundant_sphere_snapshot(
                    local_keys, candidate_centers, candidate_radii
                )
                self._bucket_fits[bucket] = _BucketFit(
                    tuple(local_keys), local
                )
            changed_buckets = set()
        if changed_buckets:
            batch_buckets = sorted(changed_buckets)
            batch_keys = [
                key for bucket in batch_buckets for key in members[bucket]
            ]
            batch_labels = np.asarray(
                [
                    bucket
                    for bucket in batch_buckets
                    for _key in members[bucket]
                ],
                dtype=np.int64,
            )
            points = np.asarray(
                [self._cells[key].representative for key in batch_keys],
                dtype=float,
            )
            offsets = np.asarray(
                [self._cells[key].cover_radius for key in batch_keys],
                dtype=float,
            )
            uncertainty = (
                np.asarray(
                    [self._cells[key].uncertainty_shape for key in batch_keys],
                    dtype=float,
                )
                if directional
                else None
            )
            batch_fit = fit_matched_voxel_proxies(
                points,
                filter_size=self.filter_size,
                cluster_size=self.cluster_size,
                maximum_aabb_overshoot=self.maximum_aabb_overshoot,
                already_filtered=True,
                filtered_point_offsets=offsets,
                filtered_point_uncertainty_shapes=uncertainty,
                normal_connection_distance=self.normal_connection_distance,
                cluster_origin=self.origin,
                maximum_uncertainty_union_inflation=(
                    self.maximum_uncertainty_union_inflation
                ),
                certificate_radius_limit=self.certificate_radius_limit,
                radius_limit_representation=self.radius_limit_representation,
                uncertainty_fusion_mode=(
                    "separate_uncertainty"
                    if self._uses_partitioned_uncertainty or self.uncertainty_fusion_mode in {"support_interval_uncertainty", "raw_join_uncertainty"}
                    else self.uncertainty_fusion_mode
                ),
                minimum_core_semi_axis=self.minimum_core_semi_axis,
            )
            batch_proxy_indices = np.asarray(
                batch_fit.filtered_cluster_indices, dtype=np.int64
            )
            for bucket in batch_buckets:
                input_indices = np.flatnonzero(
                    np.all(batch_labels == np.asarray(bucket), axis=1)
                )
                selected_proxies = np.unique(
                    batch_proxy_indices[input_indices]
                )
                for proxy_index in selected_proxies:
                    proxy_members = np.flatnonzero(
                        batch_proxy_indices == proxy_index
                    )
                    if not np.all(
                        batch_labels[proxy_members] == np.asarray(bucket)
                    ):
                        raise AssertionError("one fitted proxy crossed fixed buckets")
                remap = {
                    int(proxy_index): local_index
                    for local_index, proxy_index in enumerate(selected_proxies)
                }
                local_indices = np.asarray(
                    [remap[int(value)] for value in batch_proxy_indices[input_indices]],
                    dtype=np.int64,
                )
                local = PointCloudProxySet(
                    raw_points=batch_fit.filtered_points[input_indices].copy(),
                    filtered_points=batch_fit.filtered_points[input_indices].copy(),
                    centers=batch_fit.centers[selected_proxies].copy(),
                    sphere_radii=batch_fit.sphere_radii[selected_proxies].copy(),
                    ellipsoid_shapes=(
                        batch_fit.ellipsoid_shapes[selected_proxies].copy()
                    ),
                    rotations=batch_fit.rotations[selected_proxies].copy(),
                    cluster_keys=batch_fit.cluster_keys[selected_proxies].copy(),
                    filtered_cluster_indices=local_indices,
                    maximum_ellipsoid_overshoot=(
                        batch_fit.maximum_ellipsoid_overshoot
                    ),
                    surface_cover_radius=batch_fit.surface_cover_radius,
                    proxy_offset_radii=(
                        batch_fit.proxy_offset_radii[selected_proxies].copy()
                    ),
                    filtered_uncertainty_shapes=(
                        batch_fit.filtered_uncertainty_shapes[input_indices].copy()
                        if directional
                        else None
                    ),
                    filtered_point_offsets=(
                        batch_fit.filtered_point_offsets[input_indices].copy()
                        if batch_fit.filtered_point_offsets is not None
                        else None
                    ),
                    proxy_uncertainty_shapes=(
                        batch_fit.proxy_uncertainty_shapes[selected_proxies].copy()
                        if directional
                        else None
                    ),
                    base_sphere_radii=(
                        batch_fit.base_sphere_radii[selected_proxies].copy()
                    ),
                    base_ellipsoid_shapes=(
                        batch_fit.base_ellipsoid_shapes[selected_proxies].copy()
                    ),
                    ellipsoid_outer_shapes=(
                        batch_fit.ellipsoid_outer_shapes[selected_proxies].copy()
                    ),
                    maximum_uncertainty_union_inflation=(
                        batch_fit.maximum_uncertainty_union_inflation
                    ),
                    certificate_radius_limit=batch_fit.certificate_radius_limit,
                    uncertainty_fusion_mode=self.uncertainty_fusion_mode,
                )
                if self.ellipsoid_cover_mode == "adaptive_irredundant":
                    if local.filtered_uncertainty_shapes is None:
                        raise AssertionError(
                            "bucket-local ellipsoid cover lost CenterVox U"
                        )
                    # The declared v5 candidate family is fixed-world-bucket
                    # local. Candidates are generated and reduced only over
                    # CenterVox cells in their own cluster_size bucket. Each
                    # retained proxy therefore has a unique witness in that
                    # bucket, which is also a global unique witness.
                    local, _local_cover = reduce_ellipsoid_candidates(
                        local,
                        cell_points=local.filtered_points,
                        cell_uncertainty_shapes=(
                            local.filtered_uncertainty_shapes
                        ),
                        cell_offsets=local.filtered_point_offsets,
                    )
                self._bucket_fits[bucket] = _BucketFit(
                    tuple(members[bucket]), local
                )

        bucket_items = [self._bucket_fits[key] for key in sorted(members)]
        key_to_proxy: dict[tuple[int, int, int], int] = {}
        proxy_base = 0
        global_cluster_keys: list[np.ndarray] = []
        for bucket_key, item in zip(sorted(members), bucket_items):
            if item.keys != tuple(members[bucket_key]):
                raise AssertionError("cached proxy bucket membership is stale")
            local_indices = np.asarray(
                item.proxies.filtered_cluster_indices, dtype=np.int64
            )
            for key, local_index in zip(item.keys, local_indices):
                key_to_proxy[key] = proxy_base + int(local_index)
            local_count = len(item.proxies.centers)
            global_cluster_keys.append(
                np.column_stack(
                    (
                        np.repeat(
                            np.asarray(bucket_key, dtype=np.int64)[None, :],
                            local_count,
                            axis=0,
                        ),
                        np.arange(local_count, dtype=np.int64),
                    )
                )
            )
            proxy_base += local_count

        filtered = np.asarray(
            [self._cells[key].representative for key in ordered_keys], dtype=float
        )
        filtered_uncertainty = (
            np.asarray(
                [self._cells[key].uncertainty_shape for key in ordered_keys],
                dtype=float,
            )
            if directional
            else None
        )
        return PointCloudProxySet(
            raw_points=filtered.copy(),
            filtered_points=filtered,
            centers=self._concatenate_proxy_field(bucket_items, "centers"),
            sphere_radii=self._concatenate_proxy_field(
                bucket_items, "sphere_radii"
            ),
            ellipsoid_shapes=self._concatenate_proxy_field(
                bucket_items, "ellipsoid_shapes"
            ),
            rotations=self._concatenate_proxy_field(bucket_items, "rotations"),
            cluster_keys=np.concatenate(global_cluster_keys, axis=0),
            filtered_cluster_indices=np.asarray(
                [key_to_proxy[key] for key in ordered_keys], dtype=np.int64
            ),
            maximum_ellipsoid_overshoot=max(
                item.proxies.maximum_ellipsoid_overshoot for item in bucket_items
            ),
            surface_cover_radius=max(
                item.proxies.surface_cover_radius for item in bucket_items
            ),
            proxy_offset_radii=self._concatenate_proxy_field(
                bucket_items, "proxy_offset_radii"
            ),
            filtered_uncertainty_shapes=filtered_uncertainty,
            filtered_point_offsets=np.asarray(
                [self._cells[key].cover_radius for key in ordered_keys],
                dtype=float,
            ),
            proxy_uncertainty_shapes=(
                self._concatenate_proxy_field(
                    bucket_items, "proxy_uncertainty_shapes"
                )
                if directional
                else None
            ),
            base_sphere_radii=self._concatenate_proxy_field(
                bucket_items, "base_sphere_radii"
            ),
            base_ellipsoid_shapes=self._concatenate_proxy_field(
                bucket_items, "base_ellipsoid_shapes"
            ),
            ellipsoid_outer_shapes=self._concatenate_proxy_field(
                bucket_items, "ellipsoid_outer_shapes"
            ),
            maximum_uncertainty_union_inflation=max(
                item.proxies.maximum_uncertainty_union_inflation
                for item in bucket_items
            ),
            certificate_radius_limit=self.certificate_radius_limit,
            uncertainty_fusion_mode=self.uncertainty_fusion_mode,
            sphere_cover_mode=self.sphere_cover_mode,
            sphere_cover_candidate_count=(
                sum(
                    item.proxies.sphere_cover_candidate_count
                    for item in bucket_items
                )
                if self.sphere_cover_mode == "cap_irredundant"
                else 0
            ),
            sphere_cover_selected_count=(
                sum(
                    item.proxies.sphere_cover_selected_count
                    for item in bucket_items
                )
                if self.sphere_cover_mode == "cap_irredundant"
                else 0
            ),
            sphere_cover_reverse_deleted=(
                sum(
                    item.proxies.sphere_cover_reverse_deleted
                    for item in bucket_items
                )
                if self.sphere_cover_mode == "cap_irredundant"
                else 0
            ),
            certificate_cover_mode=(
                "bucket_local_adaptive_irredundant_ellipsoid"
                if self.ellipsoid_cover_mode == "adaptive_irredundant"
                else "matched"
            ),
            certificate_cover_candidate_count=(
                sum(
                    item.proxies.certificate_cover_candidate_count
                    for item in bucket_items
                )
                if self.ellipsoid_cover_mode == "adaptive_irredundant"
                else 0
            ),
            certificate_cover_selected_count=(
                sum(
                    item.proxies.certificate_cover_selected_count
                    for item in bucket_items
                )
                if self.ellipsoid_cover_mode == "adaptive_irredundant"
                else 0
            ),
            certificate_cover_reverse_deleted=(
                sum(
                    item.proxies.certificate_cover_reverse_deleted
                    for item in bucket_items
                )
                if self.ellipsoid_cover_mode == "adaptive_irredundant"
                else 0
            ),
            certificate_cover_minimum_unique_witnesses=(
                min(
                    item.proxies.certificate_cover_minimum_unique_witnesses
                    for item in bucket_items
                )
                if self.ellipsoid_cover_mode == "adaptive_irredundant"
                else 0
            ),
        )

    def _persistent_proxy_ids(
        self,
        proxies: PointCloudProxySet,
        ordered_keys: list[tuple[int, int, int]],
    ) -> tuple[np.ndarray, dict[int, frozenset[tuple[int, int, int]]]]:
        if (
            self.uncertainty_fusion_mode == "fused_certified_centervox"
            and self.sphere_cover_mode == "matched"
        ):
            subproxy_count = self._direct_subproxy_count
            expected = (
                np.arange(len(ordered_keys), dtype=np.int64) * subproxy_count
            )
            if not np.array_equal(
                np.asarray(proxies.filtered_cluster_indices, dtype=np.int64),
                expected,
            ):
                raise AssertionError(
                    "direct CenterVox publication lost its subproxy ownership"
                )
            ids = np.empty(len(proxies.centers), dtype=np.int64)
            current_memberships: dict[
                int, frozenset[tuple[int, int, int]]
            ] = {}
            current_subproxy_ids: dict[
                tuple[tuple[int, int, int], int], int
            ] = {}
            for cell_index, key in enumerate(ordered_keys):
                for subindex in range(subproxy_count):
                    stable_key = (key, subindex)
                    proxy_id = self._direct_subproxy_ids.get(stable_key)
                    if proxy_id is None:
                        proxy_id = self._next_proxy_id
                        self._next_proxy_id += 1
                    index = cell_index * subproxy_count + subindex
                    ids[index] = proxy_id
                    current_subproxy_ids[stable_key] = int(proxy_id)
                    current_memberships[int(proxy_id)] = frozenset((key,))
            self._direct_subproxy_ids = current_subproxy_ids
            return ids, current_memberships

        memberships = [
            frozenset(
                ordered_keys[index]
                for index in np.flatnonzero(
                    proxies.filtered_cluster_indices == proxy_index
                )
            )
            for proxy_index in range(len(proxies.centers))
        ]
        assigned_current: set[int] = set()
        assigned_previous: set[int] = set()
        ids = np.full(len(memberships), -1, dtype=np.int64)
        candidates: list[tuple[int, float, int, int]] = []
        # Each CenterVox cell belongs to exactly one previous proxy.  Invert
        # that partition once, then count overlaps by scanning current members.
        # This is exactly equivalent to all current×previous set intersections
        # but O(number of CenterVox cells) instead of O(proxy_count^2).
        previous_owner: dict[tuple[int, int, int], int] = {}
        for previous_id, previous in self._previous_memberships.items():
            for key in previous:
                if key in previous_owner:
                    raise AssertionError("previous proxy memberships overlap")
                previous_owner[key] = previous_id
        for current_index, current in enumerate(memberships):
            overlaps: dict[int, int] = {}
            for key in current:
                previous_id = previous_owner.get(key)
                if previous_id is not None:
                    overlaps[previous_id] = overlaps.get(previous_id, 0) + 1
            for previous_id, intersection in overlaps.items():
                previous = self._previous_memberships[previous_id]
                union = len(current) + len(previous) - intersection
                candidates.append(
                    (intersection, intersection / max(union, 1), previous_id, current_index)
                )
        # A one-to-one maximum-overlap continuation handles unchanged proxies,
        # merges, and splits without ever assigning one old ID twice.
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        for _intersection, _jaccard, previous_id, current_index in candidates:
            if current_index in assigned_current or previous_id in assigned_previous:
                continue
            ids[current_index] = previous_id
            assigned_current.add(current_index)
            assigned_previous.add(previous_id)
        for current_index in range(len(ids)):
            if ids[current_index] >= 0:
                continue
            ids[current_index] = self._next_proxy_id
            self._next_proxy_id += 1
        current_memberships = {
            int(proxy_id): membership
            for proxy_id, membership in zip(ids, memberships)
        }
        return ids, current_memberships

    def _key(self, point: np.ndarray) -> tuple[int, int, int]:
        key = np.floor((point - self.origin) / self.filter_size).astype(np.int64)
        return int(key[0]), int(key[1]), int(key[2])

    def _voxel_center(self, key: tuple[int, int, int]) -> np.ndarray:
        return self.origin + (np.asarray(key, dtype=float) + 0.5) * self.filter_size

    def _sample_key(self, point: np.ndarray) -> tuple[int, int, int]:
        key = np.floor(np.asarray(point) / self._sample_bin).astype(np.int64)
        return int(key[0]), int(key[1]), int(key[2])

    @staticmethod
    def _directional_leaf_geometry_equal(
        first: dict[tuple[int, int, int], _CenterSample],
        second: dict[tuple[int, int, int], _CenterSample],
    ) -> bool:
        """Compare leaf geometry while intentionally ignoring lineage counts."""

        if set(first) != set(second):
            return False
        for key in first:
            a = first[key]
            b = second[key]
            if (
                a.cover_radius != b.cover_radius
                or not np.array_equal(a.point, b.point)
                or not np.array_equal(a.uncertainty_shape, b.uncertainty_shape)
            ):
                return False
        return True

    def _insert_directional_leaf(
        self,
        previous: dict[tuple[int, int, int], _CenterSample],
        new_sample: _CenterSample,
        *,
        representative: np.ndarray,
        source_generation: int,
        precomputed_relations: dict[
            tuple[int, int, int], tuple[bool, bool]
        ] | None = None,
    ) -> tuple[dict[tuple[int, int, int], _CenterSample], bool]:
        """Insert one frame certificate into a bounded-inflation leaf union.

        Every retained leaf is an outer certificate for its lineage. Exact
        Loewner containment removes a redundant leaf. Incomparable leaves are
        joined only when the v5 trace-inflation metric is at most the
        pre-registered 1.05 factor; otherwise both are retained.
        """

        if new_sample.uncertainty_shape is None:
            raise AssertionError("partitioned directional sample lost U")
        representative = np.asarray(representative, dtype=float).reshape(3)
        displacement = np.asarray(new_sample.point, dtype=float) - representative
        new_uncertainty = np.asarray(new_sample.uncertainty_shape, dtype=float)
        translated = (
            0.5 * (new_uncertainty + new_uncertainty.T)
            if not np.any(displacement)
            else minkowski_outer_shape(
                new_uncertainty,
                np.outer(displacement, displacement),
            )
        )
        pending = _CenterSample(
            point=representative.copy(),
            cover_radius=float(new_sample.cover_radius),
            uncertainty_shape=translated,
            source_count=int(new_sample.source_count),
            source_generations=frozenset({int(source_generation)}),
        )
        leaves = dict(previous)
        pending_key: tuple[int, int, int] | None = None
        tolerance = 1.0e-10
        threshold = float(
            1.05
            if self.maximum_uncertainty_union_inflation is None
            else self.maximum_uncertainty_union_inflation
        )

        pending_is_original = True
        for key in sorted(list(leaves)):
            existing = leaves[key]
            if existing.uncertainty_shape is None:
                raise AssertionError("stored directional leaf lost U")
            relation = (
                None
                if precomputed_relations is None or not pending_is_original
                else precomputed_relations.get(key)
            )
            existing_contains_pending = (
                bool(relation[0])
                if relation is not None
                else float(
                    self._batch_containment_values(
                        pending.uncertainty_shape[None, :, :],
                        existing.uncertainty_shape[None, :, :],
                    )[0]
                )
                <= 1.0 + tolerance
            )
            pending_contains_existing = (
                bool(relation[1])
                if relation is not None
                else float(
                    self._batch_containment_values(
                        existing.uncertainty_shape[None, :, :],
                        pending.uncertainty_shape[None, :, :],
                    )[0]
                )
                <= 1.0 + tolerance
            )
            if existing_contains_pending:
                leaves[key] = _CenterSample(
                    point=existing.point,
                    cover_radius=max(existing.cover_radius, pending.cover_radius),
                    uncertainty_shape=existing.uncertainty_shape,
                    source_count=existing.source_count + pending.source_count,
                    source_generations=(
                        existing.source_generations | pending.source_generations
                    ),
                )
                return leaves, not self._directional_leaf_geometry_equal(
                    previous, leaves
                )
            if pending_contains_existing:
                if pending_key is None:
                    pending_key = key
                pending = _CenterSample(
                    point=representative.copy(),
                    cover_radius=max(existing.cover_radius, pending.cover_radius),
                    uncertainty_shape=pending.uncertainty_shape,
                    source_count=existing.source_count + pending.source_count,
                    source_generations=(
                        existing.source_generations | pending.source_generations
                    ),
                )
                del leaves[key]
                continue

            joined = loewner_union_outer_shape(
                np.stack(
                    (existing.uncertainty_shape, pending.uncertainty_shape), axis=0
                )
            )
            maximum_input_trace = max(
                float(np.trace(existing.uncertainty_shape)),
                float(np.trace(pending.uncertainty_shape)),
                1.0e-18,
            )
            trace_inflation = float(np.trace(joined)) / maximum_input_trace
            if trace_inflation <= threshold + tolerance:
                if pending_key is None:
                    pending_key = key
                pending = _CenterSample(
                    point=representative.copy(),
                    cover_radius=max(existing.cover_radius, pending.cover_radius),
                    uncertainty_shape=joined,
                    source_count=existing.source_count + pending.source_count,
                    source_generations=(
                        existing.source_generations | pending.source_generations
                    ),
                )
                del leaves[key]
                pending_is_original = False

        if pending_key is None:
            pending_key = (self._next_directional_leaf_id, 0, 0)
            self._next_directional_leaf_id += 1
        leaves[pending_key] = pending

        if precomputed_relations is not None and pending_is_original:
            # Batched relations proved every retained old leaf incomparable
            # with the unchanged pending certificate. Previous leaves were
            # already an antichain, so the generic reverse pass is redundant.
            return leaves, not self._directional_leaf_geometry_equal(
                previous, leaves
            )

        # Reverse containment deletion makes the stored leaf family
        # inclusion-minimal.  Equal certificates deterministically retain the
        # smaller stable leaf ID and inherit the complete lineage.
        for inner_key in sorted(list(leaves), reverse=True):
            if inner_key not in leaves:
                continue
            inner = leaves[inner_key]
            for outer_key in sorted(leaves):
                if outer_key == inner_key or outer_key not in leaves:
                    continue
                outer = leaves[outer_key]
                contained = float(
                    self._batch_containment_values(
                        inner.uncertainty_shape[None, :, :],
                        outer.uncertainty_shape[None, :, :],
                    )[0]
                ) <= 1.0 + tolerance
                if contained:
                    leaves[outer_key] = _CenterSample(
                        point=outer.point,
                        cover_radius=max(outer.cover_radius, inner.cover_radius),
                        uncertainty_shape=outer.uncertainty_shape,
                        source_count=outer.source_count + inner.source_count,
                        source_generations=(
                            outer.source_generations | inner.source_generations
                        ),
                    )
                    del leaves[inner_key]
                    break

        return leaves, not self._directional_leaf_geometry_equal(previous, leaves)

    def update(
        self,
        points: np.ndarray,
        point_radii: np.ndarray,
        point_uncertainty_shapes: np.ndarray | None = None,
    ) -> ProxyUpdateStats:
        started = time.perf_counter()
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        point_radii = np.asarray(point_radii, dtype=float).reshape(-1)
        if len(points) != len(point_radii):
            raise ValueError("point_radii must contain one radius per point")
        directional = point_uncertainty_shapes is not None
        if self._directional is None:
            self._directional = directional
        elif self._directional != directional:
            raise ValueError("one manager cannot mix scalar and directional updates")
        if directional:
            point_uncertainty_shapes = np.asarray(
                point_uncertainty_shapes, dtype=float
            ).reshape(-1, 3, 3)
            if len(point_uncertainty_shapes) != len(points):
                raise ValueError(
                    "point_uncertainty_shapes must contain one shape per point"
                )
        spatial_cells = (
            self._spatial_cells
            if self._uses_partitioned_uncertainty
            else self._cells
        )
        # Frame-local CenterVox is computed in batches. Every raw point center
        # is covered by the scalar residual around the selected representative;
        # every directional U is covered by one grouped PSD join. Only that
        # proved aggregate is retained, so state grows with occupied voxels,
        # not camera pixels x frames.
        integer_keys = np.floor(
            (points - self.origin[None, :]) / self.filter_size
        ).astype(np.int64)
        unique_keys, inverse = np.unique(
            integer_keys, axis=0, return_inverse=True
        )
        group_count = len(unique_keys)
        voxel_centers = self.origin[None, :] + (
            unique_keys.astype(float) + 0.5
        ) * self.filter_size
        distances = np.linalg.norm(points - voxel_centers[inverse], axis=1)
        point_order = np.lexsort(
            (np.arange(len(points), dtype=np.int64), distances, inverse)
        )
        first = np.r_[
            0,
            np.flatnonzero(
                np.diff(inverse[point_order])
            ) + 1,
        ]
        representative_indices = point_order[first]
        frame_representatives = points[representative_indices].copy()
        key_tuples = [tuple(map(int, key)) for key in unique_keys]
        # Online CenterVox uses a stable representative: the first frame picks
        # the observed point nearest the fixed world-cell center; later frames
        # measure their raw centers directly from that same point. This keeps
        # persistent proxy IDs and makes delta <= sqrt(3)*filter_size instead
        # of recursively accumulating recentering inequalities.
        if not self._uses_fused_uncertainty:
            for group_index, key in enumerate(key_tuples):
                old = spatial_cells.get(key)
                if old is not None:
                    frame_representatives[group_index] = old.representative
        displacements = points - frame_representatives[inverse]
        residual_required = np.linalg.norm(displacements, axis=1)
        if not directional:
            residual_required += point_radii
        frame_cover = np.zeros(group_count, dtype=float)
        np.maximum.at(frame_cover, inverse, residual_required)
        group_sizes = np.bincount(inverse, minlength=group_count)
        frame_uncertainty = None
        interval_outer = interval_counts = None
        if directional and self.uncertainty_fusion_mode in {"support_interval_uncertainty", "raw_join_uncertainty"}:
            interval_outer, interval_counts = self.raw_support_intervals.update(
                points, point_uncertainty_shapes, key_tuples, inverse,
                frame_representatives, point_order, first)
        raw_directional_leaves = None
        if directional and self.uncertainty_fusion_mode == "raw_directional_uncertainty":
            raw_directional_leaves = self.raw_directional_join.update(
                points, point_uncertainty_shapes, key_tuples, inverse, frame_representatives)
        if directional:
            if raw_directional_leaves is not None:
                frame_uncertainty = np.asarray([next(iter(x.values()))['shape'] for x in raw_directional_leaves])
            elif self.uncertainty_fusion_mode in {"support_interval_uncertainty", "raw_join_uncertainty"}:
                frame_uncertainty = interval_outer
            elif self._uses_fused_uncertainty:
                if self.uncertainty_fusion_mode == "fused_certified_centervox":
                    # The direct mode below accumulates every raw measurement
                    # support interval in its persistent cell frame.  A
                    # frame-local outer ellipsoid would be discarded and only
                    # add a second sqrt-budget inflation, so retain harmless
                    # placeholders until that exact support-box update.
                    frame_uncertainty = point_uncertainty_shapes[
                        representative_indices
                    ].copy()
                else:
                    frame_uncertainty = np.empty(
                        (group_count, 3, 3), dtype=float
                    )
                    ends = np.r_[first[1:], len(point_order)]
                    for group_index, (start, stop) in enumerate(zip(first, ends)):
                        member_indices = point_order[start:stop]
                        center, shape, _rotation, _half = (
                            certified_translated_ellipsoid_outer(
                                points[member_indices],
                                point_uncertainty_shapes[member_indices],
                            )
                        )
                        frame_representatives[group_index] = center
                        frame_uncertainty[group_index] = shape
            else:
                # Legacy Q+U path: symmetrize center relocation before a
                # grouped Loewner join.
                relocation_shapes = np.einsum(
                    "ni,nj->nij", displacements, displacements
                )
                translated_uncertainty = minkowski_outer_shapes(
                    point_uncertainty_shapes, relocation_shapes
                )
                frame_uncertainty = (
                    loewner_union_outer_shapes_grouped_common_frame(
                        translated_uncertainty, inverse, group_count
                    )
                )
                containment = self._batch_containment_values(
                    translated_uncertainty, frame_uncertainty[inverse]
                )
                if float(np.max(containment)) > 1.0 + 1.0e-9:
                    raise AssertionError(
                        "frame CenterVox U failed to cover translated input U"
                    )
            # Relocation has been certified directionally in U.
            frame_cover.fill(0.0)
        frame_centervox_done = time.perf_counter()

        frame_generation = self._generation + 1
        samples_by_group: list[list[_CenterSample]] = []
        sample_points: list[np.ndarray] = []
        sample_covers: list[float] = []
        sample_uncertainties: list[np.ndarray] = []
        sample_owners: list[int] = []
        partitioned_sample_dicts: list[
            dict[tuple[int, int, int], _CenterSample]
        ] = []
        partitioned_geometry_changed = np.zeros(group_count, dtype=bool)
        partitioned_relations: list[
            dict[tuple[int, int, int], tuple[bool, bool]] | None
        ] = [None] * group_count
        if self._uses_partitioned_uncertainty and raw_directional_leaves is None:
            pair_groups: list[int] = []
            pair_keys: list[tuple[int, int, int]] = []
            existing_shape_rows: list[np.ndarray] = []
            new_shape_rows: list[np.ndarray] = []
            for group_index, key in enumerate(key_tuples):
                previous = spatial_cells.get(key)
                if previous is None or not np.array_equal(
                    previous.representative,
                    frame_representatives[group_index],
                ):
                    continue
                partitioned_relations[group_index] = {}
                for sample_key, sample in sorted(previous.samples.items()):
                    pair_groups.append(group_index)
                    pair_keys.append(sample_key)
                    existing_shape_rows.append(sample.uncertainty_shape)
                    new_shape_rows.append(frame_uncertainty[group_index])
            if pair_groups:
                existing_shapes = np.asarray(existing_shape_rows, dtype=float)
                new_shapes = np.asarray(new_shape_rows, dtype=float)
                existing_contains = self._batch_containment_values(
                    new_shapes, existing_shapes
                ) <= 1.0 + 1.0e-10
                new_contains = self._batch_containment_values(
                    existing_shapes, new_shapes
                ) <= 1.0 + 1.0e-10
                for relation_index, (group_index, sample_key) in enumerate(
                    zip(pair_groups, pair_keys)
                ):
                    partitioned_relations[group_index][sample_key] = (
                        bool(existing_contains[relation_index]),
                        bool(new_contains[relation_index]),
                    )
        for group_index, key in enumerate(key_tuples):
            new_sample = _CenterSample(
                point=frame_representatives[group_index].copy(),
                cover_radius=float(frame_cover[group_index]),
                uncertainty_shape=(
                    None
                    if not directional
                    else frame_uncertainty[group_index].copy()
                ),
                source_count=int(group_sizes[group_index]),
                source_generations=frozenset({int(frame_generation)}),
            )
            if self._uses_partitioned_uncertainty:
                previous = spatial_cells.get(key)
                previous_samples = {} if previous is None else previous.samples
                if raw_directional_leaves is not None:
                    sample_dict = {(code,0,0): _CenterSample(
                        point=leaf['center'].copy(), cover_radius=0.0,
                        uncertainty_shape=leaf['shape'].copy(), source_count=leaf['count'],
                        source_generations=leaf['generations'])
                        for code,leaf in raw_directional_leaves[group_index].items()}
                    geometry_changed = not self._directional_leaf_geometry_equal(previous_samples, sample_dict)
                else:
                    sample_dict, geometry_changed = self._insert_directional_leaf(
                        previous_samples, new_sample,
                        representative=frame_representatives[group_index],
                        source_generation=frame_generation,
                        precomputed_relations=partitioned_relations[group_index])
                samples = list(sample_dict.values())
                partitioned_sample_dicts.append(sample_dict)
                partitioned_geometry_changed[group_index] = geometry_changed
            else:
                samples = (
                    []
                    if (
                        self.uncertainty_fusion_mode
                        in {"fused_certified_centervox", "support_interval_uncertainty", "raw_join_uncertainty"}
                        or key not in spatial_cells
                    )
                    else list(spatial_cells[key].samples.values())
                )
                samples.append(new_sample)
            samples_by_group.append(samples)
            for sample in samples:
                sample_points.append(sample.point)
                sample_covers.append(float(sample.cover_radius))
                sample_owners.append(group_index)
                if directional:
                    sample_uncertainties.append(sample.uncertainty_shape)

        owners = np.asarray(sample_owners, dtype=np.int64)
        aggregate_points = np.asarray(sample_points, dtype=float)
        aggregate_covers = np.asarray(sample_covers, dtype=float)
        aggregate_distances = np.linalg.norm(
            aggregate_points - voxel_centers[owners], axis=1
        )
        aggregate_order = np.lexsort(
            (
                np.arange(len(aggregate_points), dtype=np.int64),
                aggregate_distances,
                owners,
            )
        )
        aggregate_first = np.r_[
            0,
            np.flatnonzero(np.diff(owners[aggregate_order])) + 1,
        ]
        selected = aggregate_order[aggregate_first]
        merged_representatives = aggregate_points[selected]
        aggregate_displacements = aggregate_points - merged_representatives[owners]
        merged_required = np.linalg.norm(
            aggregate_displacements, axis=1
        ) + aggregate_covers
        merged_cover = np.zeros(group_count, dtype=float)
        np.maximum.at(merged_cover, owners, merged_required)
        merged_uncertainty = None
        fusion_rotations = None
        fusion_lowers = None
        fusion_uppers = None
        fusion_narrow_axes = None
        if directional:
            if self.uncertainty_fusion_mode in {"support_interval_uncertainty", "raw_join_uncertainty"}:
                merged_representatives = frame_representatives.copy()
                merged_uncertainty = interval_outer
                for i in range(group_count):
                    samples_by_group[i] = [_CenterSample(
                        point=merged_representatives[i].copy(), cover_radius=0.0,
                        uncertainty_shape=interval_outer[i].copy(),
                        source_count=int(interval_counts[i]))]
            elif self.uncertainty_fusion_mode == "fused_certified_centervox":
                # Keep one data-derived frame per persistent CenterVox and
                # accumulate exact coordinate support intervals from raw
                # measurement ellipsoids.  This flattens all camera frames
                # into one certified box before a single ellipsoid outer is
                # formed, avoiding frame->cell sqrt factors.
                merged_representatives = np.empty((group_count, 3), dtype=float)
                merged_uncertainty = np.empty((group_count, 3, 3), dtype=float)
                fusion_rotations = np.empty((group_count, 3, 3), dtype=float)
                fusion_lowers = np.empty((group_count, 3), dtype=float)
                fusion_uppers = np.empty((group_count, 3), dtype=float)
                fusion_narrow_axes = np.empty(group_count, dtype=np.int64)
                ends = np.r_[first[1:], len(point_order)]
                previous_lowers = np.full((group_count, 3), np.inf, dtype=float)
                previous_uppers = np.full((group_count, 3), -np.inf, dtype=float)
                previous_mask = np.zeros(group_count, dtype=bool)
                previous_source_counts = np.zeros(group_count, dtype=np.int64)
                for group_index, (start, stop) in enumerate(zip(first, ends)):
                    member_indices = point_order[start:stop]
                    key = key_tuples[group_index]
                    previous = spatial_cells.get(key)
                    if previous is not None and previous.fusion_rotation is not None:
                        rotation = previous.fusion_rotation
                        previous_lowers[group_index] = previous.fusion_lower
                        previous_uppers[group_index] = previous.fusion_upper
                        previous_mask[group_index] = True
                        fusion_narrow_axes[group_index] = int(
                            previous.fusion_narrow_axis
                        )
                        previous_source_counts[group_index] = sum(
                            int(item.source_count)
                            for item in previous.samples.values()
                        )
                    else:
                        _center, _shape, rotation, _half = (
                            certified_translated_ellipsoid_outer(
                                points[member_indices],
                                point_uncertainty_shapes[member_indices],
                            )
                        )
                    fusion_rotations[group_index] = rotation

                point_rotations = fusion_rotations[inverse]
                coordinates = np.einsum(
                    "ni,nik->nk", points, point_rotations
                )
                support = np.sqrt(
                    np.maximum(
                        np.einsum(
                            "nik,nij,njk->nk",
                            point_rotations,
                            point_uncertainty_shapes,
                            point_rotations,
                        ),
                        0.0,
                    )
                )
                # point_order is already grouped by the CenterVox inverse.
                # Grouped reduceat is algebraically identical to minimum.at /
                # maximum.at here, while avoiding random atomic-style writes
                # into the output for every depth pixel.
                ordered_lower = (coordinates - support)[point_order]
                ordered_upper = (coordinates + support)[point_order]
                fusion_lowers = np.minimum.reduceat(
                    ordered_lower, first, axis=0
                )
                fusion_uppers = np.maximum.reduceat(
                    ordered_upper, first, axis=0
                )
                fusion_lowers[previous_mask] = np.minimum(
                    fusion_lowers[previous_mask],
                    previous_lowers[previous_mask],
                )
                fusion_uppers[previous_mask] = np.maximum(
                    fusion_uppers[previous_mask],
                    previous_uppers[previous_mask],
                )
                local_centers = 0.5 * (fusion_lowers + fusion_uppers)
                half = np.maximum(
                    0.5 * (fusion_uppers - fusion_lowers), 1.0e-7
                )
                new_mask = ~previous_mask
                fusion_narrow_axes[new_mask] = np.argmin(
                    half[new_mask], axis=1
                )
                # This cell-level shape is retained for observability logging.
                # Use the same frozen thin-axis budget as the published
                # subproxy cover; the published controller geometry is rebuilt
                # directly from the exact support intervals below.
                narrow_weight = 1.0 / self.direct_thin_axis_inflation**2
                tangent_weight = 0.5 * (1.0 - narrow_weight)
                weights = np.full(
                    (group_count, 3), tangent_weight, dtype=float
                )
                weights[
                    np.arange(group_count), fusion_narrow_axes
                ] = narrow_weight
                axes_squared = np.square(half) / weights
                merged_representatives = np.einsum(
                    "nij,nj->ni", fusion_rotations, local_centers
                )
                merged_uncertainty = np.einsum(
                    "nik,nk,njk->nij",
                    fusion_rotations,
                    axes_squared,
                    fusion_rotations,
                )
                merged_uncertainty = 0.5 * (
                    merged_uncertainty
                    + np.swapaxes(merged_uncertainty, 1, 2)
                )
                source_counts = group_sizes + previous_source_counts
                for group_index in range(group_count):
                    center = merged_representatives[group_index]
                    shape = merged_uncertainty[group_index]
                    samples_by_group[group_index] = [
                        _CenterSample(
                            point=center.copy(),
                            cover_radius=0.0,
                            uncertainty_shape=shape.copy(),
                            source_count=int(source_counts[group_index]),
                        )
                    ]
            elif self._uses_fused_uncertainty:
                merged_representatives = np.empty((group_count, 3), dtype=float)
                merged_uncertainty = np.empty((group_count, 3, 3), dtype=float)
                for group_index, samples in enumerate(samples_by_group):
                    center, shape, _rotation, _half = (
                        certified_translated_ellipsoid_outer(
                            np.asarray([sample.point for sample in samples]),
                            np.asarray(
                                [sample.uncertainty_shape for sample in samples]
                            ),
                        )
                    )
                    merged_representatives[group_index] = center
                    merged_uncertainty[group_index] = shape
            else:
                relocation_shapes = np.einsum(
                    "ni,nj->nij",
                    aggregate_displacements,
                    aggregate_displacements,
                )
                translated_uncertainty = minkowski_outer_shapes(
                    np.asarray(sample_uncertainties, dtype=float),
                    relocation_shapes,
                )
                merged_uncertainty = (
                    loewner_union_outer_shapes_grouped_common_frame(
                        translated_uncertainty,
                        owners,
                        group_count,
                    )
                )
            # All inter-frame recentering is now carried by the directional
            # ellipsoid above, not an isotropic scalar offset.
            merged_cover.fill(0.0)
        history_union_done = time.perf_counter()

        changed: set[tuple[int, ...]] = set()
        contained_cell_reuses = 0
        # The original implementation performed one pair of 3x3
        # eigendecompositions for every reobserved cell inside the Python
        # loop below.  Evaluate the identical Loewner-containment predicate in
        # one batched call.  This changes neither the selected representative
        # nor the certificate: it only removes thousands of tiny NumPy calls
        # from each online frame.
        contained_by_previous = np.zeros(group_count, dtype=bool)
        previous_cells = [spatial_cells.get(key) for key in key_tuples]
        previous_exists = np.asarray(
            [cell is not None for cell in previous_cells], dtype=bool
        )
        if self._uses_partitioned_uncertainty:
            contained_by_previous = previous_exists & ~partitioned_geometry_changed
        elif np.any(previous_exists):
            previous_indices = np.flatnonzero(previous_exists)
            previous_representatives = np.asarray(
                [previous_cells[index].representative for index in previous_indices],
                dtype=float,
            )
            previous_covers = np.asarray(
                [previous_cells[index].cover_radius for index in previous_indices],
                dtype=float,
            )
            eligible = np.all(
                previous_representatives
                == merged_representatives[previous_indices], axis=1
            ) & (
                previous_covers + 1.0e-12
                >= merged_cover[previous_indices]
            )
            if directional and np.any(eligible):
                eligible_positions = np.flatnonzero(eligible)
                eligible_indices = previous_indices[eligible_positions]
                previous_shapes = np.asarray(
                    [
                        previous_cells[index].uncertainty_shape
                        for index in eligible_indices
                    ],
                    dtype=float,
                )
                shape_available = np.asarray(
                    [
                        previous_cells[index].uncertainty_shape is not None
                        for index in eligible_indices
                    ],
                    dtype=bool,
                )
                values = np.full(len(eligible_indices), np.inf, dtype=float)
                if np.any(shape_available):
                    values[shape_available] = self._batch_containment_values(
                        merged_uncertainty[eligible_indices[shape_available]],
                        previous_shapes[shape_available],
                    )
                eligible[eligible_positions] &= values <= 1.0 + 1.0e-10
            contained_by_previous[previous_indices[eligible]] = True
        for group_index, key in enumerate(key_tuples):
            previous = previous_cells[group_index]
            samples = samples_by_group[group_index]
            # One aggregate per source frame preserves exact re-centering and
            # avoids recursive outer-bound drift.
            sample_dict = (
                partitioned_sample_dicts[group_index]
                if self._uses_partitioned_uncertainty
                else {
                    (index, 0, 0): sample
                    for index, sample in enumerate(samples)
                }
            )
            uncertainty_shape = (
                None
                if not directional
                else merged_uncertainty[group_index].copy()
            )
            representative = merged_representatives[group_index].copy()
            current = _CenterVoxel(
                representative=representative,
                representative_distance=float(
                    np.linalg.norm(representative - voxel_centers[group_index])
                ),
                cover_radius=float(merged_cover[group_index]),
                uncertainty_shape=uncertainty_shape,
                samples=sample_dict,
                fusion_rotation=(
                    None
                    if fusion_rotations is None
                    else fusion_rotations[group_index].copy()
                ),
                fusion_lower=(
                    None
                    if fusion_lowers is None
                    else fusion_lowers[group_index].copy()
                ),
                fusion_upper=(
                    None
                    if fusion_uppers is None
                    else fusion_uppers[group_index].copy()
                ),
                fusion_narrow_axis=(
                    None
                    if fusion_narrow_axes is None
                    else int(fusion_narrow_axes[group_index])
                ),
            )
            # Reobserving a cell must not dirty its proxy bucket when the
            # already-published CenterVox certificate contains the newly
            # fused certificate.  Preserve the previous object byte-for-byte:
            # this is a conservative containment decision, not a tolerance-
            # based deletion of evidence.  It prevents harmless floating
            # changes from triggering global cover reconstruction.
            if (
                contained_by_previous[group_index]
                and not self._uses_partitioned_uncertainty
            ):
                if self.uncertainty_fusion_mode in {"support_interval_uncertainty", "raw_join_uncertainty"}:
                    current = replace(previous, samples={(0,0,0): _CenterSample(
                        point=previous.representative.copy(), cover_radius=0.0,
                        uncertainty_shape=previous.uncertainty_shape.copy(),
                        source_count=int(interval_counts[group_index]))})
                else:
                    current = previous
                contained_cell_reuses += 1
            elif contained_by_previous[group_index]:
                # Geometry is byte-identical but the new source generation is
                # retained in the leaf lineage and source count.
                contained_cell_reuses += 1
            spatial_cells[key] = current
            if not contained_by_previous[group_index]:
                changed.add(key)

        if not spatial_cells:
            raise ValueError("cannot publish proxies before receiving any points")
        if self._uses_partitioned_uncertainty:
            previous_leaf_cells = self._cells
            leaf_cells: dict[tuple[int, ...], _CenterVoxel] = {}
            for spatial_key in sorted(spatial_cells):
                spatial = spatial_cells[spatial_key]
                for sample_key, sample in sorted(spatial.samples.items()):
                    leaf_key = (*spatial_key, int(sample_key[0]))
                    leaf_cells[leaf_key] = _CenterVoxel(
                        representative=np.asarray(sample.point, dtype=float).copy(),
                        representative_distance=spatial.representative_distance,
                        cover_radius=float(sample.cover_radius),
                        uncertainty_shape=np.asarray(
                            sample.uncertainty_shape, dtype=float
                        ).copy(),
                        samples={sample_key: sample},
                    )
            changed_spatial = {tuple(key[:3]) for key in changed}
            changed = {
                key
                for key in set(previous_leaf_cells) | set(leaf_cells)
                if tuple(key[:3]) in changed_spatial
                or key not in previous_leaf_cells
                or key not in leaf_cells
            }
            self._spatial_cells = spatial_cells
            self._cells = leaf_cells

        if not self._cells:
            raise ValueError("directional partition produced no certificates")
        ordered_keys = sorted(self._cells)
        representatives: list[np.ndarray] = []
        cover_radii: list[float] = []
        uncertainty_shapes: list[np.ndarray] = []
        for key in ordered_keys:
            cell = self._cells[key]
            representative = cell.representative
            representatives.append(representative)
            cover_radii.append(cell.cover_radius)
            if directional:
                if cell.uncertainty_shape is None:
                    raise AssertionError("directional CenterVox cell lost its shape")
                uncertainty_shapes.append(cell.uncertainty_shape)
        filtered = np.asarray(representatives)
        offsets = np.asarray(cover_radii)
        cell_merge_done = time.perf_counter()
        if (
            not changed
            and self._snapshot is not None
            and tuple(ordered_keys) == self._last_ordered_keys
        ):
            # Every new observation was proved contained above, so the exact
            # published proxy certificate remains valid and inclusion-minimal
            # relative to the unchanged candidate family.
            proxies = self._snapshot
        elif self.sphere_cover_mode in {
            "adaptive_irredundant",
            "cap_irredundant",
        }:
            candidates = self._fit_incremental_snapshot(
                ordered_keys, changed, directional
            )
            if self.sphere_cover_mode == "cap_irredundant":
                # The fixed-world bucket routine already published and
                # concatenated independently irreducible local ball covers.
                proxies = candidates
            elif self.sphere_cover_mode == "adaptive_irredundant":
                candidate_centers, candidate_radii = (
                    self._build_resolution_bounded_sphere_candidates(
                        ordered_keys, candidates
                    )
                )
                proxies = self._publish_adaptive_irredundant_sphere_snapshot(
                    ordered_keys, candidate_centers, candidate_radii
                )
        elif self.uncertainty_fusion_mode == "fused_certified_centervox":
            proxies = self._publish_fused_centervox_snapshot(ordered_keys)
        else:
            proxies = self._fit_incremental_snapshot(
                ordered_keys, changed, directional
            )
        bucket_fit_done = time.perf_counter()
        proxy_ids, memberships = self._persistent_proxy_ids(proxies, ordered_keys)
        proxies = replace(proxies, proxy_ids=proxy_ids)
        digest = hashlib.sha256()
        for array in (
            np.column_stack(
                [
                    proxies.centers,
                    proxies.sphere_radii,
                    proxies.proxy_offset_radii,
                ]
            ),
            proxies.ellipsoid_shapes,
            proxies.proxy_uncertainty_shapes,
            proxies.ellipsoid_outer_shapes,
            proxy_ids,
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(memoryview(contiguous).cast("B"))
        self._snapshot_hash = digest.hexdigest()
        self._snapshot = proxies
        self._previous_memberships = memberships
        self._last_ordered_keys = tuple(ordered_keys)
        self._generation += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ProxyUpdateStats(
            generation=self._generation,
            inserted_points=len(points),
            center_voxels=len(filtered),
            proxy_count=len(proxies.centers),
            changed_center_voxels=len(changed),
            update_ms=float(elapsed_ms),
            snapshot_sha256=self._snapshot_hash,
            frame_centervox_ms=float(
                (frame_centervox_done - started) * 1000.0
            ),
            history_union_ms=float(
                (history_union_done - frame_centervox_done) * 1000.0
            ),
            cell_merge_ms=float(
                (cell_merge_done - history_union_done) * 1000.0
            ),
            bucket_fit_ms=float(
                (bucket_fit_done - cell_merge_done) * 1000.0
            ),
            publish_ids_hash_ms=float(
                (time.perf_counter() - bucket_fit_done) * 1000.0
            ),
            sphere_cover_candidate_count=(
                proxies.sphere_cover_candidate_count
            ),
            sphere_cover_selected_count=(
                proxies.sphere_cover_selected_count
            ),
            sphere_cover_reverse_deleted=(
                proxies.sphere_cover_reverse_deleted
            ),
            certificate_cover_candidate_count=(
                proxies.certificate_cover_candidate_count
            ),
            certificate_cover_selected_count=(
                proxies.certificate_cover_selected_count
            ),
            certificate_cover_reverse_deleted=(
                proxies.certificate_cover_reverse_deleted
            ),
            certificate_cover_minimum_unique_witnesses=(
                proxies.certificate_cover_minimum_unique_witnesses
            ),
            contained_cell_reuses=contained_cell_reuses,
        )

    @property
    def snapshot(self) -> PointCloudProxySet:
        if self._snapshot is None:
            raise RuntimeError("no proxy snapshot has been published")
        return self._snapshot

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def snapshot_hash(self) -> str:
        return self._snapshot_hash

    @staticmethod
    def _batch_containment_values(
        inner: np.ndarray, outer: np.ndarray
    ) -> np.ndarray:
        """Return batched lambda_max(outer^-1/2 inner outer^-1/2)."""

        inner = np.asarray(inner, dtype=float).reshape(-1, 3, 3)
        outer = np.asarray(outer, dtype=float).reshape(-1, 3, 3)
        if len(inner) != len(outer):
            raise ValueError("inner and outer batches must have equal length")
        symmetric_outer = 0.5 * (outer + np.swapaxes(outer, 1, 2))
        values, vectors = np.linalg.eigh(symmetric_outer)
        inverse_sqrt = 1.0 / np.sqrt(np.maximum(values, 1.0e-18))
        inverse_root = np.einsum(
            "nij,nj,nkj->nik", vectors, inverse_sqrt, vectors
        )
        symmetric_inner = 0.5 * (inner + np.swapaxes(inner, 1, 2))
        transformed = np.einsum(
            "nij,njk,nkl->nil", inverse_root, symmetric_inner, inverse_root
        )
        transformed = 0.5 * (
            transformed + np.swapaxes(transformed, 1, 2)
        )
        return np.linalg.eigvalsh(transformed)[:, -1]

    def _directional_coverage_audit(
        self, tolerance: float
    ) -> ProxyCoverageAudit:
        """Vectorized audit of the complete directional certificate chain."""

        proxies = self.snapshot
        keys = self._last_ordered_keys
        representatives = np.asarray(
            [self._cells[key].representative for key in keys], dtype=float
        )
        proxy_indices = np.asarray(
            proxies.filtered_cluster_indices, dtype=np.int64
        )
        cell_shapes = np.asarray(
            [self._cells[key].uncertainty_shape for key in keys], dtype=float
        )

        sample_points: list[np.ndarray] = []
        sample_shapes: list[np.ndarray] = []
        sample_owners: list[int] = []
        raw_source_count = 0
        for cell_index, key in enumerate(keys):
            for sample in self._cells[key].samples.values():
                if sample.uncertainty_shape is None:
                    raise AssertionError("directional sample lost U")
                sample_points.append(sample.point)
                sample_shapes.append(sample.uncertainty_shape)
                sample_owners.append(cell_index)
                raw_source_count += int(sample.source_count)
        owners = np.asarray(sample_owners, dtype=np.int64)
        displacements = (
            np.asarray(sample_points, dtype=float) - representatives[owners]
        )
        candidates = (
            np.asarray(sample_shapes, dtype=float)
            if self._uses_partitioned_uncertainty or self.uncertainty_fusion_mode in {"support_interval_uncertainty", "raw_join_uncertainty"}
            else minkowski_outer_shapes(
                np.asarray(sample_shapes, dtype=float),
                np.einsum("ni,nj->nij", displacements, displacements),
            )
        )
        center_values = self._batch_containment_values(
            candidates, cell_shapes[owners]
        )
        proxy_uncertainty = np.asarray(
            proxies.proxy_uncertainty_shapes, dtype=float
        )[proxy_indices]
        offset_values = self._batch_containment_values(
            cell_shapes, proxy_uncertainty
        )

        deltas = representatives - proxies.centers[proxy_indices]
        base_shapes = np.asarray(
            proxies.base_ellipsoid_shapes, dtype=float
        )[proxy_indices]
        solved = np.linalg.solve(base_shapes, deltas[:, :, None])[:, :, 0]
        ellipsoid_values = np.einsum("ni,ni->n", deltas, solved)
        base_radii = np.asarray(proxies.base_sphere_radii, dtype=float)[
            proxy_indices
        ]
        published_effective_radii = (
            np.asarray(proxies.sphere_radii, dtype=float)
            + np.asarray(proxies.proxy_offset_radii, dtype=float)
        )[proxy_indices]
        cell_uncertainty_radii = np.sqrt(
            np.maximum(np.linalg.eigvalsh(cell_shapes)[:, -1], 0.0)
        )
        sphere_slacks = (
            published_effective_radii
            - np.linalg.norm(deltas, axis=1)
            - cell_uncertainty_radii
        )

        cell_residual = np.asarray(
            [self._cells[key].cover_radius for key in keys], dtype=float
        )
        minimum_center_slack = float(np.min(1.0 - center_values))
        proxy_offsets = np.asarray(proxies.proxy_offset_radii, dtype=float)
        minimum_offset_slack = float(
            np.min(proxy_offsets[proxy_indices] - cell_residual)
        )
        minimum_measurement_uncertainty_slack = float(
            np.min(1.0 - center_values)
        )
        minimum_proxy_uncertainty_slack = float(np.min(1.0 - offset_values))
        maximum_ellipsoid_value = float(np.max(ellipsoid_values))
        minimum_sphere_slack = float(np.min(sphere_slacks))
        sphere_certificate_required = not (
            self.ellipsoid_cover_mode == "adaptive_irredundant"
            and self.radius_limit_representation == "ellipsoid"
        )
        certified = bool(
            minimum_center_slack >= -tolerance
            and minimum_offset_slack >= -tolerance
            and minimum_measurement_uncertainty_slack >= -tolerance
            and minimum_proxy_uncertainty_slack >= -tolerance
            and maximum_ellipsoid_value <= 1.0 + tolerance
            and (
                not sphere_certificate_required
                or minimum_sphere_slack >= -tolerance
            )
        )
        cover_inclusion_minimal = bool(
            self.ellipsoid_cover_mode != "adaptive_irredundant"
            or (
                proxies.certificate_cover_selected_count
                == len(proxies.centers)
                and proxies.certificate_cover_minimum_unique_witnesses > 0
            )
        )
        certified = bool(certified and cover_inclusion_minimal)
        return ProxyCoverageAudit(
            raw_sample_bins=raw_source_count,
            center_voxels=len(keys),
            proxy_count=len(proxies.centers),
            minimum_centervox_cover_slack=minimum_center_slack,
            minimum_proxy_offset_slack=minimum_offset_slack,
            maximum_base_ellipsoid_value=maximum_ellipsoid_value,
            minimum_sphere_certificate_slack=minimum_sphere_slack,
            all_raw_sample_balls_certified=certified,
            minimum_measurement_uncertainty_slack=(
                minimum_measurement_uncertainty_slack
            ),
            minimum_proxy_uncertainty_slack=minimum_proxy_uncertainty_slack,
            certificate_cover_inclusion_minimal=cover_inclusion_minimal,
            certificate_cover_candidate_count=(
                proxies.certificate_cover_candidate_count
            ),
            certificate_cover_selected_count=(
                proxies.certificate_cover_selected_count
            ),
            certificate_cover_reverse_deleted=(
                proxies.certificate_cover_reverse_deleted
            ),
            certificate_cover_minimum_unique_witnesses=(
                proxies.certificate_cover_minimum_unique_witnesses
            ),
        )

    def _adaptive_sphere_coverage_audit(
        self, tolerance: float
    ) -> ProxyCoverageAudit:
        """Independently audit coverage and inclusion minimality of balls."""

        proxies = self.snapshot
        keys = self._last_ordered_keys
        cells = [self._cells[key] for key in keys]
        if any(cell.uncertainty_shape is None for cell in cells):
            raise AssertionError("sphere audit lost CenterVox U")
        cell_centers = np.asarray(
            [cell.representative for cell in cells], dtype=float
        )
        cell_shapes = np.asarray(
            [cell.uncertainty_shape for cell in cells], dtype=float
        )
        cell_support_radii = np.sqrt(
            np.maximum(np.linalg.eigvalsh(cell_shapes)[:, -1], 0.0)
        ) + np.asarray([cell.cover_radius for cell in cells], dtype=float)
        owner = np.asarray(proxies.filtered_cluster_indices, dtype=np.int64)
        if len(owner) != len(cells) or np.any(owner < 0) or np.any(
            owner >= len(proxies.centers)
        ):
            raise AssertionError("sphere audit found an invalid cell owner")

        assigned_centers = np.asarray(proxies.centers, dtype=float)[owner]
        required = (
            np.linalg.norm(assigned_centers - cell_centers, axis=1)
            + cell_support_radii
        )
        published = np.asarray(proxies.sphere_radii, dtype=float)[owner]
        slacks = published - required
        minimum_slack = float(np.min(slacks))
        ratios = required / np.maximum(published, 1.0e-18)

        # Recompute final-radius coverage of every CenterVox uncertainty set.
        # The formal cap family is explicitly bucket-local: a candidate is not
        # a member of another fixed-world bucket's declared family even when
        # its final Euclidean ball happens to overlap that bucket.  Audit
        # unique witnesses in the same candidate universe used by selection.
        bucket_local = self.sphere_cover_mode == "cap_irredundant"
        if bucket_local:
            proxy_bucket_keys = np.asarray(
                proxies.cluster_keys, dtype=np.int64
            )[:, :3]
            cell_bucket_keys = np.asarray(
                [self._proxy_bucket_key(cell.representative) for cell in cells],
                dtype=np.int64,
            )
            if not np.all(cell_bucket_keys == proxy_bucket_keys[owner]):
                raise AssertionError("sphere owner crossed its fixed candidate bucket")
        else:
            proxy_bucket_keys = None
            cell_bucket_keys = None
        cover_counts = np.zeros(len(cells), dtype=np.int64)
        per_proxy_coverage: list[np.ndarray] = []
        for proxy_index, (center, radius) in enumerate(
            zip(proxies.centers, proxies.sphere_radii)
        ):
            eligible = (
                np.flatnonzero(
                    np.all(
                        cell_bucket_keys == proxy_bucket_keys[proxy_index],
                        axis=1,
                    )
                )
                if bucket_local
                else np.arange(len(cells), dtype=np.int64)
            )
            all_required = (
                np.linalg.norm(
                    np.asarray(center, dtype=float)[None, :]
                    - cell_centers[eligible],
                    axis=1,
                )
                + cell_support_radii[eligible]
            )
            covered = eligible[
                all_required <= float(radius) + tolerance
            ]
            per_proxy_coverage.append(covered)
            cover_counts[covered] += 1
        complete = bool(np.all(cover_counts >= 1))
        inclusion_minimal = bool(
            complete
            and all(
                np.any(cover_counts[covered] == 1)
                for covered in per_proxy_coverage
            )
        )
        cap_ok = bool(
            self.certificate_radius_limit is not None
            and np.all(
                np.asarray(proxies.sphere_radii, dtype=float)
                <= self.certificate_radius_limit + tolerance
            )
        )
        raw_source_count = sum(
            int(sample.source_count)
            for cell in cells
            for sample in cell.samples.values()
        )
        certified = bool(
            minimum_slack >= -tolerance
            and complete
            and inclusion_minimal
            and cap_ok
        )
        return ProxyCoverageAudit(
            raw_sample_bins=int(raw_source_count),
            center_voxels=len(cells),
            proxy_count=len(proxies.centers),
            minimum_centervox_cover_slack=minimum_slack,
            minimum_proxy_offset_slack=0.0,
            maximum_base_ellipsoid_value=float(np.max(np.square(ratios))),
            minimum_sphere_certificate_slack=minimum_slack,
            all_raw_sample_balls_certified=certified,
            minimum_measurement_uncertainty_slack=minimum_slack,
            minimum_proxy_uncertainty_slack=0.0,
            sphere_cover_inclusion_minimal=inclusion_minimal,
            sphere_cover_candidate_count=(
                proxies.sphere_cover_candidate_count
            ),
            sphere_cover_selected_count=(
                proxies.sphere_cover_selected_count
            ),
            sphere_cover_reverse_deleted=(
                proxies.sphere_cover_reverse_deleted
            ),
        )

    def _fused_ellipsoid_coverage_audit(
        self, tolerance: float
    ) -> ProxyCoverageAudit:
        """Replay every retained v4.2 fusion and verify the published Q.

        The primitive constructor proves containment by first enclosing every
        translated member ellipsoid in an oriented support box and then
        circumscribing that box by an ellipsoid.  This audit independently
        replays the retained cross-frame and proxy-level constructions.  The
        raw-frame construction uses the same theorem before raw pixels are
        discarded and is covered by unit tests over sampled member surfaces.
        """

        proxies = self.snapshot
        keys = self._last_ordered_keys
        maximum_replay_error = 0.0
        raw_source_count = 0
        if self.uncertainty_fusion_mode == "fused_certified_centervox":
            cells = [self._cells[key] for key in keys]
            if any(
                cell.fusion_rotation is None
                or cell.fusion_lower is None
                or cell.fusion_upper is None
                or cell.fusion_narrow_axis is None
                for cell in cells
            ):
                raise AssertionError("direct CenterVox lost support-box state")
            raw_source_count = sum(
                int(sample.source_count)
                for cell in cells
                for sample in cell.samples.values()
            )
            rotations = np.asarray(
                [cell.fusion_rotation for cell in cells], dtype=float
            )
            lowers = np.asarray([cell.fusion_lower for cell in cells], dtype=float)
            uppers = np.asarray([cell.fusion_upper for cell in cells], dtype=float)
            narrow = np.asarray(
                [cell.fusion_narrow_axis for cell in cells], dtype=np.int64
            )
            local_centers = 0.5 * (lowers + uppers)
            half = np.maximum(0.5 * (uppers - lowers), 1.0e-7)
            narrow_weight = 1.0 / self.direct_thin_axis_inflation**2
            tangent_weight = 0.5 * (1.0 - narrow_weight)
            weights = np.full_like(half, tangent_weight)
            weights[np.arange(len(cells)), narrow] = narrow_weight
            axes_squared = np.square(half) / weights
            replay_centers = np.einsum(
                "nij,nj->ni", rotations, local_centers
            )
            replay_shapes = np.einsum(
                "nik,nk,njk->nij", rotations, axes_squared, rotations
            )
            replay_shapes = 0.5 * (
                replay_shapes + np.swapaxes(replay_shapes, 1, 2)
            )
            cell_centers = np.asarray(
                [cell.representative for cell in cells], dtype=float
            )
            cell_shapes = np.asarray(
                [cell.uncertainty_shape for cell in cells], dtype=float
            )
            maximum_replay_error = max(
                float(np.max(np.abs(replay_centers - cell_centers))),
                float(np.max(np.abs(replay_shapes - cell_shapes))),
            )
            proxy_indices = np.asarray(
                proxies.filtered_cluster_indices, dtype=np.int64
            )
            subproxy_count = self._direct_subproxy_count
            expected = np.arange(len(cells), dtype=np.int64) * subproxy_count
            if not np.array_equal(proxy_indices, expected):
                raise AssertionError(
                    "direct CenterVox coverage lost subproxy ownership"
                )

            # Independently reconstruct the published tangent partition from
            # the retained support boxes.  Calling the publisher here used to
            # repeat proxy IDs, eigendecompositions, and snapshot hashing even
            # though the audit only needs the algebraic centers and Q
            # matrices.  This reconstruction deliberately contains no
            # publication side effects.
            owner = np.repeat(
                np.arange(len(cells), dtype=np.int64), subproxy_count
            )
            local_subindex = np.tile(
                np.arange(subproxy_count, dtype=np.int64), len(cells)
            )
            partition_centers = np.repeat(
                0.5 * (lowers + uppers), subproxy_count, axis=0
            )
            cell_half = np.maximum(0.5 * (uppers - lowers), 1.0e-7)
            partition_half = np.repeat(
                cell_half, subproxy_count, axis=0
            )
            tangent = np.asarray(
                [
                    [axis for axis in range(3) if axis != int(normal_axis)]
                    for normal_axis in narrow
                ],
                dtype=np.int64,
            )
            rows = np.arange(len(owner), dtype=np.int64)
            if self.direct_partition_mode == "longest_tangent_binary":
                tangent_half = cell_half[
                    np.arange(len(cells))[:, None], tangent
                ]
                split_slot = np.argmax(tangent_half, axis=1)
                split_axis = tangent[owner, split_slot[owner]]
                span = uppers[owner, split_axis] - lowers[owner, split_axis]
                partition_centers[rows, split_axis] = (
                    lowers[owner, split_axis]
                    + (local_subindex + 0.5) * span / 2.0
                )
                partition_half[rows, split_axis] = np.maximum(
                    0.25 * span, 1.0e-7
                )
            else:
                subdivisions = self.direct_tangent_subdivisions
                tangent_bins = np.column_stack(
                    (
                        local_subindex // subdivisions,
                        local_subindex % subdivisions,
                    )
                )
                for tangent_slot in range(2):
                    split_axis = tangent[owner, tangent_slot]
                    span = (
                        uppers[owner, split_axis]
                        - lowers[owner, split_axis]
                    )
                    partition_centers[rows, split_axis] = (
                        lowers[owner, split_axis]
                        + (tangent_bins[:, tangent_slot] + 0.5)
                        * span
                        / subdivisions
                    )
                    partition_half[rows, split_axis] = np.maximum(
                        0.5 * span / subdivisions, 1.0e-7
                    )

            partition_weights = np.zeros_like(partition_half)
            partition_weights[rows, narrow[owner]] = narrow_weight
            tangent_budget = 1.0 - narrow_weight
            if self.direct_partition_mode == "longest_tangent_binary":
                tangent_axes = tangent[owner]
                tangent_half_squared = np.square(
                    partition_half[rows[:, None], tangent_axes]
                )
                tangent_denominator = np.sum(
                    tangent_half_squared, axis=1
                )
                partition_weights[rows[:, None], tangent_axes] = (
                    tangent_budget
                    * tangent_half_squared
                    / np.maximum(tangent_denominator[:, None], 1.0e-30)
                )
            else:
                partition_weights[rows[:, None], tangent[owner]] = (
                    0.5 * tangent_budget
                )
            partition_weight_error = float(
                np.max(np.abs(np.sum(partition_weights, axis=1) - 1.0))
            )
            if np.any(partition_weights <= 0.0):
                raise AssertionError(
                    "direct CenterVox partition lost a positive axis weight"
                )
            partition_axes_squared = (
                np.square(partition_half) / partition_weights
            )
            expected_centers = np.einsum(
                "nij,nj->ni", rotations[owner], partition_centers
            )
            expected_shapes = np.einsum(
                "nik,nk,njk->nij",
                rotations[owner],
                partition_axes_squared,
                rotations[owner],
            )
            expected_shapes = 0.5 * (
                expected_shapes + np.swapaxes(expected_shapes, 1, 2)
            )
            expected_radii = np.sqrt(
                np.max(partition_axes_squared, axis=1)
            )
            maximum_replay_error = max(
                maximum_replay_error,
                partition_weight_error,
                float(
                    np.max(
                        np.abs(
                            expected_centers
                            - np.asarray(proxies.centers)
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            expected_shapes
                            - np.asarray(proxies.base_ellipsoid_shapes)
                        )
                    )
                ),
            )
            sphere_slack = float(
                np.min(
                    np.asarray(proxies.sphere_radii, dtype=float)
                    - expected_radii
                )
            )
        else:
            for key in keys:
                cell = self._cells[key]
                samples = list(cell.samples.values())
                raw_source_count += sum(int(item.source_count) for item in samples)
                if not samples or any(
                    item.uncertainty_shape is None for item in samples
                ):
                    raise AssertionError(
                        "fused CenterVox cell lost a member ellipsoid"
                    )
                center, shape, _rotation, _half = (
                    certified_translated_ellipsoid_outer(
                        np.asarray([item.point for item in samples], dtype=float),
                        np.asarray(
                            [item.uncertainty_shape for item in samples],
                            dtype=float,
                        ),
                    )
                )
                maximum_replay_error = max(
                    maximum_replay_error,
                    float(np.max(np.abs(center - cell.representative))),
                    float(np.max(np.abs(shape - cell.uncertainty_shape))),
                )

            proxy_indices = np.asarray(
                proxies.filtered_cluster_indices, dtype=np.int64
            )
            cell_centers = np.asarray(
                [self._cells[key].representative for key in keys], dtype=float
            )
            cell_shapes = np.asarray(
                [self._cells[key].uncertainty_shape for key in keys], dtype=float
            )
            for proxy_index in range(len(proxies.centers)):
                members = np.flatnonzero(proxy_indices == proxy_index)
                if len(members) == 0:
                    raise AssertionError(
                        "published fused proxy has no CenterVox members"
                    )
                center, shape, _rotation, _half = (
                    certified_translated_ellipsoid_outer(
                        cell_centers[members], cell_shapes[members]
                    )
                )
                maximum_replay_error = max(
                    maximum_replay_error,
                    float(np.max(np.abs(center - proxies.centers[proxy_index]))),
                    float(
                        np.max(
                            np.abs(
                                shape
                                - proxies.base_ellipsoid_shapes[proxy_index]
                            )
                        )
                    ),
                )

        # The branches above replay the same support-box theorem. In direct
        # mode each tangent sub-box obeys sum_i (h_i/a_i)^2 = sum_i w_i = 1,
        # so the union of published ellipsoids contains the complete box.

        uncertainty = np.asarray(proxies.proxy_uncertainty_shapes, dtype=float)
        zero_uncertainty_error = float(np.max(np.abs(uncertainty)))
        outer_error = float(
            np.max(
                np.abs(
                    np.asarray(proxies.ellipsoid_outer_shapes, dtype=float)
                    - np.asarray(proxies.base_ellipsoid_shapes, dtype=float)
                )
            )
        )
        offsets = np.asarray(proxies.proxy_offset_radii, dtype=float)
        offset_error = float(np.max(np.abs(offsets)))
        if self.uncertainty_fusion_mode != "fused_certified_centervox":
            longest_axes = np.sqrt(
                np.maximum(
                    np.linalg.eigvalsh(proxies.base_ellipsoid_shapes)[:, -1],
                    0.0,
                )
            )
            sphere_slack = float(
                np.min(
                    np.asarray(proxies.sphere_radii, dtype=float)
                    - longest_axes
                )
            )
        algebra_error = max(
            maximum_replay_error,
            zero_uncertainty_error,
            outer_error,
            offset_error,
        )
        certified = bool(
            algebra_error <= tolerance and sphere_slack >= -tolerance
        )
        return ProxyCoverageAudit(
            raw_sample_bins=int(raw_source_count),
            center_voxels=len(keys),
            proxy_count=len(proxies.centers),
            minimum_centervox_cover_slack=-maximum_replay_error,
            minimum_proxy_offset_slack=-offset_error,
            maximum_base_ellipsoid_value=1.0 + maximum_replay_error,
            minimum_sphere_certificate_slack=sphere_slack,
            all_raw_sample_balls_certified=certified,
            minimum_measurement_uncertainty_slack=-maximum_replay_error,
            minimum_proxy_uncertainty_slack=-max(
                zero_uncertainty_error, outer_error
            ),
        )

    def coverage_audit(self, tolerance: float = 1.0e-9) -> ProxyCoverageAudit:
        """Prove the CenterVox and matched-proxy certificate chain.

        Scalar input proves the nested sample-ball/CenterVox-ball/offset-ball
        chain. Directional input separately proves the center-relocation
        residual delta in metres and U containment with generalized
        eigenvalues, exactly matching rho_Q + rho_U + delta.
        """

        if self.uncertainty_fusion_mode == "raw_directional_uncertainty":
            assert self.raw_directional_join.total_samples > 0
            assert self.raw_directional_join.minimum_psd_slack >= -1e-12
        if self.uncertainty_fusion_mode in {"support_interval_uncertainty", "raw_join_uncertainty"}:
            acc = self.raw_support_intervals
            assert acc.total_samples > 0
            if self.uncertainty_fusion_mode == "raw_join_uncertainty":
                assert acc.minimum_psd_slack >= -1e-12
            else:
                assert acc.minimum_interval_slack >= -1e-12 and acc.max_box_corner_value <= 1+1e-12
        proxies = self.snapshot
        if len(self._last_ordered_keys) != len(proxies.filtered_points):
            raise AssertionError("CenterVox key ordering no longer matches snapshot")
        if self.sphere_cover_mode in {
            "adaptive_irredundant",
            "cap_irredundant",
        }:
            return self._adaptive_sphere_coverage_audit(tolerance)
        if self._uses_fused_uncertainty:
            return self._fused_ellipsoid_coverage_audit(tolerance)
        directional = proxies.proxy_uncertainty_shapes is not None
        if directional:
            return self._directional_coverage_audit(tolerance)

        def containment_value(inner: np.ndarray, outer: np.ndarray) -> float:
            values, vectors = np.linalg.eigh(0.5 * (outer + outer.T))
            values = np.maximum(values, 1.0e-18)
            inverse_root = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
            transformed = inverse_root @ inner @ inverse_root
            return float(
                np.max(np.linalg.eigvalsh(0.5 * (transformed + transformed.T)))
            )

        minimum_center_slack = np.inf
        minimum_offset_slack = np.inf
        maximum_ellipsoid_value = 0.0
        minimum_sphere_slack = np.inf
        sample_bins = 0
        for filtered_index, key in enumerate(self._last_ordered_keys):
            cell = self._cells[key]
            proxy_index = int(proxies.filtered_cluster_indices[filtered_index])
            proxy_center = proxies.centers[proxy_index]
            base_radius = float(proxies.base_sphere_radii[proxy_index])
            representative = cell.representative
            sample_bins += sum(
                int(sample.source_count) for sample in cell.samples.values()
            )
            if directional:
                if cell.uncertainty_shape is None:
                    raise AssertionError("directional CenterVox cell lost U")
                proxy_uncertainty = proxies.proxy_uncertainty_shapes[proxy_index]
                for sample in cell.samples.values():
                    if sample.uncertainty_shape is None:
                        raise AssertionError("directional sample lost U")
                    displacement = sample.point - representative
                    candidate = sample.uncertainty_shape
                    if float(np.linalg.norm(displacement)) > 1.0e-15:
                        candidate = minkowski_outer_shape(
                            candidate, np.outer(displacement, displacement)
                        )
                    minimum_center_slack = min(
                        minimum_center_slack,
                        1.0 - containment_value(candidate, cell.uncertainty_shape),
                    )
                minimum_offset_slack = min(
                    minimum_offset_slack,
                    1.0
                    - containment_value(cell.uncertainty_shape, proxy_uncertainty),
                )
            else:
                if proxies.proxy_offset_radii is None:
                    raise AssertionError("scalar proxy snapshot lost its offsets")
                proxy_offset = float(proxies.proxy_offset_radii[proxy_index])
                for sample in cell.samples.values():
                    required = (
                        float(np.linalg.norm(sample.point - representative))
                        + float(sample.cover_radius)
                    )
                    minimum_center_slack = min(
                        minimum_center_slack, cell.cover_radius - required
                    )
                minimum_offset_slack = min(
                    minimum_offset_slack, proxy_offset - cell.cover_radius
                )
            delta = representative - proxy_center
            ellipsoid_value = float(
                delta
                @ np.linalg.solve(proxies.base_ellipsoid_shapes[proxy_index], delta)
            )
            maximum_ellipsoid_value = max(
                maximum_ellipsoid_value, ellipsoid_value
            )
            if directional:
                # The representative lies in the base sphere and the nested
                # uncertainty ellipsoid lies in proxy U.  Therefore their
                # Minkowski sum lies in a sphere with radius
                # base_radius+sqrt(lambda_max(U)); the latter is exactly the
                # published effective sphere radius.
                minimum_sphere_slack = min(
                    minimum_sphere_slack,
                    base_radius - float(np.linalg.norm(delta)),
                )
            else:
                sphere_slack = (
                    base_radius
                    + proxy_offset
                    - float(np.linalg.norm(delta))
                    - cell.cover_radius
                )
                minimum_sphere_slack = min(minimum_sphere_slack, sphere_slack)
        certified = bool(
            minimum_center_slack >= -tolerance
            and minimum_offset_slack >= -tolerance
            and maximum_ellipsoid_value <= 1.0 + tolerance
            and minimum_sphere_slack >= -tolerance
        )
        return ProxyCoverageAudit(
            raw_sample_bins=int(sample_bins),
            center_voxels=len(self._last_ordered_keys),
            proxy_count=len(proxies.centers),
            minimum_centervox_cover_slack=float(minimum_center_slack),
            minimum_proxy_offset_slack=float(minimum_offset_slack),
            maximum_base_ellipsoid_value=float(maximum_ellipsoid_value),
            minimum_sphere_certificate_slack=float(minimum_sphere_slack),
            all_raw_sample_balls_certified=certified,
        )
