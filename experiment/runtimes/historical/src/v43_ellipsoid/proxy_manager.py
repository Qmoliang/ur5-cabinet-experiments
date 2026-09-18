"""Extracted original v4.3 ellipsoid dependency; see docs/source_manifest.json."""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib
import time
import numpy as np
from proxy_geometry import PointCloudProxySet, fit_matched_voxel_proxies, loewner_union_outer_shapes_grouped_common_frame, minkowski_outer_shape, minkowski_outer_shapes

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

@dataclass
class _CenterSample:
    point: np.ndarray
    cover_radius: float
    uncertainty_shape: np.ndarray | None
    source_count: int = 1

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
    keys: tuple[tuple[int, int, int], ...]
    proxies: PointCloudProxySet

class IncrementalMatchedProxyManager:

    def __init__(self, *, filter_size: float=0.006, cluster_size: float=0.1, maximum_aabb_overshoot: float=0.025, maximum_uncertainty_union_inflation: float | None=None, normal_connection_distance: float | None=None, certificate_radius_limit: float | None=None, uncertainty_fusion_mode: str='separate_uncertainty', direct_thin_axis_inflation: float=np.sqrt(2.0), direct_tangent_subdivisions: int=1, direct_partition_mode: str='grid', origin: np.ndarray=np.array([-1.2, -1.2, 0.0])) -> None:
        if filter_size <= 0.0 or cluster_size <= 0.0:
            raise ValueError('filter_size and cluster_size must be positive')
        self.filter_size = float(filter_size)
        self.cluster_size = float(cluster_size)
        self.maximum_aabb_overshoot = float(maximum_aabb_overshoot)
        self.maximum_uncertainty_union_inflation = None if maximum_uncertainty_union_inflation is None else float(maximum_uncertainty_union_inflation)
        if self.maximum_uncertainty_union_inflation is not None and self.maximum_uncertainty_union_inflation < 1.0:
            raise ValueError('maximum uncertainty union inflation must be >= 1')
        self.normal_connection_distance = 1.75 * self.filter_size if normal_connection_distance is None else float(normal_connection_distance)
        if certificate_radius_limit is not None and certificate_radius_limit <= 0.0:
            raise ValueError('certificate radius limit must be positive')
        self.certificate_radius_limit = None if certificate_radius_limit is None else float(certificate_radius_limit)
        self.uncertainty_fusion_mode = uncertainty_fusion_mode
        self.direct_thin_axis_inflation = float(direct_thin_axis_inflation)
        self.direct_tangent_subdivisions = int(direct_tangent_subdivisions)
        if self.direct_tangent_subdivisions < 1:
            raise ValueError('direct tangent subdivisions must be positive')
        if direct_partition_mode not in {'grid', 'longest_tangent_binary'}:
            raise ValueError('unknown direct partition mode')
        self.direct_partition_mode = direct_partition_mode
        if min(abs(self.direct_thin_axis_inflation - 1.0), abs(self.direct_thin_axis_inflation - np.sqrt(2.0))) > 1e-12 or self.direct_tangent_subdivisions != 1 or self.direct_partition_mode != 'grid':
            raise ValueError('inactive direct thin-axis controls must use identity 1.0 or the legacy default sqrt(2)')
        self.origin = np.asarray(origin, dtype=float).reshape(3)
        self._cells: dict[tuple[int, int, int], _CenterVoxel] = {}
        self._generation = -1
        self._snapshot: PointCloudProxySet | None = None
        self._snapshot_hash = ''
        self._directional: bool | None = None
        self._sample_bin = max(0.00025, self.filter_size / 64.0)
        self._next_proxy_id = 0
        self._previous_memberships: dict[int, frozenset[tuple[int, int, int]]] = {}
        self._last_ordered_keys: tuple[tuple[int, int, int], ...] = ()
        self._bucket_fits: dict[tuple[int, int, int], _BucketFit] = {}
        self._direct_subproxy_ids: dict[tuple[tuple[int, int, int], int], int] = {}

    @property
    def _direct_subproxy_count(self) -> int:
        if self.direct_partition_mode == 'longest_tangent_binary':
            return 2
        return self.direct_tangent_subdivisions ** 2

    @property
    def _uses_fused_uncertainty(self) -> bool:
        return self.uncertainty_fusion_mode in {'fused_certified_ellipsoid', 'fused_certified_centervox'}

    def _proxy_bucket_key(self, representative: np.ndarray) -> tuple[int, int, int]:
        key = np.floor((np.asarray(representative, dtype=float) - self.origin) / self.cluster_size).astype(np.int64)
        return (int(key[0]), int(key[1]), int(key[2]))

    @staticmethod
    def _concatenate_proxy_field(bucket_fits: list[_BucketFit], field: str) -> np.ndarray:
        return np.concatenate([np.asarray(getattr(item.proxies, field)) for item in bucket_fits], axis=0)

    def _fit_incremental_snapshot(self, ordered_keys: list[tuple[int, int, int]], changed_keys: set[tuple[int, int, int]], directional: bool) -> PointCloudProxySet:
        members: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
        for key in ordered_keys:
            bucket = self._proxy_bucket_key(self._cells[key].representative)
            members.setdefault(bucket, []).append(key)
        changed_buckets = {self._proxy_bucket_key(self._cells[key].representative) for key in changed_keys}
        previous_changed_buckets = {bucket for bucket, item in self._bucket_fits.items() if not changed_keys.isdisjoint(item.keys)}
        changed_buckets.update((bucket for bucket in previous_changed_buckets if bucket in members))
        changed_buckets.update((bucket for bucket in members if bucket not in self._bucket_fits))
        for bucket in set(self._bucket_fits).difference(members):
            del self._bucket_fits[bucket]
        if changed_buckets:
            batch_buckets = sorted(changed_buckets)
            batch_keys = [key for bucket in batch_buckets for key in members[bucket]]
            batch_labels = np.asarray([bucket for bucket in batch_buckets for _key in members[bucket]], dtype=np.int64)
            points = np.asarray([self._cells[key].representative for key in batch_keys], dtype=float)
            offsets = np.asarray([self._cells[key].cover_radius for key in batch_keys], dtype=float)
            uncertainty = np.asarray([self._cells[key].uncertainty_shape for key in batch_keys], dtype=float) if directional else None
            batch_fit = fit_matched_voxel_proxies(points, filter_size=self.filter_size, cluster_size=self.cluster_size, maximum_aabb_overshoot=self.maximum_aabb_overshoot, already_filtered=True, filtered_point_offsets=offsets, filtered_point_uncertainty_shapes=uncertainty, normal_connection_distance=self.normal_connection_distance, cluster_origin=self.origin, maximum_uncertainty_union_inflation=self.maximum_uncertainty_union_inflation, certificate_radius_limit=self.certificate_radius_limit, uncertainty_fusion_mode=self.uncertainty_fusion_mode)
            batch_proxy_indices = np.asarray(batch_fit.filtered_cluster_indices, dtype=np.int64)
            for bucket in batch_buckets:
                input_indices = np.flatnonzero(np.all(batch_labels == np.asarray(bucket), axis=1))
                selected_proxies = np.unique(batch_proxy_indices[input_indices])
                for proxy_index in selected_proxies:
                    proxy_members = np.flatnonzero(batch_proxy_indices == proxy_index)
                    if not np.all(batch_labels[proxy_members] == np.asarray(bucket)):
                        raise AssertionError('one fitted proxy crossed fixed buckets')
                remap = {int(proxy_index): local_index for local_index, proxy_index in enumerate(selected_proxies)}
                local_indices = np.asarray([remap[int(value)] for value in batch_proxy_indices[input_indices]], dtype=np.int64)
                local = PointCloudProxySet(raw_points=batch_fit.filtered_points[input_indices].copy(), filtered_points=batch_fit.filtered_points[input_indices].copy(), centers=batch_fit.centers[selected_proxies].copy(), sphere_radii=batch_fit.sphere_radii[selected_proxies].copy(), ellipsoid_shapes=batch_fit.ellipsoid_shapes[selected_proxies].copy(), rotations=batch_fit.rotations[selected_proxies].copy(), cluster_keys=batch_fit.cluster_keys[selected_proxies].copy(), filtered_cluster_indices=local_indices, maximum_ellipsoid_overshoot=batch_fit.maximum_ellipsoid_overshoot, surface_cover_radius=batch_fit.surface_cover_radius, proxy_offset_radii=batch_fit.proxy_offset_radii[selected_proxies].copy(), filtered_uncertainty_shapes=batch_fit.filtered_uncertainty_shapes[input_indices].copy() if directional else None, proxy_uncertainty_shapes=batch_fit.proxy_uncertainty_shapes[selected_proxies].copy() if directional else None, base_sphere_radii=batch_fit.base_sphere_radii[selected_proxies].copy(), base_ellipsoid_shapes=batch_fit.base_ellipsoid_shapes[selected_proxies].copy(), ellipsoid_outer_shapes=batch_fit.ellipsoid_outer_shapes[selected_proxies].copy(), maximum_uncertainty_union_inflation=batch_fit.maximum_uncertainty_union_inflation, certificate_radius_limit=batch_fit.certificate_radius_limit, uncertainty_fusion_mode=self.uncertainty_fusion_mode)
                self._bucket_fits[bucket] = _BucketFit(tuple(members[bucket]), local)
        bucket_items = [self._bucket_fits[key] for key in sorted(members)]
        key_to_proxy: dict[tuple[int, int, int], int] = {}
        proxy_base = 0
        global_cluster_keys: list[np.ndarray] = []
        for bucket_key, item in zip(sorted(members), bucket_items):
            if item.keys != tuple(members[bucket_key]):
                raise AssertionError('cached proxy bucket membership is stale')
            local_indices = np.asarray(item.proxies.filtered_cluster_indices, dtype=np.int64)
            for key, local_index in zip(item.keys, local_indices):
                key_to_proxy[key] = proxy_base + int(local_index)
            local_count = len(item.proxies.centers)
            global_cluster_keys.append(np.column_stack((np.repeat(np.asarray(bucket_key, dtype=np.int64)[None, :], local_count, axis=0), np.arange(local_count, dtype=np.int64))))
            proxy_base += local_count
        filtered = np.asarray([self._cells[key].representative for key in ordered_keys], dtype=float)
        filtered_uncertainty = np.asarray([self._cells[key].uncertainty_shape for key in ordered_keys], dtype=float) if directional else None
        return PointCloudProxySet(raw_points=filtered.copy(), filtered_points=filtered, centers=self._concatenate_proxy_field(bucket_items, 'centers'), sphere_radii=self._concatenate_proxy_field(bucket_items, 'sphere_radii'), ellipsoid_shapes=self._concatenate_proxy_field(bucket_items, 'ellipsoid_shapes'), rotations=self._concatenate_proxy_field(bucket_items, 'rotations'), cluster_keys=np.concatenate(global_cluster_keys, axis=0), filtered_cluster_indices=np.asarray([key_to_proxy[key] for key in ordered_keys], dtype=np.int64), maximum_ellipsoid_overshoot=max((item.proxies.maximum_ellipsoid_overshoot for item in bucket_items)), surface_cover_radius=max((item.proxies.surface_cover_radius for item in bucket_items)), proxy_offset_radii=self._concatenate_proxy_field(bucket_items, 'proxy_offset_radii'), filtered_uncertainty_shapes=filtered_uncertainty, proxy_uncertainty_shapes=self._concatenate_proxy_field(bucket_items, 'proxy_uncertainty_shapes') if directional else None, base_sphere_radii=self._concatenate_proxy_field(bucket_items, 'base_sphere_radii'), base_ellipsoid_shapes=self._concatenate_proxy_field(bucket_items, 'base_ellipsoid_shapes'), ellipsoid_outer_shapes=self._concatenate_proxy_field(bucket_items, 'ellipsoid_outer_shapes'), maximum_uncertainty_union_inflation=max((item.proxies.maximum_uncertainty_union_inflation for item in bucket_items)), certificate_radius_limit=self.certificate_radius_limit, uncertainty_fusion_mode=self.uncertainty_fusion_mode)

    def _persistent_proxy_ids(self, proxies: PointCloudProxySet, ordered_keys: list[tuple[int, int, int]]) -> tuple[np.ndarray, dict[int, frozenset[tuple[int, int, int]]]]:
        memberships = [frozenset((ordered_keys[index] for index in np.flatnonzero(proxies.filtered_cluster_indices == proxy_index))) for proxy_index in range(len(proxies.centers))]
        assigned_current: set[int] = set()
        assigned_previous: set[int] = set()
        ids = np.full(len(memberships), -1, dtype=np.int64)
        candidates: list[tuple[int, float, int, int]] = []
        previous_owner: dict[tuple[int, int, int], int] = {}
        for previous_id, previous in self._previous_memberships.items():
            for key in previous:
                if key in previous_owner:
                    raise AssertionError('previous proxy memberships overlap')
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
                candidates.append((intersection, intersection / max(union, 1), previous_id, current_index))
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
        current_memberships = {int(proxy_id): membership for proxy_id, membership in zip(ids, memberships)}
        return (ids, current_memberships)

    def _key(self, point: np.ndarray) -> tuple[int, int, int]:
        key = np.floor((point - self.origin) / self.filter_size).astype(np.int64)
        return (int(key[0]), int(key[1]), int(key[2]))

    def _voxel_center(self, key: tuple[int, int, int]) -> np.ndarray:
        return self.origin + (np.asarray(key, dtype=float) + 0.5) * self.filter_size

    def _sample_key(self, point: np.ndarray) -> tuple[int, int, int]:
        key = np.floor(np.asarray(point) / self._sample_bin).astype(np.int64)
        return (int(key[0]), int(key[1]), int(key[2]))

    def update(self, points: np.ndarray, point_radii: np.ndarray, point_uncertainty_shapes: np.ndarray | None=None) -> ProxyUpdateStats:
        started = time.perf_counter()
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        point_radii = np.asarray(point_radii, dtype=float).reshape(-1)
        if len(points) != len(point_radii):
            raise ValueError('point_radii must contain one radius per point')
        directional = point_uncertainty_shapes is not None
        if self._directional is None:
            self._directional = directional
        elif self._directional != directional:
            raise ValueError('one manager cannot mix scalar and directional updates')
        if directional:
            point_uncertainty_shapes = np.asarray(point_uncertainty_shapes, dtype=float).reshape(-1, 3, 3)
            if len(point_uncertainty_shapes) != len(points):
                raise ValueError('point_uncertainty_shapes must contain one shape per point')
        integer_keys = np.floor((points - self.origin[None, :]) / self.filter_size).astype(np.int64)
        unique_keys, inverse = np.unique(integer_keys, axis=0, return_inverse=True)
        group_count = len(unique_keys)
        voxel_centers = self.origin[None, :] + (unique_keys.astype(float) + 0.5) * self.filter_size
        distances = np.linalg.norm(points - voxel_centers[inverse], axis=1)
        point_order = np.lexsort((np.arange(len(points), dtype=np.int64), distances, inverse))
        first = np.r_[0, np.flatnonzero(np.diff(inverse[point_order])) + 1]
        representative_indices = point_order[first]
        frame_representatives = points[representative_indices].copy()
        key_tuples = [tuple(map(int, key)) for key in unique_keys]
        for group_index, key in enumerate(key_tuples):
            old = self._cells.get(key)
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
        if directional:
            relocation_shapes = np.einsum('ni,nj->nij', displacements, displacements)
            translated_uncertainty = minkowski_outer_shapes(point_uncertainty_shapes, relocation_shapes)
            frame_uncertainty = loewner_union_outer_shapes_grouped_common_frame(translated_uncertainty, inverse, group_count)
            containment = self._batch_containment_values(translated_uncertainty, frame_uncertainty[inverse])
            if float(np.max(containment)) > 1.0 + 1e-09:
                raise AssertionError('frame CenterVox U failed to cover translated input U')
            frame_cover.fill(0.0)
        frame_centervox_done = time.perf_counter()
        frame_generation = self._generation + 1
        samples_by_group: list[list[_CenterSample]] = []
        sample_points: list[np.ndarray] = []
        sample_covers: list[float] = []
        sample_uncertainties: list[np.ndarray] = []
        sample_owners: list[int] = []
        for group_index, key in enumerate(key_tuples):
            samples = [] if self.uncertainty_fusion_mode == 'fused_certified_centervox' or key not in self._cells else list(self._cells[key].samples.values())
            samples.append(_CenterSample(point=frame_representatives[group_index].copy(), cover_radius=float(frame_cover[group_index]), uncertainty_shape=None if not directional else frame_uncertainty[group_index].copy(), source_count=int(group_sizes[group_index])))
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
        aggregate_distances = np.linalg.norm(aggregate_points - voxel_centers[owners], axis=1)
        aggregate_order = np.lexsort((np.arange(len(aggregate_points), dtype=np.int64), aggregate_distances, owners))
        aggregate_first = np.r_[0, np.flatnonzero(np.diff(owners[aggregate_order])) + 1]
        selected = aggregate_order[aggregate_first]
        merged_representatives = aggregate_points[selected]
        aggregate_displacements = aggregate_points - merged_representatives[owners]
        merged_required = np.linalg.norm(aggregate_displacements, axis=1) + aggregate_covers
        merged_cover = np.zeros(group_count, dtype=float)
        np.maximum.at(merged_cover, owners, merged_required)
        merged_uncertainty = None
        fusion_rotations = None
        fusion_lowers = None
        fusion_uppers = None
        fusion_narrow_axes = None
        if directional:
            relocation_shapes = np.einsum('ni,nj->nij', aggregate_displacements, aggregate_displacements)
            translated_uncertainty = minkowski_outer_shapes(np.asarray(sample_uncertainties, dtype=float), relocation_shapes)
            merged_uncertainty = loewner_union_outer_shapes_grouped_common_frame(translated_uncertainty, owners, group_count)
            merged_cover.fill(0.0)
        history_union_done = time.perf_counter()
        changed: set[tuple[int, int, int]] = set()
        for group_index, key in enumerate(key_tuples):
            previous = self._cells.get(key)
            samples = samples_by_group[group_index]
            sample_dict = {(index, 0, 0): sample for index, sample in enumerate(samples)}
            uncertainty_shape = None if not directional else merged_uncertainty[group_index].copy()
            representative = merged_representatives[group_index].copy()
            current = _CenterVoxel(representative=representative, representative_distance=float(np.linalg.norm(representative - voxel_centers[group_index])), cover_radius=float(merged_cover[group_index]), uncertainty_shape=uncertainty_shape, samples=sample_dict, fusion_rotation=None if fusion_rotations is None else fusion_rotations[group_index].copy(), fusion_lower=None if fusion_lowers is None else fusion_lowers[group_index].copy(), fusion_upper=None if fusion_uppers is None else fusion_uppers[group_index].copy(), fusion_narrow_axis=None if fusion_narrow_axes is None else int(fusion_narrow_axes[group_index]))
            self._cells[key] = current
            if previous is None or not np.array_equal(previous.representative, current.representative) or previous.cover_radius != current.cover_radius or (directional and (not np.array_equal(previous.uncertainty_shape, current.uncertainty_shape))):
                changed.add(key)
        if not self._cells:
            raise ValueError('cannot publish proxies before receiving any points')
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
                    raise AssertionError('directional CenterVox cell lost its shape')
                uncertainty_shapes.append(cell.uncertainty_shape)
        filtered = np.asarray(representatives)
        offsets = np.asarray(cover_radii)
        cell_merge_done = time.perf_counter()
        proxies = self._fit_incremental_snapshot(ordered_keys, changed, directional)
        bucket_fit_done = time.perf_counter()
        proxy_ids, memberships = self._persistent_proxy_ids(proxies, ordered_keys)
        proxies = replace(proxies, proxy_ids=proxy_ids)
        digest = hashlib.sha256()
        for array in (np.column_stack([proxies.centers, proxies.sphere_radii, proxies.proxy_offset_radii]), proxies.ellipsoid_shapes, proxies.proxy_uncertainty_shapes, proxies.ellipsoid_outer_shapes, proxy_ids):
            contiguous = np.ascontiguousarray(array)
            digest.update(memoryview(contiguous).cast('B'))
        self._snapshot_hash = digest.hexdigest()
        self._snapshot = proxies
        self._previous_memberships = memberships
        self._last_ordered_keys = tuple(ordered_keys)
        self._generation += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ProxyUpdateStats(generation=self._generation, inserted_points=len(points), center_voxels=len(filtered), proxy_count=len(proxies.centers), changed_center_voxels=len(changed), update_ms=float(elapsed_ms), snapshot_sha256=self._snapshot_hash, frame_centervox_ms=float((frame_centervox_done - started) * 1000.0), history_union_ms=float((history_union_done - frame_centervox_done) * 1000.0), cell_merge_ms=float((cell_merge_done - history_union_done) * 1000.0), bucket_fit_ms=float((bucket_fit_done - cell_merge_done) * 1000.0), publish_ids_hash_ms=float((time.perf_counter() - bucket_fit_done) * 1000.0))

    @property
    def snapshot(self) -> PointCloudProxySet:
        if self._snapshot is None:
            raise RuntimeError('no proxy snapshot has been published')
        return self._snapshot

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def snapshot_hash(self) -> str:
        return self._snapshot_hash

    @staticmethod
    def _batch_containment_values(inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
        inner = np.asarray(inner, dtype=float).reshape(-1, 3, 3)
        outer = np.asarray(outer, dtype=float).reshape(-1, 3, 3)
        if len(inner) != len(outer):
            raise ValueError('inner and outer batches must have equal length')
        symmetric_outer = 0.5 * (outer + np.swapaxes(outer, 1, 2))
        values, vectors = np.linalg.eigh(symmetric_outer)
        inverse_sqrt = 1.0 / np.sqrt(np.maximum(values, 1e-18))
        inverse_root = np.einsum('nij,nj,nkj->nik', vectors, inverse_sqrt, vectors)
        symmetric_inner = 0.5 * (inner + np.swapaxes(inner, 1, 2))
        transformed = np.einsum('nij,njk,nkl->nil', inverse_root, symmetric_inner, inverse_root)
        transformed = 0.5 * (transformed + np.swapaxes(transformed, 1, 2))
        return np.linalg.eigvalsh(transformed)[:, -1]

    def _directional_coverage_audit(self, tolerance: float) -> ProxyCoverageAudit:
        proxies = self.snapshot
        keys = self._last_ordered_keys
        representatives = np.asarray([self._cells[key].representative for key in keys], dtype=float)
        proxy_indices = np.asarray(proxies.filtered_cluster_indices, dtype=np.int64)
        cell_shapes = np.asarray([self._cells[key].uncertainty_shape for key in keys], dtype=float)
        sample_points: list[np.ndarray] = []
        sample_shapes: list[np.ndarray] = []
        sample_owners: list[int] = []
        raw_source_count = 0
        for cell_index, key in enumerate(keys):
            for sample in self._cells[key].samples.values():
                if sample.uncertainty_shape is None:
                    raise AssertionError('directional sample lost U')
                sample_points.append(sample.point)
                sample_shapes.append(sample.uncertainty_shape)
                sample_owners.append(cell_index)
                raw_source_count += int(sample.source_count)
        owners = np.asarray(sample_owners, dtype=np.int64)
        displacements = np.asarray(sample_points, dtype=float) - representatives[owners]
        candidates = minkowski_outer_shapes(np.asarray(sample_shapes, dtype=float), np.einsum('ni,nj->nij', displacements, displacements))
        center_values = self._batch_containment_values(candidates, cell_shapes[owners])
        proxy_uncertainty = np.asarray(proxies.proxy_uncertainty_shapes, dtype=float)[proxy_indices]
        offset_values = self._batch_containment_values(cell_shapes, proxy_uncertainty)
        deltas = representatives - proxies.centers[proxy_indices]
        base_shapes = np.asarray(proxies.base_ellipsoid_shapes, dtype=float)[proxy_indices]
        solved = np.linalg.solve(base_shapes, deltas[:, :, None])[:, :, 0]
        ellipsoid_values = np.einsum('ni,ni->n', deltas, solved)
        base_radii = np.asarray(proxies.base_sphere_radii, dtype=float)[proxy_indices]
        published_effective_radii = (np.asarray(proxies.sphere_radii, dtype=float) + np.asarray(proxies.proxy_offset_radii, dtype=float))[proxy_indices]
        cell_uncertainty_radii = np.sqrt(np.maximum(np.linalg.eigvalsh(cell_shapes)[:, -1], 0.0))
        sphere_slacks = published_effective_radii - np.linalg.norm(deltas, axis=1) - cell_uncertainty_radii
        cell_residual = np.asarray([self._cells[key].cover_radius for key in keys], dtype=float)
        minimum_center_slack = float(np.min(1.0 - center_values))
        proxy_offsets = np.asarray(proxies.proxy_offset_radii, dtype=float)
        minimum_offset_slack = float(np.min(proxy_offsets[proxy_indices] - cell_residual))
        minimum_measurement_uncertainty_slack = float(np.min(1.0 - center_values))
        minimum_proxy_uncertainty_slack = float(np.min(1.0 - offset_values))
        maximum_ellipsoid_value = float(np.max(ellipsoid_values))
        minimum_sphere_slack = float(np.min(sphere_slacks))
        certified = bool(minimum_center_slack >= -tolerance and minimum_offset_slack >= -tolerance and (minimum_measurement_uncertainty_slack >= -tolerance) and (minimum_proxy_uncertainty_slack >= -tolerance) and (maximum_ellipsoid_value <= 1.0 + tolerance) and (minimum_sphere_slack >= -tolerance))
        return ProxyCoverageAudit(raw_sample_bins=raw_source_count, center_voxels=len(keys), proxy_count=len(proxies.centers), minimum_centervox_cover_slack=minimum_center_slack, minimum_proxy_offset_slack=minimum_offset_slack, maximum_base_ellipsoid_value=maximum_ellipsoid_value, minimum_sphere_certificate_slack=minimum_sphere_slack, all_raw_sample_balls_certified=certified, minimum_measurement_uncertainty_slack=minimum_measurement_uncertainty_slack, minimum_proxy_uncertainty_slack=minimum_proxy_uncertainty_slack)

    def coverage_audit(self, tolerance: float=1e-09) -> ProxyCoverageAudit:
        proxies = self.snapshot
        if len(self._last_ordered_keys) != len(proxies.filtered_points):
            raise AssertionError('CenterVox key ordering no longer matches snapshot')
        directional = proxies.proxy_uncertainty_shapes is not None
        if directional:
            return self._directional_coverage_audit(tolerance)

        def containment_value(inner: np.ndarray, outer: np.ndarray) -> float:
            values, vectors = np.linalg.eigh(0.5 * (outer + outer.T))
            values = np.maximum(values, 1e-18)
            inverse_root = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
            transformed = inverse_root @ inner @ inverse_root
            return float(np.max(np.linalg.eigvalsh(0.5 * (transformed + transformed.T))))
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
            sample_bins += sum((int(sample.source_count) for sample in cell.samples.values()))
            if directional:
                if cell.uncertainty_shape is None:
                    raise AssertionError('directional CenterVox cell lost U')
                proxy_uncertainty = proxies.proxy_uncertainty_shapes[proxy_index]
                for sample in cell.samples.values():
                    if sample.uncertainty_shape is None:
                        raise AssertionError('directional sample lost U')
                    displacement = sample.point - representative
                    candidate = sample.uncertainty_shape
                    if float(np.linalg.norm(displacement)) > 1e-15:
                        candidate = minkowski_outer_shape(candidate, np.outer(displacement, displacement))
                    minimum_center_slack = min(minimum_center_slack, 1.0 - containment_value(candidate, cell.uncertainty_shape))
                minimum_offset_slack = min(minimum_offset_slack, 1.0 - containment_value(cell.uncertainty_shape, proxy_uncertainty))
            else:
                if proxies.proxy_offset_radii is None:
                    raise AssertionError('scalar proxy snapshot lost its offsets')
                proxy_offset = float(proxies.proxy_offset_radii[proxy_index])
                for sample in cell.samples.values():
                    required = float(np.linalg.norm(sample.point - representative)) + float(sample.cover_radius)
                    minimum_center_slack = min(minimum_center_slack, cell.cover_radius - required)
                minimum_offset_slack = min(minimum_offset_slack, proxy_offset - cell.cover_radius)
            delta = representative - proxy_center
            ellipsoid_value = float(delta @ np.linalg.solve(proxies.base_ellipsoid_shapes[proxy_index], delta))
            maximum_ellipsoid_value = max(maximum_ellipsoid_value, ellipsoid_value)
            if directional:
                minimum_sphere_slack = min(minimum_sphere_slack, base_radius - float(np.linalg.norm(delta)))
            else:
                sphere_slack = base_radius + proxy_offset - float(np.linalg.norm(delta)) - cell.cover_radius
                minimum_sphere_slack = min(minimum_sphere_slack, sphere_slack)
        certified = bool(minimum_center_slack >= -tolerance and minimum_offset_slack >= -tolerance and (maximum_ellipsoid_value <= 1.0 + tolerance) and (minimum_sphere_slack >= -tolerance))
        return ProxyCoverageAudit(raw_sample_bins=int(sample_bins), center_voxels=len(self._last_ordered_keys), proxy_count=len(proxies.centers), minimum_centervox_cover_slack=float(minimum_center_slack), minimum_proxy_offset_slack=float(minimum_offset_slack), maximum_base_ellipsoid_value=float(maximum_ellipsoid_value), minimum_sphere_certificate_slack=float(minimum_sphere_slack), all_raw_sample_balls_certified=certified)
