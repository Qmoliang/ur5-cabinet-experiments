"""Extracted original v4.3 ellipsoid dependency; see docs/source_manifest.json."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial import cKDTree
from robot import BoxObstacle
NORMAL_ESTIMATION_WORKERS = 4

@dataclass(frozen=True)
class PointCloudProxySet:
    raw_points: np.ndarray
    filtered_points: np.ndarray
    centers: np.ndarray
    sphere_radii: np.ndarray
    ellipsoid_shapes: np.ndarray
    rotations: np.ndarray
    cluster_keys: np.ndarray
    filtered_cluster_indices: np.ndarray
    maximum_ellipsoid_overshoot: float
    surface_cover_radius: float = 0.0
    validation_spacing: float = 0.0
    validation_sampled_max_distance: float = 0.0
    validation_lipschitz_correction: float = 0.0
    proxy_offset_radii: np.ndarray | None = None
    filtered_uncertainty_shapes: np.ndarray | None = None
    proxy_uncertainty_shapes: np.ndarray | None = None
    base_sphere_radii: np.ndarray | None = None
    base_ellipsoid_shapes: np.ndarray | None = None
    ellipsoid_outer_shapes: np.ndarray | None = None
    proxy_ids: np.ndarray | None = None
    maximum_uncertainty_union_inflation: float = 1.0
    certificate_radius_limit: float | None = None
    uncertainty_fusion_mode: str = 'separate_uncertainty'

def minkowski_outer_shape(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = 0.5 * (np.asarray(first, dtype=float) + np.asarray(first, dtype=float).T)
    second = 0.5 * (np.asarray(second, dtype=float) + np.asarray(second, dtype=float).T)
    first_trace = max(float(np.trace(first)), 1e-18)
    second_trace = max(float(np.trace(second)), 1e-18)
    beta = math.sqrt(second_trace / first_trace)
    return (1.0 + beta) * first + (1.0 + 1.0 / beta) * second

def minkowski_outer_shapes(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first, second = np.broadcast_arrays(np.asarray(first, dtype=float), np.asarray(second, dtype=float))
    first = 0.5 * (first + np.swapaxes(first, -1, -2))
    second = 0.5 * (second + np.swapaxes(second, -1, -2))
    first_trace = np.maximum(np.trace(first, axis1=-2, axis2=-1), 1e-18)
    second_trace = np.maximum(np.trace(second, axis1=-2, axis2=-1), 1e-18)
    beta = np.sqrt(second_trace / first_trace)
    return (1.0 + beta)[..., None, None] * first + (1.0 + 1.0 / beta)[..., None, None] * second

def loewner_union_outer_shape(shapes: np.ndarray) -> np.ndarray:
    shapes = np.asarray(shapes, dtype=float).reshape(-1, 3, 3)
    if len(shapes) == 0:
        return np.eye(3) * 1e-18
    symmetric = 0.5 * (shapes + np.swapaxes(shapes, 1, 2))
    if len(symmetric) == 1:
        return symmetric[0].copy()
    reference = np.mean(symmetric, axis=0) + np.eye(3) * 1e-18
    values, vectors = np.linalg.eigh(reference)
    values = np.maximum(values, 1e-18)
    inverse_root = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    whitened = np.einsum('ij,njk,kl->nil', inverse_root, symmetric, inverse_root)
    whitened = 0.5 * (whitened + np.swapaxes(whitened, 1, 2))
    scale = max(1.0, float(np.max(np.linalg.eigvalsh(whitened)[:, -1])))
    scaled_mean = 0.5 * scale * (reference + reference.T)
    traces = np.trace(symmetric, axis1=1, axis2=2)
    flat = symmetric.reshape(len(symmetric), 9)
    order = np.lexsort(tuple((flat[:, column] for column in range(8, -1, -1))) + (-traces,))
    joined = symmetric[int(order[0])].copy()
    for index in order[1:]:
        difference = 0.5 * (symmetric[int(index)] - joined + (symmetric[int(index)] - joined).T)
        diff_values, diff_vectors = np.linalg.eigh(difference)
        positive = diff_vectors @ np.diag(np.maximum(diff_values, 0.0)) @ diff_vectors.T
        joined = 0.5 * (joined + positive + (joined + positive).T)
    minimum_slack = float(np.min(np.linalg.eigvalsh(joined[None, :, :] - symmetric)[:, 0]))
    if minimum_slack < 0.0:
        joined += np.eye(3) * (-minimum_slack + 1e-18)
    return joined if float(np.trace(joined)) < float(np.trace(scaled_mean)) else scaled_mean

def loewner_union_outer_shapes_grouped(shapes: np.ndarray, group_indices: np.ndarray, group_count: int | None=None) -> np.ndarray:
    shapes = np.asarray(shapes, dtype=float).reshape(-1, 3, 3)
    groups = np.asarray(group_indices, dtype=np.int64).reshape(-1)
    if len(shapes) != len(groups):
        raise ValueError('one group index is required per shape')
    if len(shapes) == 0:
        return np.empty((0, 3, 3), dtype=float)
    count = int(np.max(groups)) + 1 if group_count is None else int(group_count)
    if count <= 0 or np.min(groups) < 0 or np.max(groups) >= count:
        raise ValueError('group indices must be dense and in range')
    symmetric = 0.5 * (shapes + np.swapaxes(shapes, 1, 2))
    traces = np.trace(symmetric, axis1=1, axis2=2)
    flat = symmetric.reshape(len(symmetric), 9)
    order = np.lexsort(tuple((flat[:, column] for column in range(8, -1, -1))) + (-traces, groups))
    counts = np.bincount(groups, minlength=count)
    if np.any(counts == 0):
        raise ValueError('every group must contain at least one shape')
    starts = np.r_[0, np.cumsum(counts[:-1])]
    joined = symmetric[order[starts]].copy()
    maximum_rank = int(np.max(counts))
    for rank in range(1, maximum_rank):
        active = np.flatnonzero(counts > rank)
        members = symmetric[order[starts[active] + rank]]
        difference = 0.5 * (members - joined[active] + np.swapaxes(members - joined[active], 1, 2))
        values, vectors = np.linalg.eigh(difference)
        positive = np.einsum('nij,nj,nkj->nik', vectors, np.maximum(values, 0.0), vectors)
        joined[active] = 0.5 * (joined[active] + positive + np.swapaxes(joined[active] + positive, 1, 2))
    slacks = np.linalg.eigvalsh(joined[groups] - symmetric)[:, 0]
    minimum_by_group = np.full(count, np.inf)
    np.minimum.at(minimum_by_group, groups, slacks)
    repair = np.maximum(-minimum_by_group, 0.0) + 1e-18
    joined += repair[:, None, None] * np.eye(3)[None, :, :]
    references = np.zeros((count, 3, 3), dtype=float)
    np.add.at(references, groups, symmetric)
    references /= counts[:, None, None]
    references += np.eye(3)[None, :, :] * 1e-18
    ref_values, ref_vectors = np.linalg.eigh(references)
    inverse_root = np.einsum('nij,nj,nkj->nik', ref_vectors, 1.0 / np.sqrt(np.maximum(ref_values, 1e-18)), ref_vectors)
    whitened = np.einsum('nij,njk,nkl->nil', inverse_root[groups], symmetric, inverse_root[groups])
    whitened = 0.5 * (whitened + np.swapaxes(whitened, 1, 2))
    member_scales = np.linalg.eigvalsh(whitened)[:, -1]
    scales = np.ones(count, dtype=float)
    np.maximum.at(scales, groups, member_scales)
    scaled_means = scales[:, None, None] * references
    use_joined = np.trace(joined, axis1=1, axis2=2) < np.trace(scaled_means, axis1=1, axis2=2)
    return np.where(use_joined[:, None, None], joined, scaled_means)

def loewner_union_outer_shapes_grouped_common_frame(shapes: np.ndarray, group_indices: np.ndarray, group_count: int | None=None) -> np.ndarray:
    shapes = np.asarray(shapes, dtype=float).reshape(-1, 3, 3)
    groups = np.asarray(group_indices, dtype=np.int64).reshape(-1)
    if len(shapes) != len(groups):
        raise ValueError('one group index is required per shape')
    if len(shapes) == 0:
        return np.empty((0, 3, 3), dtype=float)
    count = int(np.max(groups)) + 1 if group_count is None else int(group_count)
    if count <= 0 or np.min(groups) < 0 or np.max(groups) >= count:
        raise ValueError('group indices must be dense and in range')
    counts = np.bincount(groups, minlength=count)
    if np.any(counts == 0):
        raise ValueError('every group must contain at least one shape')
    symmetric = 0.5 * (shapes + np.swapaxes(shapes, 1, 2))
    references = np.zeros((count, 3, 3), dtype=float)
    np.add.at(references, groups, symmetric)
    references /= counts[:, None, None]
    _, frames = np.linalg.eigh(references)
    member_frames = frames[groups]
    transformed = np.einsum('nji,njk,nkl->nil', member_frames, symmetric, member_frames)
    diagonal = np.diagonal(transformed, axis1=1, axis2=2)
    row_bounds = diagonal + (np.sum(np.abs(transformed), axis=2) - np.abs(diagonal))
    bounds = np.zeros((count, 3), dtype=float)
    np.maximum.at(bounds, groups, row_bounds)
    bounds += (np.maximum(np.max(bounds, axis=1), 1e-18) * 2e-14 + 1e-18)[:, None]
    return np.einsum('nij,nj,nkj->nik', frames, bounds, frames)

def _axis_samples(half_extent: float, spacing: float) -> np.ndarray:
    count = max(2, int(math.ceil(2.0 * half_extent / spacing)) + 1)
    return np.linspace(-half_extent, half_extent, count)

def sample_box_surface_point_cloud(boxes: tuple[BoxObstacle, ...], spacing: float=0.012) -> np.ndarray:
    batches: list[np.ndarray] = []
    for box in boxes:
        center = np.asarray(box.center, dtype=float)
        half = np.asarray(box.half_size, dtype=float)
        coordinates = [_axis_samples(float(value), spacing) for value in half]
        for normal_axis in range(3):
            tangent_axes = [axis for axis in range(3) if axis != normal_axis]
            mesh = np.meshgrid(coordinates[tangent_axes[0]], coordinates[tangent_axes[1]], indexing='ij')
            for sign in (-1.0, 1.0):
                local = np.zeros((mesh[0].size, 3), dtype=float)
                local[:, normal_axis] = sign * half[normal_axis]
                local[:, tangent_axes[0]] = mesh[0].reshape(-1)
                local[:, tangent_axes[1]] = mesh[1].reshape(-1)
                batches.append(center + local)
    points = np.vstack(batches)
    return np.unique(np.round(points, decimals=9), axis=0)

def center_voxel_filter(points: np.ndarray, voxel_size: float=0.012) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points.reshape(0, 3)
    origin = np.min(points, axis=0) - 1e-09
    keys = np.floor((points - origin) / voxel_size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys = keys[order]
    ordered_points = points[order]
    selected: list[np.ndarray] = []
    start = 0
    while start < len(ordered_points):
        end = start + 1
        while end < len(ordered_points) and np.array_equal(keys[end], keys[start]):
            end += 1
        voxel_center = origin + (keys[start].astype(float) + 0.5) * voxel_size
        candidates = ordered_points[start:end]
        index = int(np.argmin(np.linalg.norm(candidates - voxel_center, axis=1)))
        selected.append(candidates[index])
        start = end
    return np.asarray(selected)

def _principal_frame(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / max(1, len(points) - 1)
    values, vectors = np.linalg.eigh(covariance)
    rotation = vectors[:, np.argsort(values)[::-1]]
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    return rotation

def estimate_point_normals(points: np.ndarray, neighbor_count: int=12) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) <= 2:
        normals = np.zeros_like(points)
        normals[:, 2] = 1.0
        return normals
    count = min(max(4, neighbor_count), len(points))
    tree = cKDTree(points)
    _, neighbors = tree.query(points, k=count, workers=NORMAL_ESTIMATION_WORKERS)
    local = points[np.asarray(neighbors, dtype=np.int64)]
    centered = local - np.mean(local, axis=1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', centered, centered)
    _, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    normal_norms = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(normal_norms[:, None], 1e-12)
    return normals

def _split_by_normal_similarity(indices: np.ndarray, points: np.ndarray, normals: np.ndarray, connection_distance: float, maximum_angle_degrees: float=25.0) -> list[np.ndarray]:
    cosine_threshold = float(np.cos(np.deg2rad(maximum_angle_degrees)))
    local_points = points[indices]
    parents = np.arange(len(indices), dtype=int)

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        anchor_similarity = abs(float(normals[indices[left_root]] @ normals[indices[right_root]]))
        if anchor_similarity < cosine_threshold:
            return
        if left_root < right_root:
            parents[right_root] = left_root
        else:
            parents[left_root] = right_root
    tree = cKDTree(local_points)
    for left, right in sorted(tree.query_pairs(connection_distance)):
        similarity = abs(float(normals[indices[left]] @ normals[indices[right]]))
        if similarity >= cosine_threshold:
            union(int(left), int(right))
    components: dict[int, list[int]] = {}
    for local_index, point_index in enumerate(indices):
        components.setdefault(find(local_index), []).append(int(point_index))
    return [np.asarray(group, dtype=int) for group in components.values()]

def fit_matched_voxel_proxies(raw_points: np.ndarray, filter_size: float=0.012, cluster_size: float=0.04, surface_uncertainty: float=0.003, maximum_aabb_overshoot: float=0.02, *, already_filtered: bool=False, filtered_point_offsets: np.ndarray | None=None, filtered_point_uncertainty_shapes: np.ndarray | None=None, normal_connection_distance: float | None=None, cluster_origin: np.ndarray | None=None, maximum_uncertainty_union_inflation: float | None=None, certificate_radius_limit: float | None=None, uncertainty_fusion_mode: str='separate_uncertainty') -> PointCloudProxySet:
    filtered = np.asarray(raw_points, dtype=float).reshape(-1, 3) if already_filtered else center_voxel_filter(raw_points, filter_size)
    if len(filtered) == 0:
        raise ValueError('Cannot fit proxies to an empty point cloud')
    if filtered_point_offsets is not None:
        if not already_filtered:
            raise ValueError('filtered_point_offsets require already_filtered=True')
        filtered_point_offsets = np.asarray(filtered_point_offsets, dtype=float).reshape(-1)
        if len(filtered_point_offsets) != len(filtered):
            raise ValueError('filtered_point_offsets must contain one radius per point')
    if filtered_point_uncertainty_shapes is not None:
        if not already_filtered:
            raise ValueError('filtered_point_uncertainty_shapes require already_filtered=True')
        filtered_point_uncertainty_shapes = np.asarray(filtered_point_uncertainty_shapes, dtype=float).reshape(-1, 3, 3)
        if len(filtered_point_uncertainty_shapes) != len(filtered):
            raise ValueError('filtered_point_uncertainty_shapes must contain one shape per point')
    if maximum_uncertainty_union_inflation is not None and maximum_uncertainty_union_inflation < 1.0:
        raise ValueError('maximum uncertainty union inflation must be at least one')
    if certificate_radius_limit is not None and certificate_radius_limit <= 0.0:
        raise ValueError('certificate radius limit must be positive')
    fused_uncertainty = uncertainty_fusion_mode == 'fused_certified_ellipsoid' and filtered_point_uncertainty_shapes is not None
    origin = np.min(filtered, axis=0) - 1e-09 if cluster_origin is None else np.asarray(cluster_origin, dtype=float).reshape(3)
    keys = np.floor((filtered - origin) / cluster_size).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    normals = estimate_point_normals(filtered)
    connection_distance = 1.75 * filter_size if normal_connection_distance is None else float(normal_connection_distance)
    if connection_distance <= 0.0:
        raise ValueError('normal_connection_distance must be positive')
    inverse_order = np.argsort(inverse, kind='stable')
    group_boundaries = np.flatnonzero(np.diff(inverse[inverse_order])) + 1
    spatial_groups = np.split(inverse_order, group_boundaries)
    initial_groups = [normal_group for spatial_group in spatial_groups for normal_group in _split_by_normal_similarity(spatial_group, filtered, normals, connection_distance=connection_distance)]

    def fit_group(indices: np.ndarray):
        cluster = filtered[indices]
        rotation = _principal_frame(cluster)
        mean = np.mean(cluster, axis=0)
        local = (cluster - mean) @ rotation
        local_min = np.min(local, axis=0)
        local_max = np.max(local, axis=0)
        local_center = 0.5 * (local_min + local_max)
        center = mean + rotation @ local_center
        half = 0.5 * (local_max - local_min)
        seed_axes = np.maximum(half, 0.0001)
        centered_local = local - local_center
        mahalanobis = np.sqrt(np.sum((centered_local / seed_axes[None, :]) ** 2, axis=1))
        axes = seed_axes * max(float(np.max(mahalanobis)), 1.0)
        shape = rotation @ np.diag(axes ** 2) @ rotation.T
        radius = float(np.max(np.linalg.norm(cluster - center, axis=1)))
        ellipsoid_world_support = np.sqrt(rotation ** 2 @ axes ** 2)
        point_support = np.max(np.abs(cluster - center), axis=0)
        overshoot = float(np.max(ellipsoid_world_support - point_support))
        return (center, radius, shape, rotation, overshoot)
    refined: list[tuple[np.ndarray, tuple]] = []

    def uncertainty_inflation(indices: np.ndarray) -> float:
        if filtered_point_uncertainty_shapes is None or fused_uncertainty:
            return 1.0
        members = filtered_point_uncertainty_shapes[indices]
        union = loewner_union_outer_shape(members)
        maximum_trace = max(float(np.max(np.trace(members, axis1=1, axis2=2))), 1e-18)
        return float(np.trace(union) / maximum_trace)

    def uncertainty_split(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if filtered_point_uncertainty_shapes is None or len(indices) < 2:
            return None
        members = filtered_point_uncertainty_shapes[indices]
        traces = np.maximum(np.trace(members, axis1=1, axis2=2), 1e-18)
        features = (members / traces[:, None, None]).reshape(len(indices), 9)
        first = 0
        second = int(np.argmax(np.linalg.norm(features - features[first], axis=1)))
        third = int(np.argmax(np.linalg.norm(features - features[second], axis=1)))
        left_distance = np.linalg.norm(features - features[second], axis=1)
        right_distance = np.linalg.norm(features - features[third], axis=1)
        left_mask = left_distance <= right_distance
        if np.all(left_mask) or not np.any(left_mask):
            return None
        return (indices[left_mask], indices[~left_mask])

    def certificate_requirements(indices: np.ndarray, fit: tuple | None=None) -> tuple[float, float]:
        _center, radius, shape, _rotation, _overshoot = fit_group(indices) if fit is None else fit
        offset = 0.0 if filtered_point_offsets is None else float(np.max(filtered_point_offsets[indices]))
        if filtered_point_uncertainty_shapes is None or fused_uncertainty:
            uncertainty_shape = np.zeros((3, 3), dtype=float)
            outer_shape = shape
        else:
            uncertainty_shape = loewner_union_outer_shape(filtered_point_uncertainty_shapes[indices])
            outer_shape = minkowski_outer_shape(shape, uncertainty_shape)
        uncertainty_radius = math.sqrt(max(float(np.linalg.eigvalsh(uncertainty_shape)[-1]), 0.0))
        sphere_required = radius + uncertainty_radius + offset
        ellipsoid_required = math.sqrt(max(float(np.linalg.eigvalsh(outer_shape)[-1]), 0.0)) + offset
        return (float(sphere_required), float(ellipsoid_required))

    def certificate_split(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if len(indices) < 2:
            return None
        cluster = filtered[indices]
        spread = np.ptp(cluster, axis=0)
        if float(np.max(spread)) <= 1e-15:
            split = uncertainty_split(indices)
            if split is not None:
                return split
            order = np.sort(indices)
        else:
            split_axis = int(np.argmax(spread))
            order = indices[np.argsort(cluster[:, split_axis], kind='stable')]
        midpoint = len(order) // 2
        if not 0 < midpoint < len(order):
            return None
        return (order[:midpoint], order[midpoint:])

    def refine(indices: np.ndarray, depth: int=0) -> None:
        fit = fit_group(indices)
        if maximum_uncertainty_union_inflation is not None and depth < 16 and (uncertainty_inflation(indices) > maximum_uncertainty_union_inflation):
            split = uncertainty_split(indices)
            if split is not None:
                refine(split[0], depth + 1)
                refine(split[1], depth + 1)
                return
        if certificate_radius_limit is not None:
            required = certificate_requirements(indices, fit)
            if max(required) > certificate_radius_limit + 1e-12:
                split = certificate_split(indices)
                if split is None or depth >= 32:
                    raise ValueError(f'certificate radius limit is smaller than an indivisible proxy requirement: limit={certificate_radius_limit:.9g}, sphere={required[0]:.9g}, ellipsoid={required[1]:.9g}')
                refine(split[0], depth + 1)
                refine(split[1], depth + 1)
                return
        if fit[-1] > maximum_aabb_overshoot and depth < 16 and (len(indices) >= 4):
            cluster = filtered[indices]
            split_axis = int(np.argmax(np.ptp(cluster, axis=0)))
            order = indices[np.argsort(cluster[:, split_axis])]
            midpoint = len(order) // 2
            if 0 < midpoint < len(order):
                refine(order[:midpoint], depth + 1)
                refine(order[midpoint:], depth + 1)
                return
        refined.append((indices, fit))
    for group in initial_groups:
        refine(group)
    centers: list[np.ndarray] = []
    radii: list[float] = []
    shapes: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    proxy_offsets: list[float] = []
    proxy_uncertainty_shapes: list[np.ndarray] = []
    base_radii: list[float] = []
    base_shapes: list[np.ndarray] = []
    outer_shapes: list[np.ndarray] = []
    refined_inverse = np.empty(len(filtered), dtype=int)
    maximum_overshoot = 0.0
    maximum_union_inflation = 1.0
    for proxy_index, (indices, _fit) in enumerate(refined):
        refined_inverse[indices] = proxy_index
    batched_proxy_uncertainty = None
    if filtered_point_uncertainty_shapes is not None and (not fused_uncertainty):
        batched_proxy_uncertainty = loewner_union_outer_shapes_grouped(filtered_point_uncertainty_shapes, refined_inverse, len(refined))
        member_traces = np.trace(filtered_point_uncertainty_shapes, axis1=1, axis2=2)
        maximum_member_traces = np.zeros(len(refined), dtype=float)
        np.maximum.at(maximum_member_traces, refined_inverse, member_traces)
        inflations = np.trace(batched_proxy_uncertainty, axis1=1, axis2=2) / np.maximum(maximum_member_traces, 1e-18)
        maximum_union_inflation = max(maximum_union_inflation, float(np.max(inflations)))
    for proxy_index, (indices, fit) in enumerate(refined):
        center, radius, shape, rotation, overshoot = fit
        maximum_overshoot = max(maximum_overshoot, overshoot)
        centers.append(center)
        base_radii.append(radius)
        base_shapes.append(shape)
        if filtered_point_uncertainty_shapes is None or fused_uncertainty:
            uncertainty_shape = np.zeros((3, 3))
            effective_radius = radius
            effective_shape = shape
        else:
            uncertainty_shape = batched_proxy_uncertainty[proxy_index]
            effective_radius = radius + math.sqrt(max(float(np.linalg.eigvalsh(uncertainty_shape)[-1]), 0.0))
            effective_shape = shape
        outer_shape = shape if filtered_point_uncertainty_shapes is None or fused_uncertainty else minkowski_outer_shape(shape, uncertainty_shape)
        proxy_offset = 0.0 if filtered_point_offsets is None else float(np.max(filtered_point_offsets[indices]))
        if certificate_radius_limit is not None:
            sphere_required = effective_radius + proxy_offset
            ellipsoid_required = math.sqrt(max(float(np.linalg.eigvalsh(outer_shape)[-1]), 0.0)) + proxy_offset
            if sphere_required > certificate_radius_limit + 1e-09 or ellipsoid_required > certificate_radius_limit + 1e-09:
                raise AssertionError('refined proxy exceeded certificate radius limit')
            effective_radius = certificate_radius_limit - proxy_offset
        radii.append(effective_radius)
        shapes.append(effective_shape)
        rotations.append(rotation)
        proxy_uncertainty_shapes.append(uncertainty_shape)
        outer_shapes.append(outer_shape)
        proxy_offsets.append(proxy_offset)
    result = PointCloudProxySet(raw_points=np.asarray(raw_points, dtype=float), filtered_points=filtered, centers=np.asarray(centers), sphere_radii=np.asarray(radii), ellipsoid_shapes=np.asarray(shapes), rotations=np.asarray(rotations), cluster_keys=np.arange(len(refined), dtype=int)[:, None], filtered_cluster_indices=refined_inverse, maximum_ellipsoid_overshoot=float(maximum_overshoot), proxy_offset_radii=None if filtered_point_offsets is None and filtered_point_uncertainty_shapes is None else np.asarray(proxy_offsets, dtype=float), filtered_uncertainty_shapes=None if filtered_point_uncertainty_shapes is None else filtered_point_uncertainty_shapes.copy(), proxy_uncertainty_shapes=None if filtered_point_uncertainty_shapes is None else np.asarray(proxy_uncertainty_shapes), base_sphere_radii=np.asarray(base_radii), base_ellipsoid_shapes=np.asarray(base_shapes), ellipsoid_outer_shapes=np.asarray(outer_shapes), surface_cover_radius=0.0 if filtered_point_uncertainty_shapes is None else float(np.sqrt(max(np.max(np.linalg.eigvalsh(np.asarray(filtered_point_uncertainty_shapes))[:, -1]), 0.0))), maximum_uncertainty_union_inflation=float(maximum_union_inflation), certificate_radius_limit=None if certificate_radius_limit is None else float(certificate_radius_limit), uncertainty_fusion_mode=uncertainty_fusion_mode)
    assert_proxy_coverage(result, surface_uncertainty)
    return result

def assert_proxy_coverage(proxies: PointCloudProxySet, surface_uncertainty: float=0.003) -> None:
    """Verify every filtered point lies in its matched sphere and ellipsoid."""
    for point, proxy_index in zip(proxies.filtered_points, proxies.filtered_cluster_indices):
        center = proxies.centers[proxy_index]
        sphere_slack = np.linalg.norm(point - center) - proxies.sphere_radii[proxy_index]
        if float(sphere_slack) > 1e-09:
            raise AssertionError('Filtered point is not covered by its sphere proxy')
        delta = point - center
        shape = proxies.ellipsoid_shapes[proxy_index]
        value = float(delta @ np.linalg.solve(shape, delta))
        if value > 1.0 + 1e-08:
            raise AssertionError('Filtered point is not covered by its ellipsoid proxy')
