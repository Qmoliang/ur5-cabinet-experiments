"""Extracted original v4.3 ellipsoid dependency; see docs/source_manifest.json."""
from __future__ import annotations
import numpy as np
from camera import DepthObservation, environment_endpoint_mask
from voxel_index import NativeMultilevelMVT
from scene import CONTACT_DISTANCE, NEAR_DISTANCE, SAFETY_MARGIN
from controller import ProtocolLiuQPController
MAP_VOXEL_SIZE = 0.005
MAXIMUM_AABB_OVERSHOOT = 0.025
PROXY_CLUSTER_SIZE = 0.10
MVT_BASE_VOXEL_SIZE = 0.015
MVT_NUMERICAL_OUTWARD_PADDING = 5e-06
MOTION_BROADPHASE_PADDING = 0.0
PROXY_WORKSPACE_LOWER = np.array([-0.25, -0.15, -0.02])
PROXY_WORKSPACE_UPPER = np.array([0.95, 0.95, 1.15])

def _crop_proxy_observation(observation: DepthObservation) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(observation.points, dtype=float)
    keep = environment_endpoint_mask(observation)
    keep &= np.all(points >= PROXY_WORKSPACE_LOWER, axis=1)
    keep &= np.all(points <= PROXY_WORKSPACE_UPPER, axis=1)
    return (points[keep], np.asarray(observation.sample_radii, dtype=float)[keep])

def _environment_half_extents(proxies, representation: str) -> np.ndarray:
    offsets = np.asarray(proxies.proxy_offset_radii, dtype=float)
    if representation == 'sphere':
        radii = np.asarray(proxies.sphere_radii, dtype=float) + offsets
        return np.repeat(radii[:, None], 3, axis=1)
    shapes = proxies.ellipsoid_outer_shapes if proxies.proxy_uncertainty_shapes is not None else proxies.base_ellipsoid_shapes
    return np.sqrt(np.maximum(np.diagonal(np.asarray(shapes), axis1=1, axis2=2), 0.0)) + offsets[:, None]

def _build_mvt_only(proxies, representation: str, robot_radii: np.ndarray, *, simd: bool) -> NativeMultilevelMVT:
    half = _environment_half_extents(proxies, representation)
    indexed_half = half + MVT_NUMERICAL_OUTWARD_PADDING
    query_padding = NEAR_DISTANCE + SAFETY_MARGIN + MOTION_BROADPHASE_PADDING + MVT_NUMERICAL_OUTWARD_PADDING
    maximum_query_half = float(np.max(robot_radii) + query_padding)
    table = NativeMultilevelMVT(proxies.centers, indexed_half, base_voxel_size=MVT_BASE_VOXEL_SIZE, maximum_query_half_extent=maximum_query_half, query_padding=query_padding, simd=simd)
    return table

def _build_mvt_and_audit(proxies, representation: str, robot_positions: np.ndarray, robot_radii: np.ndarray, *, simd: bool) -> tuple[NativeMultilevelMVT, int]:
    half = _environment_half_extents(proxies, representation)
    query_padding = NEAR_DISTANCE + SAFETY_MARGIN + MOTION_BROADPHASE_PADDING + MVT_NUMERICAL_OUTWARD_PADDING
    table = _build_mvt_only(proxies, representation, robot_radii, simd=simd)
    missing = 0
    actual_rows = table.query_spheres(robot_positions, robot_radii)
    lo = proxies.centers - half
    hi = proxies.centers + half
    for center, radius, actual in zip(robot_positions, robot_radii, actual_rows):
        query_radius = float(radius) + query_padding
        delta = np.maximum(np.maximum(lo - center, center - hi), 0.0)
        expected = np.flatnonzero(np.einsum('ij,ij->i', delta, delta) <= query_radius * query_radius)
        missing += len(np.setdiff1d(expected, actual, assume_unique=True))
    if missing:
        table.close()
        raise AssertionError(f'multilevel MVT missed {missing} ball-AABB oracle candidates')
    return (table, missing)

def _controller_uncertainty_shapes(proxies) -> np.ndarray | None:
    shapes = proxies.proxy_uncertainty_shapes
    if shapes is None:
        return None
    shapes = np.asarray(shapes, dtype=float)
    if not np.any(shapes):
        return None
    return shapes

def _new_controller(representation: str, model, data, scene, robot, proxies, obstacle_index, *, ellipsoid_pair_threads: int=8, ellipsoid_pair_affinity_mask: int=0) -> ProtocolLiuQPController:
    common = dict(model=model, data=data, scene=scene, robot_spheres=robot, obstacle_centers=proxies.centers, representation=representation, obstacle_offsets=proxies.proxy_offset_radii, proxy_ids=proxies.proxy_ids, safety_margin=SAFETY_MARGIN, near_distance=NEAR_DISTANCE, contact_distance=CONTACT_DISTANCE, obstacle_index=obstacle_index, redundant_plane_pruning=True, ellipsoid_pair_threads=ellipsoid_pair_threads, ellipsoid_pair_affinity_mask=ellipsoid_pair_affinity_mask)
    if representation == 'sphere':
        return ProtocolLiuQPController(**common, obstacle_radii=proxies.sphere_radii)
    return ProtocolLiuQPController(**common, obstacle_shapes=proxies.base_ellipsoid_shapes, obstacle_uncertainty_shapes=_controller_uncertainty_shapes(proxies))
