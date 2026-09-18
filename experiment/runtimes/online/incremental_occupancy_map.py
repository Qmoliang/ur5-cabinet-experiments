"""Sparse causal free/occupied/unknown map for wrist-depth LiuQP.

Only rays from observations received so far are integrated.  Robot-hit pixels
must already have been removed *after* depth rendering, so their shadows remain
unknown.  The map is deliberately simple and auditable: integer log-odds-like
scores in a sparse voxel dictionary plus conservative occupied endpoint balls.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from depth_camera_perception import DepthObservation, environment_endpoint_mask


UNKNOWN = 0
FREE = 1
OCCUPIED = 2


@dataclass(frozen=True)
class MapUpdateStats:
    frame_index: int
    rays: int
    free_updates: int
    occupied_updates: int
    free_voxels: int
    occupied_voxels: int
    unknown_queries: int


@dataclass(frozen=True)
class SweptVolumeCertificate:
    safe: bool
    occupied_voxels: int
    unknown_voxels: int
    checked_voxels: int


class IncrementalOccupancyMap:
    """Three-state sparse voxel map with hard unknown-space certification."""

    def __init__(
        self,
        voxel_size: float = 0.015,
        *,
        free_decrement: int = 1,
        occupied_increment: int = 4,
        minimum_score: int = -8,
        maximum_score: int = 8,
        occupied_threshold: int = 2,
        free_threshold: int = -1,
    ) -> None:
        if voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")
        self.voxel_size = float(voxel_size)
        self.free_decrement = int(free_decrement)
        self.occupied_increment = int(occupied_increment)
        self.minimum_score = int(minimum_score)
        self.maximum_score = int(maximum_score)
        self.occupied_threshold = int(occupied_threshold)
        self.free_threshold = int(free_threshold)
        self._scores: dict[tuple[int, int, int], int] = {}
        self._frame_index = -1
        self._unknown_queries = 0
        self._surface_points: list[np.ndarray] = []
        self._surface_radii: list[np.ndarray] = []
        self._ball_offset_cache: dict[int, np.ndarray] = {}

    def key(self, point: np.ndarray) -> tuple[int, int, int]:
        value = np.floor(np.asarray(point, dtype=float) / self.voxel_size).astype(int)
        return int(value[0]), int(value[1]), int(value[2])

    def center(self, key: tuple[int, int, int]) -> np.ndarray:
        return (np.asarray(key, dtype=float) + 0.5) * self.voxel_size

    def state_key(self, key: tuple[int, int, int]) -> int:
        score = self._scores.get(key)
        if score is None:
            return UNKNOWN
        if score >= self.occupied_threshold:
            return OCCUPIED
        if score <= self.free_threshold:
            return FREE
        return UNKNOWN

    def state(self, point: np.ndarray) -> int:
        return self.state_key(self.key(point))

    def _add_score(self, key: tuple[int, int, int], delta: int) -> None:
        score = self._scores.get(key, 0) + int(delta)
        self._scores[key] = min(self.maximum_score, max(self.minimum_score, score))

    def _ray_free_keys(
        self,
        origin: np.ndarray,
        endpoint: np.ndarray,
        endpoint_radius: float,
    ) -> list[tuple[int, int, int]]:
        delta = np.asarray(endpoint, dtype=float) - np.asarray(origin, dtype=float)
        distance = float(np.linalg.norm(delta))
        free_length = max(0.0, distance - float(endpoint_radius) - 0.5 * self.voxel_size)
        if free_length <= 0.0 or distance <= 1.0e-12:
            return []
        direction = delta / distance
        count = max(1, int(math.ceil(free_length / (0.45 * self.voxel_size))))
        samples = np.asarray(origin)[None, :] + np.linspace(
            0.0, free_length, count, endpoint=False
        )[:, None] * direction[None, :]
        keys = np.floor(samples / self.voxel_size).astype(np.int64)
        if len(keys) > 1:
            keep = np.ones(len(keys), dtype=bool)
            keep[1:] = np.any(keys[1:] != keys[:-1], axis=1)
            keys = keys[keep]
        return [(int(k[0]), int(k[1]), int(k[2])) for k in keys]

    def _ball_key_array(self, center: np.ndarray, radius: float) -> np.ndarray:
        radius = max(0.0, float(radius))
        half_diagonal = math.sqrt(3.0) * 0.5 * self.voxel_size
        center = np.asarray(center, dtype=float)
        base = np.floor(center / self.voxel_size).astype(np.int64)
        reach = int(
            math.ceil((radius + half_diagonal) / self.voxel_size)
        ) + 1
        offsets = self._ball_offset_cache.get(reach)
        if offsets is None:
            axis = np.arange(-reach, reach + 1, dtype=np.int64)
            offsets = np.stack(
                np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
            ).reshape(-1, 3)
            self._ball_offset_cache[reach] = offsets
        keys = base[None, :] + offsets
        voxel_centers = (keys.astype(float) + 0.5) * self.voxel_size
        keep = (
            np.linalg.norm(voxel_centers - center[None, :], axis=1)
            <= radius + half_diagonal
        )
        return keys[keep]

    def _ball_keys(self, center: np.ndarray, radius: float) -> list[tuple[int, int, int]]:
        return [
            (int(key[0]), int(key[1]), int(key[2]))
            for key in self._ball_key_array(center, radius)
        ]

    def integrate(self, observations: Iterable[DepthObservation]) -> MapUpdateStats:
        observations = list(observations)
        self._frame_index += 1
        free_updates = 0
        occupied_updates = 0
        rays = 0
        for observation in observations:
            if len(observation.points) != len(observation.sample_radii):
                raise ValueError("each depth point needs one conservative radius")
            endpoint_mask = environment_endpoint_mask(observation)
            if np.any(endpoint_mask):
                self._surface_points.append(
                    np.asarray(observation.points[endpoint_mask], dtype=float).copy()
                )
                self._surface_radii.append(
                    np.asarray(
                        observation.sample_radii[endpoint_mask], dtype=float
                    ).copy()
                )
            for endpoint, radius, endpoint_occupied in zip(
                observation.points,
                observation.sample_radii,
                endpoint_mask,
            ):
                rays += 1
                for key in self._ray_free_keys(
                    observation.camera_position, endpoint, float(radius)
                ):
                    self._add_score(key, -self.free_decrement)
                    free_updates += 1
                if endpoint_occupied:
                    for key in self._ball_keys(endpoint, float(radius)):
                        self._add_score(key, self.occupied_increment)
                        occupied_updates += 1
        counts = self.state_counts()
        return MapUpdateStats(
            frame_index=self._frame_index,
            rays=rays,
            free_updates=free_updates,
            occupied_updates=occupied_updates,
            free_voxels=counts[FREE],
            occupied_voxels=counts[OCCUPIED],
            unknown_queries=self._unknown_queries,
        )

    def mark_current_robot_free(
        self, centers: np.ndarray, radii: np.ndarray, padding: float = 0.0
    ) -> None:
        """Declare only the robot's currently occupied, already-safe volume free."""

        for center, radius in zip(np.asarray(centers), np.asarray(radii)):
            for key in self._ball_keys(center, float(radius) + float(padding)):
                # Never erase an observed occupied endpoint.
                if self.state_key(key) != OCCUPIED:
                    self._scores[key] = self.minimum_score

    def certify_swept_spheres(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        radii: np.ndarray,
        *,
        margin: float = 0.0,
    ) -> SweptVolumeCertificate:
        occupied = 0
        unknown = 0
        checked: set[tuple[int, int, int]] = set()
        for start, end, radius in zip(np.asarray(starts), np.asarray(ends), np.asarray(radii)):
            distance = float(np.linalg.norm(end - start))
            steps = max(1, int(math.ceil(distance / (0.45 * self.voxel_size))))
            for alpha in np.linspace(0.0, 1.0, steps + 1):
                center = (1.0 - alpha) * start + alpha * end
                for raw_key in self._ball_key_array(
                    center, float(radius) + float(margin)
                ):
                    key = (
                        int(raw_key[0]),
                        int(raw_key[1]),
                        int(raw_key[2]),
                    )
                    if key in checked:
                        continue
                    checked.add(key)
                    state = self.state_key(key)
                    if state == OCCUPIED:
                        occupied += 1
                    elif state == UNKNOWN:
                        unknown += 1
        self._unknown_queries += unknown
        return SweptVolumeCertificate(
            safe=(occupied == 0 and unknown == 0),
            occupied_voxels=occupied,
            unknown_voxels=unknown,
            checked_voxels=len(checked),
        )

    def state_counts(self) -> dict[int, int]:
        counts = {UNKNOWN: 0, FREE: 0, OCCUPIED: 0}
        for key in self._scores:
            counts[self.state_key(key)] += 1
        return counts

    def surface_samples(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._surface_points:
            return np.zeros((0, 3)), np.zeros(0)
        return np.vstack(self._surface_points), np.concatenate(self._surface_radii)

    def snapshot_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        keys = np.asarray(list(self._scores), dtype=np.int32).reshape(-1, 3)
        states = np.fromiter((self.state_key(tuple(map(int, k))) for k in keys), dtype=np.int8)
        return keys, states

    def clone_for_query(self) -> "IncrementalOccupancyMap":
        """Publish an immutable-by-convention causal snapshot to control.

        The worker continues updating its own dictionary; the controller sees
        only this copy and therefore cannot observe a partially integrated
        frame. Surface points are deliberately omitted because the controller
        needs only three-state sweep queries.
        """

        clone = IncrementalOccupancyMap(
            self.voxel_size,
            free_decrement=self.free_decrement,
            occupied_increment=self.occupied_increment,
            minimum_score=self.minimum_score,
            maximum_score=self.maximum_score,
            occupied_threshold=self.occupied_threshold,
            free_threshold=self.free_threshold,
        )
        clone._scores = self._scores.copy()
        clone._frame_index = self._frame_index
        return clone
