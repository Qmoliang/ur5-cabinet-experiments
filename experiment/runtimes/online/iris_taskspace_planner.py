"""Low-frequency point-cloud planner and IRIS-style separating-plane corridor.

The planner consumes only matched obstacle proxies.  A* supplies collision-free
seeds; obstacle support planes then certify local convex task-space regions.
The regions are not hand-authored waypoints and are never read from EP17 data.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np
from scipy.optimize import minimize

from ellipsoid_model import (
    closest_point_on_ellipsoid,
    optimal_support_separating_normal,
    support_radius,
)
from multilevel_voxel_table import MultilevelVoxelTable
from native_ellipsoid_support import NativeEllipsoidSupport


@dataclass(frozen=True)
class ConvexRegion:
    seed: np.ndarray
    center: np.ndarray
    shape: np.ndarray
    A: np.ndarray
    b: np.ndarray
    covered_path_start: int
    covered_path_end: int


@dataclass(frozen=True)
class IrisTaskspacePlan:
    grid_path: np.ndarray
    smoothed_path: np.ndarray
    control_targets: np.ndarray
    regions: tuple[ConvexRegion, ...]
    expanded_nodes: int
    resolution: float
    minimum_path_clearance: float


class EllipsoidToolFreeSpace:
    def __init__(
        self,
        centers: np.ndarray,
        shapes: np.ndarray,
        offsets: np.ndarray,
        tool_radius: float,
        safety_margin: float,
        tool_shapes: np.ndarray | None = None,
        tool_offsets: np.ndarray | None = None,
        obstacle_uncertainty_shapes: np.ndarray | None = None,
        obstacle_pruning_shapes: np.ndarray | None = None,
    ) -> None:
        self.centers = np.asarray(centers, dtype=float)
        self.shapes = np.asarray(shapes, dtype=float)
        self.uncertainty_shapes = (
            np.zeros_like(self.shapes)
            if obstacle_uncertainty_shapes is None
            else np.asarray(obstacle_uncertainty_shapes, dtype=float).reshape(-1, 3, 3)
        )
        self.pruning_shapes = (
            self.shapes
            if obstacle_pruning_shapes is None
            else np.asarray(obstacle_pruning_shapes, dtype=float).reshape(-1, 3, 3)
        )
        self.offsets = np.asarray(offsets, dtype=float)
        self.tool_radius = float(tool_radius)
        self.safety_margin = float(safety_margin)
        if tool_shapes is None:
            self.exact_sphere_tool = True
            self.tool_shapes = np.eye(3)[None, :, :] * self.tool_radius**2
            self.tool_offsets = np.zeros((1, 3))
        else:
            self.exact_sphere_tool = False
            self.tool_shapes = np.asarray(tool_shapes, dtype=float).reshape(-1, 3, 3)
            self.tool_offsets = np.asarray(tool_offsets, dtype=float).reshape(-1, 3)
            if len(self.tool_shapes) != len(self.tool_offsets):
                raise ValueError("tool_shapes and tool_offsets must have equal length")
        self.native_support = NativeEllipsoidSupport()
        maximum_axis = float(np.sqrt(np.max(np.linalg.eigvalsh(self.pruning_shapes))))
        self.table = MultilevelVoxelTable.from_ellipsoids(
            self.centers,
            self.pruning_shapes,
            voxel_size=max(0.02, self.tool_radius + self.safety_margin),
            query_padding=maximum_axis + float(np.max(self.offsets)) + 0.02,
            offset_radii=self.offsets,
        )

    def clearance(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=float)
        best = np.inf
        for tool_offset, tool_shape in zip(self.tool_offsets, self.tool_shapes):
            tool_center = point + tool_offset
            candidates = self.table.query_ellipsoid(tool_center, tool_shape)
            if not len(candidates):
                continue
            if self.exact_sphere_tool and not np.any(
                self.uncertainty_shapes[candidates]
            ):
                for index in candidates:
                    closest = closest_point_on_ellipsoid(
                        self.centers[index], self.shapes[index], tool_center
                    )
                    best = min(
                        best,
                        float(np.linalg.norm(closest.surface_point - tool_center))
                        - self.tool_radius
                        - self.offsets[index]
                        - self.safety_margin,
                    )
                continue
            uncertainty = self.uncertainty_shapes[candidates]
            if np.any(uncertainty):
                normals, _ = self.native_support.normals_sum(
                    tool_center,
                    tool_shape,
                    self.centers[candidates],
                    self.shapes[candidates],
                    uncertainty,
                )
            else:
                normals, _ = self.native_support.normals(
                    tool_center,
                    tool_shape,
                    self.centers[candidates],
                    self.shapes[candidates],
                )
            tool_shape_normals = normals @ tool_shape.T
            tool_extents = np.sqrt(
                np.maximum(
                    np.einsum("ni,ni->n", normals, tool_shape_normals), 0.0
                )
            )
            obstacle_shape_normals = np.einsum(
                "nij,nj->ni", self.shapes[candidates], normals
            )
            obstacle_extents = np.sqrt(
                np.maximum(
                    np.einsum("ni,ni->n", normals, obstacle_shape_normals),
                    0.0,
                )
            )
            uncertainty_shape_normals = np.einsum(
                "nij,nj->ni", self.uncertainty_shapes[candidates], normals
            )
            obstacle_extents += np.sqrt(
                np.maximum(
                    np.einsum(
                        "ni,ni->n", normals, uncertainty_shape_normals
                    ),
                    0.0,
                )
            )
            values = (
                np.einsum(
                    "ni,ni->n",
                    normals,
                    self.centers[candidates] - tool_center,
                )
                - tool_extents
                - obstacle_extents
                - self.offsets[candidates]
                - self.safety_margin
            )
            best = min(best, float(np.min(values)))
        return float(best)

    def segment_is_free(self, first: np.ndarray, second: np.ndarray, step: float) -> bool:
        distance = float(np.linalg.norm(second - first))
        count = max(2, int(math.ceil(distance / step)) + 1)
        return all(
            self.clearance(point) >= 0.0
            for point in np.linspace(first, second, count)
        )


def _astar(
    free_space: EllipsoidToolFreeSpace,
    start: np.ndarray,
    goal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    resolution: float,
) -> tuple[np.ndarray, int]:
    dimensions = np.floor((upper - lower) / resolution).astype(int) + 1

    def index_of(point: np.ndarray) -> tuple[int, int, int]:
        index = np.rint((point - lower) / resolution).astype(int)
        index = np.clip(index, 0, dimensions - 1)
        return tuple(int(value) for value in index)

    def point_of(index: tuple[int, int, int]) -> np.ndarray:
        return lower + resolution * np.asarray(index, dtype=float)

    start_index = index_of(start)
    goal_index = index_of(goal)
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    queue: list[tuple[float, float, tuple[int, int, int]]] = []
    heapq.heappush(queue, (0.0, 0.0, start_index))
    costs = {start_index: 0.0}
    parents: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    safe_cache: dict[tuple[int, int, int], bool] = {}
    expanded = 0
    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current_cost > costs.get(current, np.inf) + 1.0e-12:
            continue
        expanded += 1
        if current == goal_index:
            break
        for delta in offsets:
            neighbor = tuple(current[axis] + delta[axis] for axis in range(3))
            if any(neighbor[axis] < 0 or neighbor[axis] >= dimensions[axis] for axis in range(3)):
                continue
            if neighbor not in safe_cache:
                safe_cache[neighbor] = free_space.clearance(point_of(neighbor)) >= 0.0
            if not safe_cache[neighbor]:
                continue
            step_cost = resolution * float(np.linalg.norm(delta))
            candidate = current_cost + step_cost
            if candidate + 1.0e-12 >= costs.get(neighbor, np.inf):
                continue
            costs[neighbor] = candidate
            parents[neighbor] = current
            heuristic = float(np.linalg.norm(point_of(neighbor) - point_of(goal_index)))
            heapq.heappush(queue, (candidate + heuristic, candidate, neighbor))
    if goal_index not in costs:
        raise RuntimeError("ellipsoid free-space A* found no path")
    indices = [goal_index]
    while indices[-1] != start_index:
        indices.append(parents[indices[-1]])
    indices.reverse()
    points = np.asarray([point_of(index) for index in indices])
    points[0] = start
    points[-1] = goal
    return points, expanded


def _shortcut_path(
    path: np.ndarray, free_space: EllipsoidToolFreeSpace, check_step: float
) -> np.ndarray:
    selected = [path[0]]
    current = 0
    while current < len(path) - 1:
        furthest = current + 1
        for candidate in range(len(path) - 1, current, -1):
            if free_space.segment_is_free(path[current], path[candidate], check_step):
                furthest = candidate
                break
        selected.append(path[furthest])
        current = furthest
    return np.asarray(selected)


def _resample_path(path: np.ndarray, maximum_step: float) -> np.ndarray:
    result = [path[0]]
    for first, second in zip(path[:-1], path[1:]):
        count = max(1, int(math.ceil(float(np.linalg.norm(second - first)) / maximum_step)))
        result.extend(np.linspace(first, second, count + 1)[1:])
    return np.asarray(result)


def _decode_mvie(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = parameters[:3]
    lower = np.array(
        [
            [math.exp(parameters[3]), 0.0, 0.0],
            [parameters[4], math.exp(parameters[5]), 0.0],
            [parameters[6], parameters[7], math.exp(parameters[8])],
        ]
    )
    return center, lower


def _maximum_volume_inscribed_ellipsoid(
    A: np.ndarray, b: np.ndarray, seed: np.ndarray, initial_radius: float
) -> tuple[np.ndarray, np.ndarray]:
    initial = np.r_[seed, math.log(initial_radius), 0.0, math.log(initial_radius), 0.0, 0.0, math.log(initial_radius)]

    def objective(parameters: np.ndarray) -> float:
        return -float(parameters[3] + parameters[5] + parameters[8])

    def constraints(parameters: np.ndarray) -> np.ndarray:
        center, lower = _decode_mvie(parameters)
        return b - A @ center - np.linalg.norm(A @ lower, axis=1)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        constraints={"type": "ineq", "fun": constraints},
        options={"maxiter": 180, "ftol": 1.0e-9, "disp": False},
    )
    if not result.success or float(np.min(constraints(result.x))) < -1.0e-6:
        return seed.copy(), np.eye(3) * initial_radius**2
    center, lower = _decode_mvie(result.x)
    return center, lower @ lower.T


def _optimal_minkowski_support_normal(
    delta: np.ndarray,
    region_shape: np.ndarray,
    tool_shape: np.ndarray,
    obstacle_shape: np.ndarray,
    max_iterations: int = 24,
) -> np.ndarray:
    """IRIS normal for region + reflected tool versus obstacle support."""

    delta = np.asarray(delta, dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0])
    normal = delta / distance
    shapes = (region_shape, tool_shape, obstacle_shape)

    def objective(direction: np.ndarray) -> float:
        return float(
            direction @ delta
            - sum(support_radius(item, direction) for item in shapes)
        )

    value = objective(normal)
    for _ in range(max_iterations):
        gradient = delta.copy()
        for item in shapes:
            extent = max(support_radius(item, normal), 1.0e-12)
            gradient -= item @ normal / extent
        tangent = gradient - normal * float(normal @ gradient)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1.0e-10:
            break
        direction = tangent / tangent_norm
        improved = False
        for step in (0.5, 0.25, 0.125, 0.0625, 0.03125):
            candidate = normal + step * direction
            candidate /= np.linalg.norm(candidate)
            candidate_value = objective(candidate)
            if candidate_value > value + 1.0e-12:
                normal = candidate
                value = candidate_value
                improved = True
                break
        if not improved:
            break
    return normal


def inflate_iris_region(
    seed: np.ndarray,
    free_space: EllipsoidToolFreeSpace,
    lower: np.ndarray,
    upper: np.ndarray,
    iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Alternate IRIS support-plane separation and 3-D MVIE inflation."""

    center = np.asarray(seed, dtype=float).copy()
    shape = np.eye(3) * 0.004**2
    for _ in range(iterations):
        rows = []
        bounds = []
        for tool_offset, tool_shape in zip(
            free_space.tool_offsets, free_space.tool_shapes
        ):
            tool_center = center + tool_offset
            normals, _ = free_space.native_support.normals(
                tool_center,
                tool_shape,
                free_space.centers,
                free_space.shapes,
            )
            obstacle_extents = np.sqrt(
                np.maximum(
                    np.einsum(
                        "ni,nij,nj->n",
                        normals,
                        free_space.shapes,
                        normals,
                    ),
                    0.0,
                )
            )
            uncertainty_extents = np.sqrt(
                np.maximum(
                    np.einsum(
                        "ni,nij,nj->n",
                        normals,
                        free_space.uncertainty_shapes,
                        normals,
                    ),
                    0.0,
                )
            )
            obstacle_extents += uncertainty_extents
            tool_extents = np.sqrt(
                np.maximum(
                    np.einsum("ni,ij,nj->n", normals, tool_shape, normals),
                    0.0,
                )
            )
            candidate_bounds = (
                np.einsum("ni,ni->n", normals, free_space.centers)
                - obstacle_extents
                - free_space.offsets
                - tool_extents
                - normals @ tool_offset
                - free_space.safety_margin
            )
            safe = normals @ seed <= candidate_bounds + 1.0e-7
            safe_indices = np.flatnonzero(safe)
            if len(safe_indices):
                tangent_bounds = (
                    np.einsum("ni,ni->n", normals, free_space.centers)
                    - obstacle_extents
                    - free_space.offsets
                )
                clearances = (
                    candidate_bounds[safe_indices]
                    - normals[safe_indices] @ seed
                )
                order = safe_indices[np.argsort(clearances)]
                active_local = free_space.native_support.prune_planes(
                    free_space.centers[order],
                    free_space.pruning_shapes[order],
                    free_space.offsets[order],
                    normals[order],
                    tangent_bounds[order],
                )
                active = order[active_local]
            else:
                active = np.zeros(0, dtype=int)
            for normal, bound in zip(normals[active], candidate_bounds[active]):
                bound = (
                    float(bound)
                )
                rows.append(normal)
                bounds.append(bound)
        for axis in range(3):
            positive = np.zeros(3)
            positive[axis] = 1.0
            rows.extend((positive, -positive))
            bounds.extend((float(upper[axis]), float(-lower[axis])))
        A = np.asarray(rows, dtype=float)
        b = np.asarray(bounds, dtype=float)
        center, shape = _maximum_volume_inscribed_ellipsoid(A, b, seed, 0.003)
    return center, shape, A, b


def plan_iris_taskspace_corridor(
    start: np.ndarray,
    goal: np.ndarray,
    centers: np.ndarray,
    shapes: np.ndarray,
    offsets: np.ndarray,
    tool_radius: float,
    safety_margin: float = 0.006,
    resolution: float = 0.010,
    maximum_control_step: float = 0.020,
    tool_shapes: np.ndarray | None = None,
    tool_offsets: np.ndarray | None = None,
    obstacle_uncertainty_shapes: np.ndarray | None = None,
    obstacle_pruning_shapes: np.ndarray | None = None,
) -> IrisTaskspacePlan:
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    lower = np.minimum(start, goal) - np.array([0.02, 0.085, 0.085])
    upper = np.maximum(start, goal) + np.array([0.02, 0.085, 0.085])
    free_space = EllipsoidToolFreeSpace(
        centers,
        shapes,
        offsets,
        tool_radius,
        safety_margin,
        tool_shapes=tool_shapes,
        tool_offsets=tool_offsets,
        obstacle_uncertainty_shapes=obstacle_uncertainty_shapes,
        obstacle_pruning_shapes=obstacle_pruning_shapes,
    )
    grid_path, expanded = _astar(
        free_space, start, goal, lower, upper, resolution
    )
    smoothed = _shortcut_path(grid_path, free_space, resolution * 0.45)
    targets = _resample_path(smoothed, maximum_control_step)
    regions = []
    path_index = 0
    while path_index < len(targets):
        center, shape, A, b = inflate_iris_region(
            targets[path_index], free_space, lower, upper
        )
        inside = A @ targets.T <= b[:, None] + 1.0e-8
        end = path_index
        while end + 1 < len(targets) and bool(np.all(inside[:, end + 1])):
            end += 1
        regions.append(
            ConvexRegion(
                seed=targets[path_index].copy(),
                center=center,
                shape=shape,
                A=A,
                b=b,
                covered_path_start=path_index,
                covered_path_end=end,
            )
        )
        path_index = max(path_index + 1, end)
    clearances = [free_space.clearance(point) for point in targets]
    return IrisTaskspacePlan(
        grid_path=grid_path,
        smoothed_path=smoothed,
        control_targets=targets,
        regions=tuple(regions),
        expanded_nodes=expanded,
        resolution=resolution,
        minimum_path_clearance=float(min(clearances)),
    )
