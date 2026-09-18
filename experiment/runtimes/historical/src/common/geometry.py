"""Extracted original v4.3 ellipsoid dependency; see docs/source_manifest.json."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class EllipsoidClosestPointResult:
    normal: np.ndarray
    surface_point: np.ndarray
    multiplier: float
    newton_iterations: int
    bisection_iterations: int
    residual: float

@dataclass(frozen=True)
class SupportNormalResult:
    normal: np.ndarray
    obstacle_surface_point: np.ndarray
    clearance_without_robot: float
    iterations: int
    residual: float

def support_radius(shape_matrix: np.ndarray, normal: np.ndarray) -> float:
    value = float(normal @ shape_matrix @ normal)
    return float(np.sqrt(max(value, 0.0)))

def support_angular_gradient(shape_matrix: np.ndarray, normal: np.ndarray) -> np.ndarray:
    radius = support_radius(shape_matrix, normal)
    if radius <= 1e-12:
        return np.zeros(3)
    return np.cross(shape_matrix @ normal, normal) / radius

def closest_point_on_ellipsoid(center: np.ndarray, shape_matrix: np.ndarray, point: np.ndarray, *, initial_multiplier: float | None=None, eigenvalues: np.ndarray | None=None, rotation: np.ndarray | None=None, maximum_newton_iterations: int=10, maximum_bisection_iterations: int=48, tolerance: float=1e-12) -> EllipsoidClosestPointResult:
    if eigenvalues is None or rotation is None:
        eigenvalues, rotation = np.linalg.eigh(np.asarray(shape_matrix, dtype=float))
    eigenvalues = np.maximum(np.asarray(eigenvalues, dtype=float), 1e-18)
    rotation = np.asarray(rotation, dtype=float)
    local = rotation.T @ (np.asarray(point, dtype=float) - np.asarray(center, dtype=float))
    inside_measure = float(np.sum(local * local / eigenvalues))
    if inside_measure <= 1.0 + 1e-12:
        delta = np.asarray(center, dtype=float) - np.asarray(point, dtype=float)
        norm = float(np.linalg.norm(delta))
        normal = np.array([1.0, 0.0, 0.0]) if norm <= 1e-12 else delta / norm
        return EllipsoidClosestPointResult(normal=normal, surface_point=np.asarray(point, dtype=float).copy(), multiplier=0.0, newton_iterations=0, bisection_iterations=0, residual=float(max(0.0, 1.0 - inside_measure)))

    def equation(value: float) -> float:
        return float(np.sum(eigenvalues * local * local / (value + eigenvalues) ** 2) - 1.0)
    lower = 0.0
    upper = max(float(np.linalg.norm(local) * np.sqrt(np.max(eigenvalues))), 1e-12)
    while equation(upper) > 0.0:
        upper *= 2.0
    if initial_multiplier is not None and math.isfinite(float(initial_multiplier)) and (lower < float(initial_multiplier) < upper):
        multiplier = float(initial_multiplier)
    else:
        multiplier = 0.5 * (lower + upper)
    newton_count = 0
    bisection_count = 0
    residual = equation(multiplier)
    for _ in range(maximum_newton_iterations):
        newton_count += 1
        residual = equation(multiplier)
        if abs(residual) <= tolerance:
            break
        if residual > 0.0:
            lower = multiplier
        else:
            upper = multiplier
        derivative = float(-2.0 * np.sum(eigenvalues * local * local / (multiplier + eigenvalues) ** 3))
        candidate = multiplier - residual / derivative
        if not math.isfinite(candidate) or candidate <= lower or candidate >= upper:
            candidate = 0.5 * (lower + upper)
            bisection_count += 1
        multiplier = float(candidate)
    residual = equation(multiplier)
    for _ in range(maximum_bisection_iterations):
        if abs(residual) <= tolerance:
            break
        if residual > 0.0:
            lower = multiplier
        else:
            upper = multiplier
        multiplier = 0.5 * (lower + upper)
        bisection_count += 1
        residual = equation(multiplier)
    surface_local = eigenvalues * local / (multiplier + eigenvalues)
    surface_world = np.asarray(center, dtype=float) + rotation @ surface_local
    delta = surface_world - np.asarray(point, dtype=float)
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-12:
        fallback = np.asarray(center, dtype=float) - np.asarray(point, dtype=float)
        fallback_norm = float(np.linalg.norm(fallback))
        normal = np.array([1.0, 0.0, 0.0]) if fallback_norm <= 1e-12 else fallback / fallback_norm
    else:
        normal = delta / norm
    return EllipsoidClosestPointResult(normal=normal, surface_point=surface_world, multiplier=float(multiplier), newton_iterations=int(newton_count), bisection_iterations=int(bisection_count), residual=float(abs(residual)))

def closest_point_normal_on_ellipsoid(center: np.ndarray, shape_matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    return closest_point_on_ellipsoid(center, shape_matrix, point).normal

def optimal_support_normal_with_uncertainty(robot_center: np.ndarray, obstacle_center: np.ndarray, obstacle_shape: np.ndarray, uncertainty_shape: np.ndarray, *, initial_normal: np.ndarray | None=None, maximum_iterations: int=32, tolerance: float=1e-10) -> SupportNormalResult:
    robot_center = np.asarray(robot_center, dtype=float).reshape(3)
    obstacle_center = np.asarray(obstacle_center, dtype=float).reshape(3)
    obstacle_shape = np.asarray(obstacle_shape, dtype=float).reshape(3, 3)
    uncertainty_shape = np.asarray(uncertainty_shape, dtype=float).reshape(3, 3)
    delta = obstacle_center - robot_center
    delta_norm = float(np.linalg.norm(delta))
    if initial_normal is None or float(np.linalg.norm(initial_normal)) <= 1e-12:
        normal = np.array([1.0, 0.0, 0.0]) if delta_norm <= 1e-12 else delta / delta_norm
    else:
        normal = np.asarray(initial_normal, dtype=float).reshape(3).copy()
        normal /= np.linalg.norm(normal)

    def support(shape: np.ndarray, direction: np.ndarray) -> float:
        return float(np.sqrt(max(float(direction @ shape @ direction), 1e-24)))

    def objective(direction: np.ndarray) -> float:
        return float(direction @ delta - support(obstacle_shape, direction) - support(uncertainty_shape, direction))
    value = objective(normal)
    residual = float('inf')
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        obstacle_extent = support(obstacle_shape, normal)
        uncertainty_extent = support(uncertainty_shape, normal)
        gradient = delta - obstacle_shape @ normal / obstacle_extent - uncertainty_shape @ normal / uncertainty_extent
        tangent = gradient - normal * float(normal @ gradient)
        residual = float(np.linalg.norm(tangent))
        if residual <= tolerance:
            break
        direction = tangent / residual
        step = 0.5
        accepted = False
        for _ in range(48):
            candidate = normal + step * direction
            candidate /= np.linalg.norm(candidate)
            candidate_value = objective(candidate)
            if candidate_value >= value + 0.0001 * step * residual:
                normal = candidate
                value = candidate_value
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    obstacle_extent = support(obstacle_shape, normal)
    uncertainty_extent = support(uncertainty_shape, normal)
    obstacle_support_point = obstacle_center - obstacle_shape @ normal / obstacle_extent - uncertainty_shape @ normal / uncertainty_extent
    return SupportNormalResult(normal=normal, obstacle_surface_point=obstacle_support_point, clearance_without_robot=float(value), iterations=int(iterations), residual=float(residual))

def optimal_support_separating_normal(robot_center: np.ndarray, robot_shape: np.ndarray, obstacle_center: np.ndarray, obstacle_shape: np.ndarray, max_iterations: int=24) -> np.ndarray:
    robot_shape = np.asarray(robot_shape, dtype=float)
    obstacle_shape = np.asarray(obstacle_shape, dtype=float)
    isotropic_robot = bool(np.allclose(robot_shape, np.eye(3) * np.trace(robot_shape) / 3.0, rtol=1e-08, atol=1e-12))
    if isotropic_robot:
        return closest_point_normal_on_ellipsoid(obstacle_center, obstacle_shape, robot_center)
    delta = np.asarray(obstacle_center, dtype=float) - np.asarray(robot_center, dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-12:
        return np.array([1.0, 0.0, 0.0])
    normal = delta / distance

    def objective(direction: np.ndarray) -> float:
        return float(direction @ delta - support_radius(robot_shape, direction) - support_radius(obstacle_shape, direction))
    value = objective(normal)
    for _ in range(max_iterations):
        robot_extent = max(support_radius(robot_shape, normal), 1e-12)
        obstacle_extent = max(support_radius(obstacle_shape, normal), 1e-12)
        gradient = delta - robot_shape @ normal / robot_extent - obstacle_shape @ normal / obstacle_extent
        tangent = gradient - normal * float(normal @ gradient)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-10:
            break
        direction = tangent / tangent_norm
        improved = False
        for step in (0.5, 0.25, 0.125, 0.0625, 0.03125):
            candidate = normal + step * direction
            candidate /= np.linalg.norm(candidate)
            candidate_value = objective(candidate)
            if candidate_value > value + 1e-12:
                normal = candidate
                value = candidate_value
                improved = True
                break
        if not improved:
            break
    return normal

def optimal_support_sum_separating_normal(robot_center: np.ndarray, robot_shape: np.ndarray, obstacle_center: np.ndarray, obstacle_shape: np.ndarray, uncertainty_shape: np.ndarray, max_iterations: int=24) -> np.ndarray:
    delta = np.asarray(obstacle_center, dtype=float) - np.asarray(robot_center, dtype=float)
    shapes = tuple((np.asarray(item, dtype=float) for item in (robot_shape, obstacle_shape, uncertainty_shape)))
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-12:
        return np.array([1.0, 0.0, 0.0])
    normal = delta / distance

    def objective(direction: np.ndarray) -> float:
        return float(direction @ delta - sum((support_radius(q, direction) for q in shapes)))
    value = objective(normal)
    for _ in range(max_iterations):
        gradient = delta.copy()
        for shape in shapes:
            extent = max(support_radius(shape, normal), 1e-12)
            gradient -= shape @ normal / extent
        tangent = gradient - normal * float(normal @ gradient)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-10:
            break
        direction = tangent / tangent_norm
        improved = False
        for step in (0.5, 0.25, 0.125, 0.0625, 0.03125):
            candidate = normal + step * direction
            candidate /= np.linalg.norm(candidate)
            candidate_value = objective(candidate)
            if candidate_value > value + 1e-12:
                normal, value, improved = (candidate, candidate_value, True)
                break
        if not improved:
            break
    return normal
