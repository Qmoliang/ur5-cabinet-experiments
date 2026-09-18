"""Conservative ellipsoid certificates for the UR5e LiuQP ablation.

This is inspired by IRIS's anisotropic ellipsoids, but it is not IRIS region
inflation. It replaces direction-independent ball support radii with
ellipsoid support functions inside the same sequential QP architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from model import BoxObstacle
from model import CertificateSphere


@dataclass(frozen=True)
class RobotEllipsoid:
    body_id: int
    body_name: str
    source_geom_id: int
    local_center: np.ndarray
    local_rotation: np.ndarray
    semi_axes: np.ndarray


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


def spheres_as_isotropic_ellipsoids(
    spheres: list[CertificateSphere],
) -> list[RobotEllipsoid]:
    """Convert a sphere certificate to the isotropic ellipsoid limit."""

    return [
        RobotEllipsoid(
            body_id=sphere.body_id,
            body_name=sphere.body_name,
            source_geom_id=sphere.source_geom_id,
            local_center=sphere.local_center.copy(),
            local_rotation=np.eye(3),
            semi_axes=np.full(3, sphere.radius),
        )
        for sphere in spheres
    ]


def support_radius(shape_matrix: np.ndarray, normal: np.ndarray) -> float:
    """Return sqrt(n^T Q n) for E={c+x | x^T Q^-1 x <= 1}."""

    value = float(normal @ shape_matrix @ normal)
    return float(np.sqrt(max(value, 0.0)))


def support_angular_gradient(shape_matrix: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Derivative of support radius with respect to world angular velocity."""

    radius = support_radius(shape_matrix, normal)
    if radius <= 1.0e-12:
        return np.zeros(3)
    return np.cross(shape_matrix @ normal, normal) / radius


def closest_point_on_ellipsoid(
    center: np.ndarray,
    shape_matrix: np.ndarray,
    point: np.ndarray,
    *,
    initial_multiplier: float | None = None,
    eigenvalues: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
    maximum_newton_iterations: int = 10,
    maximum_bisection_iterations: int = 48,
    tolerance: float = 1.0e-12,
) -> EllipsoidClosestPointResult:
    """Closest point with safeguarded Newton and bisection fallback.

    For E={c+R x | sum(x_i^2/a_i^2)<=1}, the closest point satisfies
    x_i=a_i^2 y_i/(lambda+a_i^2).  The scalar lambda>=0 is the unique
    root of sum(a_i^2 y_i^2/(lambda+a_i^2)^2)=1.  This normal maximizes
    n^T(c-p)-rho_E(n), which is exactly the separating-plane clearance
    needed when the robot proxy is a sphere.
    """

    if eigenvalues is None or rotation is None:
        eigenvalues, rotation = np.linalg.eigh(
            np.asarray(shape_matrix, dtype=float)
        )
    eigenvalues = np.maximum(np.asarray(eigenvalues, dtype=float), 1.0e-18)
    rotation = np.asarray(rotation, dtype=float)
    local = rotation.T @ (np.asarray(point, dtype=float) - np.asarray(center, dtype=float))
    inside_measure = float(np.sum(local * local / eigenvalues))
    if inside_measure <= 1.0 + 1.0e-12:
        delta = np.asarray(center, dtype=float) - np.asarray(point, dtype=float)
        norm = float(np.linalg.norm(delta))
        normal = (
            np.array([1.0, 0.0, 0.0])
            if norm <= 1.0e-12
            else delta / norm
        )
        return EllipsoidClosestPointResult(
            normal=normal,
            surface_point=np.asarray(point, dtype=float).copy(),
            multiplier=0.0,
            newton_iterations=0,
            bisection_iterations=0,
            residual=float(max(0.0, 1.0 - inside_measure)),
        )

    def equation(value: float) -> float:
        return float(
            np.sum(eigenvalues * local * local / (value + eigenvalues) ** 2) - 1.0
        )

    lower = 0.0
    upper = max(float(np.linalg.norm(local) * np.sqrt(np.max(eigenvalues))), 1.0e-12)
    while equation(upper) > 0.0:
        upper *= 2.0

    if (
        initial_multiplier is not None
        and math.isfinite(float(initial_multiplier))
        and lower < float(initial_multiplier) < upper
    ):
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
        derivative = float(
            -2.0
            * np.sum(
                eigenvalues
                * local
                * local
                / (multiplier + eigenvalues) ** 3
            )
        )
        candidate = multiplier - residual / derivative
        if (
            not math.isfinite(candidate)
            or candidate <= lower
            or candidate >= upper
        ):
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
    if norm <= 1.0e-12:
        fallback = np.asarray(center, dtype=float) - np.asarray(point, dtype=float)
        fallback_norm = float(np.linalg.norm(fallback))
        normal = (
            np.array([1.0, 0.0, 0.0])
            if fallback_norm <= 1.0e-12
            else fallback / fallback_norm
        )
    else:
        normal = delta / norm
    return EllipsoidClosestPointResult(
        normal=normal,
        surface_point=surface_world,
        multiplier=float(multiplier),
        newton_iterations=int(newton_count),
        bisection_iterations=int(bisection_count),
        residual=float(abs(residual)),
    )


def closest_point_normal_on_ellipsoid(
    center: np.ndarray,
    shape_matrix: np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    """Backward-compatible unit normal wrapper."""

    return closest_point_on_ellipsoid(center, shape_matrix, point).normal


def optimal_support_normal_with_uncertainty(
    robot_center: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_shape: np.ndarray,
    uncertainty_shape: np.ndarray,
    *,
    initial_normal: np.ndarray | None = None,
    maximum_iterations: int = 32,
    tolerance: float = 1.0e-10,
) -> SupportNormalResult:
    """Maximize the exact support separation for ``E(Q) (+) E(U)``.

    The optimized objective on the unit sphere is

        n' (o-p) - sqrt(n'Qn) - sqrt(n'Un).

    This is the IRIS-style separating-plane support problem needed for a
    directional depth certificate.  Backtracking Riemannian gradient ascent
    starts from the previous cycle's stable-proxy normal when available.  It
    does not replace the two support terms by one outer ellipsoid.
    """

    robot_center = np.asarray(robot_center, dtype=float).reshape(3)
    obstacle_center = np.asarray(obstacle_center, dtype=float).reshape(3)
    obstacle_shape = np.asarray(obstacle_shape, dtype=float).reshape(3, 3)
    uncertainty_shape = np.asarray(uncertainty_shape, dtype=float).reshape(3, 3)
    delta = obstacle_center - robot_center
    delta_norm = float(np.linalg.norm(delta))
    if initial_normal is None or float(np.linalg.norm(initial_normal)) <= 1.0e-12:
        normal = (
            np.array([1.0, 0.0, 0.0])
            if delta_norm <= 1.0e-12
            else delta / delta_norm
        )
    else:
        normal = np.asarray(initial_normal, dtype=float).reshape(3).copy()
        normal /= np.linalg.norm(normal)

    def support(shape: np.ndarray, direction: np.ndarray) -> float:
        return float(np.sqrt(max(float(direction @ shape @ direction), 1.0e-24)))

    def objective(direction: np.ndarray) -> float:
        return float(
            direction @ delta
            - support(obstacle_shape, direction)
            - support(uncertainty_shape, direction)
        )

    value = objective(normal)
    residual = float("inf")
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        obstacle_extent = support(obstacle_shape, normal)
        uncertainty_extent = support(uncertainty_shape, normal)
        gradient = (
            delta
            - obstacle_shape @ normal / obstacle_extent
            - uncertainty_shape @ normal / uncertainty_extent
        )
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
            if candidate_value >= value + 1.0e-4 * step * residual:
                normal = candidate
                value = candidate_value
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

    obstacle_extent = support(obstacle_shape, normal)
    uncertainty_extent = support(uncertainty_shape, normal)
    obstacle_support_point = (
        obstacle_center
        - obstacle_shape @ normal / obstacle_extent
        - uncertainty_shape @ normal / uncertainty_extent
    )
    return SupportNormalResult(
        normal=normal,
        obstacle_surface_point=obstacle_support_point,
        clearance_without_robot=float(value),
        iterations=int(iterations),
        residual=float(residual),
    )


def optimal_support_separating_normal(
    robot_center: np.ndarray,
    robot_shape: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_shape: np.ndarray,
    max_iterations: int = 24,
) -> np.ndarray:
    """Maximize n^T(o-p)-rho_robot(n)-rho_obstacle(n), ||n||=1.

    This is the support-function form of the IRIS separating-hyperplane
    subproblem.  The sphere/ellipsoid case uses the exact closest-point
    solution; the general ellipsoid/ellipsoid case uses deterministic
    projected ascent with monotone backtracking.
    """

    robot_shape = np.asarray(robot_shape, dtype=float)
    obstacle_shape = np.asarray(obstacle_shape, dtype=float)
    isotropic_robot = bool(
        np.allclose(
            robot_shape,
            np.eye(3) * np.trace(robot_shape) / 3.0,
            rtol=1.0e-8,
            atol=1.0e-12,
        )
    )
    if isotropic_robot:
        return closest_point_normal_on_ellipsoid(
            obstacle_center, obstacle_shape, robot_center
        )

    delta = np.asarray(obstacle_center, dtype=float) - np.asarray(robot_center, dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0])
    normal = delta / distance

    def objective(direction: np.ndarray) -> float:
        return float(
            direction @ delta
            - support_radius(robot_shape, direction)
            - support_radius(obstacle_shape, direction)
        )

    value = objective(normal)
    for _ in range(max_iterations):
        robot_extent = max(support_radius(robot_shape, normal), 1.0e-12)
        obstacle_extent = max(support_radius(obstacle_shape, normal), 1.0e-12)
        gradient = (
            delta
            - robot_shape @ normal / robot_extent
            - obstacle_shape @ normal / obstacle_extent
        )
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


def optimal_support_sum_separating_normal(
    robot_center: np.ndarray,
    robot_shape: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_shape: np.ndarray,
    uncertainty_shape: np.ndarray,
    max_iterations: int = 24,
) -> np.ndarray:
    """Maximize n·(o-p)-rho_R(n)-rho_B(n)-rho_U(n), ||n||=1."""

    delta = np.asarray(obstacle_center, dtype=float) - np.asarray(robot_center, dtype=float)
    shapes = tuple(
        np.asarray(item, dtype=float)
        for item in (robot_shape, obstacle_shape, uncertainty_shape)
    )
    distance = float(np.linalg.norm(delta))
    if distance <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0])
    normal = delta / distance

    def objective(direction: np.ndarray) -> float:
        return float(direction @ delta - sum(support_radius(q, direction) for q in shapes))

    value = objective(normal)
    for _ in range(max_iterations):
        gradient = delta.copy()
        for shape in shapes:
            extent = max(support_radius(shape, normal), 1.0e-12)
            gradient -= shape @ normal / extent
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
                normal, value, improved = candidate, candidate_value, True
                break
        if not improved:
            break
    return normal


def _quat_matrix(quat: np.ndarray) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def build_robot_ellipsoid_certificate(
    model: mujoco.MjModel,
    axial_spacing: float = 0.035,
    axial_overlap_factor: float = 1.0,
) -> list[RobotEllipsoid]:
    """Conservatively cover capsule/cylinder primitives by ellipsoid chains.

    Consecutive centers are at most s apart. For an interior cylinder slab,
    choose axial semi-axis a=k*s and radial semi-axis
    b=r/sqrt(1-(s/(2a))^2). Every point in the slab then lies in at least one
    ellipsoid. Capsule hemispheres are covered by endpoint sphere-ellipsoids.
    """

    proxies: list[RobotEllipsoid] = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id == 0:
            continue
        geom_type = int(model.geom_type[geom_id])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            half = np.asarray(model.geom_size[geom_id], dtype=float)
            counts = np.maximum(1, np.ceil((2.0 * half) / 0.024).astype(int))
            step = 2.0 * half / counts
            geom_rotation = _quat_matrix(model.geom_quat[geom_id])
            body_name = model.body(body_id).name
            axes = [
                np.linspace(
                    -half[axis] + 0.5 * step[axis],
                    half[axis] - 0.5 * step[axis],
                    counts[axis],
                )
                for axis in range(3)
            ]
            semi_axes = np.sqrt(3.0) * 0.5 * step
            for x in axes[0]:
                for y in axes[1]:
                    for z in axes[2]:
                        local = model.geom_pos[geom_id] + geom_rotation @ np.array([x, y, z])
                        proxies.append(
                            RobotEllipsoid(
                                body_id=body_id,
                                body_name=body_name,
                                source_geom_id=geom_id,
                                local_center=local.copy(),
                                local_rotation=geom_rotation.copy(),
                                semi_axes=semi_axes.copy(),
                            )
                        )
            continue
        if geom_type not in (
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        ):
            continue
        radius = float(model.geom_size[geom_id, 0])
        half_length = float(model.geom_size[geom_id, 1])
        count = max(2, int(np.ceil((2.0 * half_length) / axial_spacing)) + 1)
        axial = np.linspace(-half_length, half_length, count)
        spacing = float(axial[1] - axial[0])
        axial_axis = max(axial_overlap_factor * spacing, 0.5 * spacing + 1.0e-9)
        ratio = 0.5 * spacing / axial_axis
        radial_axis = radius / np.sqrt(max(1.0 - ratio * ratio, 1.0e-12))
        geom_rotation = _quat_matrix(model.geom_quat[geom_id])
        body_name = model.body(body_id).name
        for z in axial:
            local = model.geom_pos[geom_id] + geom_rotation @ np.array([0.0, 0.0, z])
            proxies.append(
                RobotEllipsoid(
                    body_id=body_id,
                    body_name=body_name,
                    source_geom_id=geom_id,
                    local_center=local.copy(),
                    local_rotation=geom_rotation.copy(),
                    semi_axes=np.array([radial_axis, radial_axis, axial_axis]),
                )
            )
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            for z in (-half_length, half_length):
                local = model.geom_pos[geom_id] + geom_rotation @ np.array([0.0, 0.0, z])
                proxies.append(
                    RobotEllipsoid(
                        body_id=body_id,
                        body_name=body_name,
                        source_geom_id=geom_id,
                        local_center=local.copy(),
                        local_rotation=np.eye(3),
                        semi_axes=np.full(3, radius),
                    )
                )
    return proxies


def ellipsoid_world_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    certificate: list[RobotEllipsoid],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centers, translational/rotational Jacobians, and world Q matrices."""

    count = len(certificate)
    positions = np.zeros((count, 3))
    jac_pos = np.zeros((count, 3, model.nv))
    jac_rot = np.zeros((count, 3, model.nv))
    shapes = np.zeros((count, 3, 3))
    for index, proxy in enumerate(certificate):
        body_rotation = data.xmat[proxy.body_id].reshape(3, 3)
        position = data.xpos[proxy.body_id] + body_rotation @ proxy.local_center
        world_rotation = body_rotation @ proxy.local_rotation
        positions[index] = position
        shapes[index] = world_rotation @ np.diag(proxy.semi_axes**2) @ world_rotation.T
        mujoco.mj_jac(model, data, jac_pos[index], jac_rot[index], position, proxy.body_id)
    return positions, jac_pos, jac_rot, shapes


def sample_box_ellipsoid_tree(
    boxes: tuple[BoxObstacle, ...],
    cell_size: float = 0.075,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cover each box-tree cell by its minimum-volume enclosing ellipsoid."""

    centers: list[np.ndarray] = []
    shapes: list[np.ndarray] = []
    owners: list[int] = []
    for owner, box in enumerate(boxes):
        half = np.asarray(box.half_size, dtype=float)
        count = np.maximum(1, np.ceil((2.0 * half) / cell_size).astype(int))
        step = 2.0 * half / count
        axes_grid = [
            np.linspace(-half[i] + 0.5 * step[i], half[i] - 0.5 * step[i], count[i])
            for i in range(3)
        ]
        semi_axes = np.sqrt(3.0) * 0.5 * step
        shape = np.diag(semi_axes**2)
        for x in axes_grid[0]:
            for y in axes_grid[1]:
                for z in axes_grid[2]:
                    centers.append(np.asarray(box.center) + np.array([x, y, z]))
                    shapes.append(shape.copy())
                    owners.append(owner)
    return np.asarray(centers), np.asarray(shapes), np.asarray(owners, dtype=int)
