"""LiuQP with conservative ellipsoid support-function constraints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import mujoco
import numpy as np
import osqp
from scipy import sparse

from ellipsoid_model import (
    RobotEllipsoid,
    closest_point_on_ellipsoid,
    ellipsoid_world_state,
    optimal_support_separating_normal,
    optimal_support_sum_separating_normal,
    support_angular_gradient,
    support_radius,
)
from model import DT, JOINT_NAMES, SceneDefinition, attachment_position
from multilevel_voxel_table import MultilevelVoxelTable
from native_ellipsoid_support import NativeEllipsoidSupport


@dataclass(frozen=True)
class EllipsoidSeparatingPlane:
    obstacle_index: int
    normal_to_obstacle: np.ndarray
    tangent_point: np.ndarray
    offset: float
    robot_support: float
    obstacle_support: float
    clearance: float
    support_angular_gradient: np.ndarray


@dataclass
class EllipsoidStepMetrics:
    status: str
    solve_ms: float
    ee_error: float
    task_speed: float
    qdot_norm: float
    min_clearance: float
    raw_obstacle_proxies: int
    broadphase_candidate_pairs: int
    active_obstacle_rows: int
    redundant_proxies_removed: int
    near_penalty_terms: int
    contact_repulsion_rows: int
    workspace_rows: int
    closest_point_newton_iterations: int
    closest_point_bisection_iterations: int
    support_projected_ascent_iterations: int
    limiting_robot_index: int
    limiting_obstacle_index: int

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


class EllipsoidLiuQPController:
    """Same local QP objective as LiuQP, with ellipsoid support constraints."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        scene: SceneDefinition,
        robot_ellipsoids: list[RobotEllipsoid],
        obstacle_centers: np.ndarray,
        obstacle_shapes: np.ndarray,
        *,
        obstacle_offsets: np.ndarray | float = 0.0,
        obstacle_uncertainty_shapes: np.ndarray | None = None,
        obstacle_pruning_shapes: np.ndarray | None = None,
        safety_margin: float = 0.006,
        near_distance: float = 0.12,
        contact_distance: float = 0.002,
        obstacle_index: MultilevelVoxelTable | None = None,
        native_support_batch: bool = True,
        redundant_plane_pruning: bool = True,
        native_plane_pruning: bool = True,
    ) -> None:
        self.model = model
        self.data = data
        self.scene = scene
        self.robot_ellipsoids = robot_ellipsoids
        self.obstacle_centers = np.asarray(obstacle_centers, dtype=float)
        self.obstacle_shapes = np.asarray(obstacle_shapes, dtype=float)
        self.obstacle_uncertainty_shapes = (
            np.zeros_like(self.obstacle_shapes)
            if obstacle_uncertainty_shapes is None
            else np.asarray(obstacle_uncertainty_shapes, dtype=float).reshape(-1, 3, 3)
        )
        self.obstacle_pruning_shapes = (
            self.obstacle_shapes
            if obstacle_pruning_shapes is None
            else np.asarray(obstacle_pruning_shapes, dtype=float).reshape(-1, 3, 3)
        )
        (
            self.obstacle_eigenvalues,
            self.obstacle_rotations,
        ) = np.linalg.eigh(self.obstacle_shapes)
        self.obstacle_offsets = np.broadcast_to(
            np.asarray(obstacle_offsets, dtype=float),
            (len(self.obstacle_centers),),
        ).copy()
        self.safety_margin = float(safety_margin)
        self.near_distance = float(near_distance)
        self.contact_distance = float(contact_distance)
        self.obstacle_index = obstacle_index
        self.native_support_batch = bool(native_support_batch)
        self.redundant_plane_pruning = bool(redundant_plane_pruning)
        self.native_plane_pruning = bool(native_plane_pruning)
        self._last_prune_candidate_count = len(self.obstacle_centers)
        self.nv = len(JOINT_NAMES)
        self.dt = DT
        self.ee_site_id = model.site("attachment_site").id
        self.orientation_target = data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        self.orientation_weight = (
            30.0
            if scene.name.startswith(("shelf_drawer", "incremental_drawer"))
            else 0.0
        )
        self.orientation_gain = 2.0
        self.previous_velocity = np.zeros(self.nv)
        self.objective_dither_std = 0.0
        self.objective_dither_rng = np.random.default_rng(0)
        self._closest_multiplier_cache: dict[tuple[int, int], float] = {}
        self._closest_newton_iterations = 0
        self._closest_bisection_iterations = 0
        self._support_projected_iterations = 0
        self._native_support = NativeEllipsoidSupport()
        self.task_region_A: np.ndarray | None = None
        self.task_region_b: np.ndarray | None = None
        self.posture_target: np.ndarray | None = None
        self.posture_weight = 80.0
        self.posture_gain = 2.0

        self.task_gain = 2.0
        self.max_task_speed = 0.18
        self.task_weight = 90.0
        self.near_weight = 22.0
        self.velocity_regularization = 1.0
        self.repulsive_speed = 0.035
        self.velocity_limits = np.array([1.0, 1.0, 1.0, 1.4, 1.4, 1.4])
        self.joint_min = np.array(
            [model.jnt_range[model.joint(name).id, 0] for name in JOINT_NAMES]
        )
        self.joint_max = np.array(
            [model.jnt_range[model.joint(name).id, 1] for name in JOINT_NAMES]
        )
        self.joint_padding = 0.015
        self.workspace_A = np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.workspace_b = np.array([0.95, 0.95, 0.95, 0.95, 0.0, 1.35])

    def task_feedback(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ee = attachment_position(self.model, self.data)
        jacobian = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacobian, jac_rot, self.ee_site_id)
        jacobian = jacobian[:, : self.nv]
        desired_velocity = self.task_gain * (np.asarray(target) - ee)
        speed = float(np.linalg.norm(desired_velocity))
        if speed > self.max_task_speed:
            desired_velocity *= self.max_task_speed / speed
        return ee, jacobian, desired_velocity

    def orientation_feedback(self) -> tuple[np.ndarray, np.ndarray]:
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.ee_site_id)
        current = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        error_matrix = self.orientation_target @ current.T
        error = 0.5 * np.array(
            [
                error_matrix[2, 1] - error_matrix[1, 2],
                error_matrix[0, 2] - error_matrix[2, 0],
                error_matrix[1, 0] - error_matrix[0, 1],
            ]
        )
        return jac_rot[:, : self.nv], self.orientation_gain * error

    def joint_velocity_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        q = self.data.qpos[: self.nv]
        position_lower = (self.joint_min + self.joint_padding - q) / self.dt
        position_upper = (self.joint_max - self.joint_padding - q) / self.dt
        return (
            np.maximum(-self.velocity_limits, position_lower),
            np.minimum(self.velocity_limits, position_upper),
        )

    def separating_plane(
        self,
        robot_center: np.ndarray,
        robot_shape: np.ndarray,
        obstacle_index: int,
        robot_index: int = -1,
    ) -> EllipsoidSeparatingPlane:
        obstacle_center = self.obstacle_centers[obstacle_index]
        obstacle_shape = self.obstacle_shapes[obstacle_index]
        delta = obstacle_center - robot_center
        distance = float(np.linalg.norm(delta))
        isotropic_robot = bool(
            np.allclose(
                robot_shape,
                np.eye(3) * np.trace(robot_shape) / 3.0,
                rtol=1.0e-8,
                atol=1.0e-12,
            )
        )
        if isotropic_robot and not np.any(
            self.obstacle_uncertainty_shapes[obstacle_index]
        ):
            cache_key = (int(robot_index), int(obstacle_index))
            closest = closest_point_on_ellipsoid(
                obstacle_center,
                obstacle_shape,
                robot_center,
                initial_multiplier=self._closest_multiplier_cache.get(cache_key),
                eigenvalues=self.obstacle_eigenvalues[obstacle_index],
                rotation=self.obstacle_rotations[obstacle_index],
            )
            normal = closest.normal
            self._closest_multiplier_cache[cache_key] = closest.multiplier
            self._closest_newton_iterations += closest.newton_iterations
            self._closest_bisection_iterations += closest.bisection_iterations
        else:
            uncertainty_shape = self.obstacle_uncertainty_shapes[obstacle_index]
            normal = (
                optimal_support_sum_separating_normal(
                    robot_center,
                    robot_shape,
                    obstacle_center,
                    obstacle_shape,
                    uncertainty_shape,
                )
                if np.any(uncertainty_shape)
                else optimal_support_separating_normal(
                    robot_center, robot_shape, obstacle_center, obstacle_shape
                )
            )
        robot_extent = support_radius(robot_shape, normal)
        obstacle_extent = (
            support_radius(obstacle_shape, normal)
            + support_radius(
                self.obstacle_uncertainty_shapes[obstacle_index], normal
            )
            + self.obstacle_offsets[obstacle_index]
        )
        tangent = obstacle_center - obstacle_extent * normal
        offset = float(normal @ tangent)
        # The normal is generally not the center-line direction for anisotropic
        # ellipsoids.  The separating-plane slack is therefore the projected
        # center separation, not ||o-p||.  Using Euclidean distance here makes
        # the QP constraint looser than the support plane checked by safety
        # backtracking and can reject every proposed integration step.
        clearance = (
            float(normal @ delta)
            - robot_extent
            - obstacle_extent
            - self.safety_margin
        )
        return EllipsoidSeparatingPlane(
            obstacle_index=obstacle_index,
            normal_to_obstacle=normal,
            tangent_point=tangent,
            offset=offset,
            robot_support=robot_extent,
            obstacle_support=obstacle_extent,
            clearance=float(clearance),
            support_angular_gradient=support_angular_gradient(robot_shape, normal),
        )

    def prune_redundant_obstacle_ellipsoids(
        self,
        robot_center: np.ndarray,
        robot_shape: np.ndarray,
        robot_index: int = -1,
    ) -> list[EllipsoidSeparatingPlane]:
        if not len(self.obstacle_centers):
            self._last_prune_candidate_count = 0
            return []
        if self.obstacle_index is None:
            candidates = np.arange(len(self.obstacle_centers), dtype=int)
        else:
            candidates = self.obstacle_index.query_ellipsoid(robot_center, robot_shape)
        self._last_prune_candidate_count = len(candidates)
        if not len(candidates):
            return []
        delta = self.obstacle_centers[candidates] - robot_center
        distances = np.linalg.norm(delta, axis=1)
        normals = delta / np.maximum(distances[:, None], 1.0e-12)
        robot_extent = np.sqrt(
            np.maximum(np.einsum("ni,ij,nj->n", normals, robot_shape, normals), 0.0)
        )
        obstacle_extent = np.sqrt(
            np.maximum(
                np.einsum(
                    "ni,nij,nj->n",
                    normals,
                    self.obstacle_shapes[candidates],
                    normals,
                ),
                0.0,
            )
        )
        obstacle_extent += np.sqrt(
            np.maximum(
                np.einsum(
                    "ni,nij,nj->n",
                    normals,
                    self.obstacle_uncertainty_shapes[candidates],
                    normals,
                ),
                0.0,
            )
        )
        obstacle_extent = obstacle_extent + self.obstacle_offsets[candidates]
        centerline_clearance = (
            distances - robot_extent - obstacle_extent - self.safety_margin
        )
        if self.obstacle_index is not None:
            keep = centerline_clearance <= self.near_distance
            candidates = candidates[keep]
            centerline_clearance = centerline_clearance[keep]
            if not len(candidates):
                return []
        remaining = list(candidates[np.argsort(centerline_clearance)])
        isotropic_robot = bool(
            np.allclose(
                robot_shape,
                np.eye(3) * np.trace(robot_shape) / 3.0,
                rtol=1.0e-8,
                atol=1.0e-12,
            )
        )
        native_plane_arrays = None
        native_local_indices: dict[int, int] = {}
        if not isotropic_robot and self.native_support_batch:
            uncertainty = self.obstacle_uncertainty_shapes[candidates]
            if np.any(uncertainty):
                normals, iterations = self._native_support.normals_sum(
                    robot_center,
                    robot_shape,
                    self.obstacle_centers[candidates],
                    self.obstacle_shapes[candidates],
                    uncertainty,
                )
            else:
                normals, iterations = self._native_support.normals(
                    robot_center,
                    robot_shape,
                    self.obstacle_centers[candidates],
                    self.obstacle_shapes[candidates],
                )
            self._support_projected_iterations += int(np.sum(iterations))
            robot_shape_normals = normals @ robot_shape.T
            robot_extents = np.sqrt(
                np.maximum(
                    np.einsum("ni,ni->n", normals, robot_shape_normals),
                    0.0,
                )
            )
            obstacle_shape_normals = np.einsum(
                "nij,nj->ni", self.obstacle_shapes[candidates], normals
            )
            obstacle_extents = np.sqrt(
                np.maximum(
                    np.einsum("ni,ni->n", normals, obstacle_shape_normals),
                    0.0,
                )
            )
            uncertainty_shape_normals = np.einsum(
                "nij,nj->ni",
                self.obstacle_uncertainty_shapes[candidates],
                normals,
            )
            obstacle_extents += np.sqrt(
                np.maximum(
                    np.einsum(
                        "ni,ni->n", normals, uncertainty_shape_normals
                    ),
                    0.0,
                )
            )
            obstacle_extents += self.obstacle_offsets[candidates]
            tangents = (
                self.obstacle_centers[candidates]
                - obstacle_extents[:, None] * normals
            )
            offsets = np.einsum("ni,ni->n", normals, tangents)
            clearances = (
                np.einsum(
                    "ni,ni->n",
                    normals,
                    self.obstacle_centers[candidates] - robot_center,
                )
                - robot_extents
                - obstacle_extents
                - self.safety_margin
            )
            angular_gradients = np.cross(
                robot_shape_normals, normals
            ) / np.maximum(robot_extents[:, None], 1.0e-12)
            native_plane_arrays = (
                normals,
                tangents,
                offsets,
                robot_extents,
                obstacle_extents,
                clearances,
                angular_gradients,
            )
            native_local_indices = {
                int(obstacle_index): local_index
                for local_index, obstacle_index in enumerate(candidates)
            }
            if not self.redundant_plane_pruning:
                return [
                    EllipsoidSeparatingPlane(
                        obstacle_index=int(obstacle_index),
                        normal_to_obstacle=normals[local],
                        tangent_point=tangents[local],
                        offset=float(offsets[local]),
                        robot_support=float(robot_extents[local]),
                        obstacle_support=float(obstacle_extents[local]),
                        clearance=float(clearances[local]),
                        support_angular_gradient=angular_gradients[local],
                    )
                    for local, obstacle_index in enumerate(candidates)
                ]
            if self.native_plane_pruning:
                order = np.argsort(centerline_clearance)
                active_sorted_local = self._native_support.prune_planes(
                    self.obstacle_centers[candidates][order],
                    self.obstacle_pruning_shapes[candidates][order],
                    self.obstacle_offsets[candidates][order],
                    normals[order],
                    offsets[order],
                )
                active_local = order[active_sorted_local]
                return [
                    EllipsoidSeparatingPlane(
                        obstacle_index=int(candidates[local]),
                        normal_to_obstacle=normals[local],
                        tangent_point=tangents[local],
                        offset=float(offsets[local]),
                        robot_support=float(robot_extents[local]),
                        obstacle_support=float(obstacle_extents[local]),
                        clearance=float(clearances[local]),
                        support_angular_gradient=angular_gradients[local],
                    )
                    for local in active_local
                ]
        active: list[EllipsoidSeparatingPlane] = []
        tolerance = 1.0e-10
        while remaining:
            nearest = int(remaining[0])
            if native_plane_arrays is not None:
                local = native_local_indices[nearest]
                (
                    normals,
                    tangents,
                    offsets,
                    robot_extents,
                    obstacle_extents,
                    clearances,
                    angular_gradients,
                ) = native_plane_arrays
                plane = EllipsoidSeparatingPlane(
                    obstacle_index=nearest,
                    normal_to_obstacle=normals[local],
                    tangent_point=tangents[local],
                    offset=float(offsets[local]),
                    robot_support=float(robot_extents[local]),
                    obstacle_support=float(obstacle_extents[local]),
                    clearance=float(clearances[local]),
                    support_angular_gradient=angular_gradients[local],
                )
            else:
                plane = self.separating_plane(
                    robot_center,
                    robot_shape,
                    nearest,
                    robot_index=robot_index,
                )
            n = plane.normal_to_obstacle
            shapes = self.obstacle_shapes[remaining]
            projected_extent = np.sqrt(
                np.maximum(np.einsum("i,nij,j->n", n, shapes, n), 0.0)
            )
            uncertainty_shapes = self.obstacle_uncertainty_shapes[remaining]
            projected_extent += np.sqrt(
                np.maximum(
                    np.einsum("i,nij,j->n", n, uncertainty_shapes, n),
                    0.0,
                )
            )
            projected_extent = projected_extent + self.obstacle_offsets[remaining]
            projection_min = self.obstacle_centers[remaining] @ n - projected_extent
            behind = projection_min >= plane.offset - tolerance
            active.append(plane)
            remaining = [index for index, remove in zip(remaining, behind) if not remove]
        return active

    def add_workspace_constraints(
        self,
        rows: list[np.ndarray],
        lower: list[float],
        upper: list[float],
        position: np.ndarray,
        jac_pos: np.ndarray,
        jac_rot: np.ndarray,
        shape: np.ndarray,
    ) -> int:
        for normal, offset in zip(self.workspace_A, self.workspace_b):
            extent = support_radius(shape, normal)
            omega_gradient = support_angular_gradient(shape, normal)
            row = normal @ jac_pos + omega_gradient @ jac_rot
            rows.append(row)
            lower.append(-np.inf)
            upper.append(float((offset - extent - normal @ position) / self.dt))
        return len(self.workspace_A)

    def workspace_constraints_batch(
        self,
        positions: np.ndarray,
        jac_pos: np.ndarray,
        jac_rot: np.ndarray,
        shapes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized equivalent of six support constraints per ellipsoid."""

        normals = self.workspace_A
        extents = np.sqrt(
            np.maximum(
                np.einsum("ai,rij,aj->ra", normals, shapes, normals),
                0.0,
            )
        )
        shape_normals = np.einsum("rij,aj->rai", shapes, normals)
        angular_gradients = np.cross(
            shape_normals, normals[None, :, :]
        ) / np.maximum(extents[:, :, None], 1.0e-12)
        translation_rows = np.einsum(
            "ai,rij->raj", normals, jac_pos
        )
        rotation_rows = np.einsum(
            "rai,rij->raj", angular_gradients, jac_rot
        )
        bounds = (
            self.workspace_b[None, :]
            - extents
            - positions @ normals.T
        ) / self.dt
        return (
            (translation_rows + rotation_rows).reshape(-1, self.nv),
            bounds.reshape(-1),
        )

    def solve(self, target: np.ndarray) -> tuple[np.ndarray, EllipsoidStepMetrics]:
        self._closest_newton_iterations = 0
        self._closest_bisection_iterations = 0
        self._support_projected_iterations = 0
        ee, ee_jacobian, desired_velocity = self.task_feedback(target)
        positions, jac_pos_full, jac_rot_full, shapes = ellipsoid_world_state(
            self.model, self.data, self.robot_ellipsoids
        )
        jac_pos = jac_pos_full[:, :, : self.nv]
        jac_rot = jac_rot_full[:, :, : self.nv]

        hessian = self.velocity_regularization * np.eye(self.nv)
        hessian += self.task_weight * (ee_jacobian.T @ ee_jacobian)
        gradient = -self.task_weight * (ee_jacobian.T @ desired_velocity)
        if self.orientation_weight > 0.0:
            orientation_jacobian, desired_omega = self.orientation_feedback()
            hessian += self.orientation_weight * (
                orientation_jacobian.T @ orientation_jacobian
            )
            gradient -= self.orientation_weight * (
                orientation_jacobian.T @ desired_omega
            )
        if self.posture_target is not None:
            posture_velocity = self.posture_gain * (
                np.asarray(self.posture_target) - self.data.qpos[: self.nv]
            )
            posture_velocity = np.clip(
                posture_velocity, -self.velocity_limits, self.velocity_limits
            )
            hessian += self.posture_weight * np.eye(self.nv)
            gradient -= self.posture_weight * posture_velocity
        if self.objective_dither_std > 0.0:
            gradient += self.objective_dither_rng.normal(
                0.0, self.objective_dither_std, size=self.nv
            )
        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []

        qdot_lower, qdot_upper = self.joint_velocity_bounds()
        for joint in range(self.nv):
            row = np.zeros(self.nv)
            row[joint] = 1.0
            rows.append(row)
            lower.append(float(qdot_lower[joint]))
            upper.append(float(qdot_upper[joint]))

        workspace_rows = active_rows = near_terms = contact_rows = 0
        if self.task_region_A is not None:
            region_rows = self.task_region_A @ ee_jacobian
            region_bounds = (
                self.task_region_b - self.task_region_A @ ee
            ) / self.dt
            rows.extend(region_rows)
            lower.extend(np.full(len(region_rows), -np.inf))
            upper.extend(region_bounds)
            workspace_rows += len(region_rows)
        workspace_matrix, workspace_bounds = self.workspace_constraints_batch(
            positions, jac_pos, jac_rot, shapes
        )
        rows.extend(workspace_matrix)
        lower.extend(np.full(len(workspace_matrix), -np.inf))
        upper.extend(workspace_bounds)
        workspace_rows += len(workspace_matrix)
        candidate_pairs = 0
        minimum_clearance = np.inf
        limiting_robot_index = -1
        limiting_obstacle_index = -1
        for robot_index, (position, jpos, jrot, shape) in enumerate(
            zip(positions, jac_pos, jac_rot, shapes)
        ):
            active_planes = self.prune_redundant_obstacle_ellipsoids(
                position,
                shape,
                robot_index=robot_index,
            )
            candidate_pairs += self._last_prune_candidate_count
            for plane in active_planes:
                normal = plane.normal_to_obstacle
                projected_jacobian = (
                    normal @ jpos + plane.support_angular_gradient @ jrot
                )
                if plane.clearance < minimum_clearance:
                    minimum_clearance = plane.clearance
                    limiting_robot_index = robot_index
                    limiting_obstacle_index = plane.obstacle_index
                rows.append(projected_jacobian)
                lower.append(-np.inf)
                upper.append(float(plane.clearance / self.dt))
                active_rows += 1
                if plane.clearance <= self.contact_distance:
                    penetration = max(0.0, -plane.clearance)
                    repulsion = self.repulsive_speed * min(
                        1.0, 0.25 + penetration / max(self.safety_margin, 1.0e-9)
                    )
                    rows.append(projected_jacobian)
                    lower.append(-np.inf)
                    upper.append(-float(repulsion))
                    contact_rows += 1
                elif plane.clearance < self.near_distance:
                    hessian += self.near_weight * np.outer(
                        projected_jacobian, projected_jacobian
                    )
                    near_terms += 1

        smoothness_weight = 0.04
        hessian += smoothness_weight * np.eye(self.nv)
        gradient -= smoothness_weight * self.previous_velocity
        matrix = sparse.csc_matrix(np.vstack(rows))
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.triu(sparse.csc_matrix(hessian), format="csc"),
            q=gradient,
            A=matrix,
            l=np.asarray(lower),
            u=np.asarray(upper),
            verbose=False,
            polishing=False,
            warm_starting=True,
            eps_abs=2.0e-5,
            eps_rel=2.0e-5,
            max_iter=6000,
        )
        if np.any(self.previous_velocity):
            solver.warm_start(x=self.previous_velocity)
        started = time.perf_counter()
        result = solver.solve(raise_error=False)
        solve_ms = (time.perf_counter() - started) * 1000.0
        status = result.info.status.lower()
        if result.x is None or not status.startswith("solved"):
            qdot = np.zeros(self.nv)
        else:
            qdot = np.clip(np.asarray(result.x, dtype=float), qdot_lower, qdot_upper)
        self.previous_velocity = qdot.copy()

        raw_count = len(self.robot_ellipsoids) * len(self.obstacle_centers)
        return qdot, EllipsoidStepMetrics(
            status=status,
            solve_ms=float(solve_ms),
            ee_error=float(np.linalg.norm(np.asarray(target) - ee)),
            task_speed=float(np.linalg.norm(desired_velocity)),
            qdot_norm=float(np.linalg.norm(qdot)),
            min_clearance=float(minimum_clearance),
            raw_obstacle_proxies=int(raw_count),
            broadphase_candidate_pairs=int(candidate_pairs),
            active_obstacle_rows=int(active_rows),
            redundant_proxies_removed=int(raw_count - active_rows),
            near_penalty_terms=int(near_terms),
            contact_repulsion_rows=int(contact_rows),
            workspace_rows=int(workspace_rows),
            closest_point_newton_iterations=int(self._closest_newton_iterations),
            closest_point_bisection_iterations=int(
                self._closest_bisection_iterations
            ),
            support_projected_ascent_iterations=int(
                self._support_projected_iterations
            ),
            limiting_robot_index=int(limiting_robot_index),
            limiting_obstacle_index=int(limiting_obstacle_index),
        )
