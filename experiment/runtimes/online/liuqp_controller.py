"""Sequential LiuQP controller with equation-by-equation traceability.

Notation follows Liu and Yim (2021):

* qdot is the paper's joint-rate vector ``dot(Theta)``;
* p_i and J_i are a robot certificate-sphere center and its Jacobian;
* o_j, r_j are an obstacle sphere center and radius;
* s_ij points from p_i toward o_j;
* P_ij is the obstacle-sphere tangent separating plane.

The paper writes one-step constraints with an implicit short control interval.
This implementation keeps the dimensions explicit by using
``p_next ~= p_i + J_i qdot dt``.  Setting dt=1 recovers the printed form of
Eqs. (22), (23), and (27).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import mujoco
import numpy as np
import osqp
from scipy import sparse

from model import (
    DT,
    JOINT_NAMES,
    CertificateSphere,
    SceneDefinition,
    attachment_position,
    certificate_world_state,
)
from multilevel_voxel_table import MultilevelVoxelTable


@dataclass(frozen=True)
class SeparatingPlane:
    """Paper Eq. (23): tangent plane for one robot/obstacle sphere pair."""

    obstacle_index: int
    normal_to_obstacle: np.ndarray  # s_ij = (o_j - p_i) / ||o_j - p_i||
    tangent_point: np.ndarray  # o'_j = o_j - r_j s_ij
    offset: float  # b_ij = s_ij^T o'_j
    clearance: float  # ||o_j-p_i|| - r_i - r_j - safety_margin


@dataclass
class StepMetrics:
    status: str
    solve_ms: float
    ee_error: float
    task_speed: float
    qdot_norm: float
    min_clearance: float
    raw_obstacle_spheres: int
    broadphase_candidate_pairs: int
    active_obstacle_rows: int
    redundant_spheres_removed: int
    near_penalty_terms: int
    contact_repulsion_rows: int
    workspace_rows: int

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


class LiuQPController:
    """One linearly constrained convex QP per MuJoCo control step."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        scene: SceneDefinition,
        robot_spheres: list[CertificateSphere],
        obstacle_centers: np.ndarray,
        obstacle_radii: np.ndarray,
        *,
        safety_margin: float = 0.006,
        near_distance: float = 0.12,
        contact_distance: float = 0.002,
        obstacle_index: MultilevelVoxelTable | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.scene = scene
        self.robot_spheres = robot_spheres
        self.obstacle_centers = np.asarray(obstacle_centers, dtype=float)
        self.obstacle_radii = np.asarray(obstacle_radii, dtype=float)
        self.safety_margin = float(safety_margin)
        self.near_distance = float(near_distance)
        self.contact_distance = float(contact_distance)
        self.obstacle_index = obstacle_index
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
        self.task_region_A: np.ndarray | None = None
        self.task_region_b: np.ndarray | None = None
        self.posture_target: np.ndarray | None = None
        self.posture_weight = 80.0
        self.posture_gain = 2.0

        # Paper Eq. (17) feedback controller parameters.
        self.task_gain = 2.0
        self.max_task_speed = 0.18

        # Paper Eqs. (25)-(26) objective weights lambda and mu_ij.
        self.task_weight = 90.0
        self.near_weight = 22.0
        self.velocity_regularization = 1.0

        # Paper Eq. (27) repulsive velocity magnitude (-gamma_ij > 0).
        self.repulsive_speed = 0.035

        # Paper Eq. (20): conservative UR5e joint-rate limits for this demo.
        self.velocity_limits = np.array([1.0, 1.0, 1.0, 1.4, 1.4, 1.4])
        self.joint_min = np.array(
            [model.jnt_range[model.joint(name).id, 0] for name in JOINT_NAMES]
        )
        self.joint_max = np.array(
            [model.jnt_range[model.joint(name).id, 1] for name in JOINT_NAMES]
        )
        self.joint_padding = 0.015

        # Paper Eq. (22): polyhedral workspace a^T p <= b.  The floor is
        # z>=0; remaining faces are a loose numerical guard around the scene.
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

    # ------------------------------------------------------------------
    # Paper Eq. (17): J_P qdot = V_tilde + K(P_tilde - P).
    # Eq. (25) places this feedback law in a soft least-squares objective so
    # safety constraints retain priority when exact tracking is infeasible.
    # ------------------------------------------------------------------
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
        """Keep a drawer retrieval tool aligned while preserving soft priority."""

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

    # ------------------------------------------------------------------
    # Paper Eqs. (19)-(20): position-implied and actuator velocity bounds.
    # ------------------------------------------------------------------
    def joint_velocity_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        q = self.data.qpos[: self.nv]
        position_lower = (self.joint_min + self.joint_padding - q) / self.dt
        position_upper = (self.joint_max - self.joint_padding - q) / self.dt
        lower = np.maximum(-self.velocity_limits, position_lower)
        upper = np.minimum(self.velocity_limits, position_upper)
        return lower, upper

    # ------------------------------------------------------------------
    # Paper Eq. (23): construct the fast, closed-form tangent hyperplane.
    # ------------------------------------------------------------------
    def separating_plane(
        self,
        robot_center: np.ndarray,
        robot_radius: float,
        obstacle_index: int,
    ) -> SeparatingPlane:
        obstacle_center = self.obstacle_centers[obstacle_index]
        obstacle_radius = float(self.obstacle_radii[obstacle_index])
        delta = obstacle_center - robot_center
        distance = float(np.linalg.norm(delta))
        if distance < 1.0e-12:
            # A deterministic fallback keeps the QP finite for a pathological
            # coincident-center initialization.
            normal = np.array([1.0, 0.0, 0.0])
        else:
            normal = delta / distance
        tangent = obstacle_center - obstacle_radius * normal
        offset = float(normal @ tangent)
        clearance = distance - robot_radius - obstacle_radius - self.safety_margin
        return SeparatingPlane(
            obstacle_index=obstacle_index,
            normal_to_obstacle=normal,
            tangent_point=tangent,
            offset=offset,
            clearance=float(clearance),
        )

    # ------------------------------------------------------------------
    # LiuQP Sec. 4.2 / Fig. 9: delete redundant obstacle spheres.
    # ------------------------------------------------------------------
    def prune_redundant_obstacle_spheres(
        self,
        robot_center: np.ndarray,
        robot_radius: float,
    ) -> list[SeparatingPlane]:
        """Return S_i after the paper's tangent-plane erase/remove rule.

        For a kept sphere j, its near tangent plane is

            P_ij: s_ij^T x = b_ij,
            b_ij = s_ij^T(o_j - r_j s_ij).

        A sphere k lies completely behind that plane when

            min_{x in ball(o_k,r_k)} s_ij^T x
              = s_ij^T o_k - r_k >= b_ij.

        In that case, any robot center that remains on the safe side of P_ij
        cannot reach sphere k, so k is removed for this robot sphere and step.
        The nearest remaining sphere is selected first, matching Fig. 9's
        occlusion interpretation.
        """

        if not len(self.obstacle_centers):
            self._last_prune_candidate_count = 0
            return []
        if self.obstacle_index is None:
            candidates = np.arange(len(self.obstacle_centers), dtype=int)
        else:
            candidates = self.obstacle_index.query_sphere(robot_center, robot_radius)
        self._last_prune_candidate_count = len(candidates)
        if not len(candidates):
            return []
        center_distance = np.linalg.norm(
            self.obstacle_centers[candidates] - robot_center, axis=1
        )
        surface_gap = (
            center_distance - self.obstacle_radii[candidates] - robot_radius
        )
        if self.obstacle_index is not None:
            keep = surface_gap <= self.near_distance
            candidates = candidates[keep]
            surface_gap = surface_gap[keep]
            if not len(candidates):
                return []
        remaining = list(candidates[np.argsort(surface_gap)])
        active: list[SeparatingPlane] = []
        tolerance = 1.0e-10
        while remaining:
            nearest = int(remaining[0])
            plane = self.separating_plane(robot_center, robot_radius, nearest)
            active.append(plane)
            projection_min = (
                self.obstacle_centers[remaining] @ plane.normal_to_obstacle
                - self.obstacle_radii[remaining]
            )
            # Erase the kept sphere itself and every sphere fully behind P_ij.
            behind = projection_min >= plane.offset - tolerance
            remaining = [index for index, remove in zip(remaining, behind) if not remove]
        return active

    # ------------------------------------------------------------------
    # Paper Eq. (22): module sphere must remain inside every workspace face.
    # a^T(p_i + J_i qdot dt) <= b - r_i.
    # ------------------------------------------------------------------
    def add_workspace_constraints(
        self,
        rows: list[np.ndarray],
        lower: list[float],
        upper: list[float],
        position: np.ndarray,
        jacobian: np.ndarray,
        radius: float,
    ) -> int:
        for normal, offset in zip(self.workspace_A, self.workspace_b):
            rows.append(normal @ jacobian)
            lower.append(-np.inf)
            upper.append(float((offset - radius - normal @ position) / self.dt))
        return len(self.workspace_A)

    def solve(self, target: np.ndarray) -> tuple[np.ndarray, StepMetrics]:
        """Assemble and solve Eqs. (25)-(27) for the current MuJoCo state."""

        ee, ee_jacobian, desired_velocity = self.task_feedback(target)
        positions, jacobians_full, robot_radii = certificate_world_state(
            self.model, self.data, self.robot_spheres
        )
        jacobians = jacobians_full[:, :, : self.nv]

        # Paper Eq. (25): ||qdot||^2 + lambda ||J_P qdot-v_des||^2.
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

        # Paper Eqs. (19)-(20).
        qdot_lower, qdot_upper = self.joint_velocity_bounds()
        for joint in range(self.nv):
            row = np.zeros(self.nv)
            row[joint] = 1.0
            rows.append(row)
            lower.append(float(qdot_lower[joint]))
            upper.append(float(qdot_upper[joint]))

        workspace_rows = 0
        if self.task_region_A is not None:
            region_rows = self.task_region_A @ ee_jacobian
            region_bounds = (
                self.task_region_b - self.task_region_A @ ee
            ) / self.dt
            rows.extend(region_rows)
            lower.extend(np.full(len(region_rows), -np.inf))
            upper.extend(region_bounds)
            workspace_rows += len(region_rows)
        active_rows = 0
        candidate_pairs = 0
        near_terms = 0
        contact_rows = 0
        minimum_clearance = np.inf
        for position, jacobian, radius in zip(positions, jacobians, robot_radii):
            workspace_rows += self.add_workspace_constraints(
                rows, lower, upper, position, jacobian, float(radius)
            )
            active_planes = self.prune_redundant_obstacle_spheres(
                position, float(radius)
            )
            candidate_pairs += self._last_prune_candidate_count
            for plane in active_planes:
                normal = plane.normal_to_obstacle
                projected_jacobian = normal @ jacobian
                minimum_clearance = min(minimum_clearance, plane.clearance)

                # Paper Eq. (23), dimensionally explicit:
                # s^T[p_i + J_i qdot dt] <= b_ij-r_i-margin.
                rows.append(projected_jacobian)
                lower.append(-np.inf)
                upper.append(float(plane.clearance / self.dt))
                active_rows += 1

                if plane.clearance <= self.contact_distance:
                    # Paper Eq. (27): s^T J_i qdot <= gamma_ij <= 0.
                    # The Eq. (26) term is deliberately omitted for this pair.
                    penetration = max(0.0, -plane.clearance)
                    repulsion = self.repulsive_speed * min(
                        1.0, 0.25 + penetration / max(self.safety_margin, 1.0e-9)
                    )
                    rows.append(projected_jacobian)
                    lower.append(-np.inf)
                    upper.append(-float(repulsion))
                    contact_rows += 1
                elif plane.clearance < self.near_distance:
                    # Paper Eq. (26): +mu_ij ||v_i . s_ij||^2.
                    hessian += self.near_weight * np.outer(
                        projected_jacobian, projected_jacobian
                    )
                    near_terms += 1

        # A tiny smoothness term is a numerical tie-breaker for redundant UR5
        # joints.  It does not change any hard LiuQP constraint.
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
            qdot = np.asarray(result.x, dtype=float)
            qdot = np.clip(qdot, qdot_lower, qdot_upper)
        self.previous_velocity = qdot.copy()

        raw_count = len(self.robot_spheres) * len(self.obstacle_centers)
        return qdot, StepMetrics(
            status=status,
            solve_ms=float(solve_ms),
            ee_error=float(np.linalg.norm(np.asarray(target) - ee)),
            task_speed=float(np.linalg.norm(desired_velocity)),
            qdot_norm=float(np.linalg.norm(qdot)),
            min_clearance=float(minimum_clearance),
            raw_obstacle_spheres=int(raw_count),
            broadphase_candidate_pairs=int(candidate_pairs),
            active_obstacle_rows=int(active_rows),
            redundant_spheres_removed=int(raw_count - active_rows),
            near_penalty_terms=int(near_terms),
            contact_repulsion_rows=int(contact_rows),
            workspace_rows=int(workspace_rows),
        )
