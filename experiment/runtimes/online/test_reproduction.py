"""Numerical checks for the UR5e LiuQP reproduction."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from liuqp_controller import LiuQPController
from model import (
    DT,
    JOINT_NAMES,
    SCENES,
    attachment_position,
    build_model,
    build_robot_certificate,
    certificate_world_state,
    sample_box_sphere_tree,
    set_configuration,
)


class LiuQPReproductionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = SCENES["shelf"]
        self.model = build_model(self.scene)
        self.data = mujoco.MjData(self.model)
        set_configuration(self.model, self.data, np.asarray(self.scene.q0))
        self.certificate = build_robot_certificate(self.model)
        centers, radii, _ = sample_box_sphere_tree(self.scene.boxes)
        self.controller = LiuQPController(
            self.model,
            self.data,
            self.scene,
            self.certificate,
            centers,
            radii,
        )

    def test_official_ur5e_joint_tree_is_preserved(self) -> None:
        self.assertEqual(self.model.nq, 6)
        self.assertEqual(self.model.nv, 6)
        self.assertEqual(
            tuple(self.model.joint(index).name for index in range(self.model.njnt)),
            JOINT_NAMES,
        )
        self.assertEqual(self.model.body("wrist_3_link").parentid, self.model.body("wrist_2_link").id)

    def test_attachment_jacobian_matches_centered_finite_difference(self) -> None:
        q0 = self.data.qpos.copy()
        analytic = np.zeros((3, self.model.nv))
        rotational = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model,
            self.data,
            analytic,
            rotational,
            self.model.site("attachment_site").id,
        )
        finite = np.zeros_like(analytic)
        epsilon = 1.0e-6
        for joint in range(self.model.nv):
            direction = np.zeros(self.model.nv)
            direction[joint] = 1.0
            q_plus = q0.copy()
            q_minus = q0.copy()
            mujoco.mj_integratePos(self.model, q_plus, direction, epsilon)
            mujoco.mj_integratePos(self.model, q_minus, direction, -epsilon)
            set_configuration(self.model, self.data, q_plus)
            plus = attachment_position(self.model, self.data)
            set_configuration(self.model, self.data, q_minus)
            minus = attachment_position(self.model, self.data)
            finite[:, joint] = (plus - minus) / (2.0 * epsilon)
        set_configuration(self.model, self.data, q0)
        np.testing.assert_allclose(analytic[:, :6], finite[:, :6], atol=2.0e-6, rtol=2.0e-5)

    def test_sphere_certificate_is_nonempty_and_finite(self) -> None:
        positions, jacobians, radii = certificate_world_state(
            self.model, self.data, self.certificate
        )
        self.assertGreater(len(self.certificate), 40)
        self.assertTrue(np.all(np.isfinite(positions)))
        self.assertTrue(np.all(np.isfinite(jacobians)))
        self.assertTrue(np.all(radii > 0.0))

    def test_redundancy_rule_covers_every_erased_sphere(self) -> None:
        positions, _, robot_radii = certificate_world_state(
            self.model, self.data, self.certificate
        )
        robot_center = positions[len(positions) // 2]
        robot_radius = float(robot_radii[len(robot_radii) // 2])
        active = self.controller.prune_redundant_obstacle_spheres(
            robot_center, robot_radius
        )
        active_indices = {plane.obstacle_index for plane in active}
        self.assertLess(len(active), len(self.controller.obstacle_centers))
        for obstacle_index, (center, radius) in enumerate(
            zip(self.controller.obstacle_centers, self.controller.obstacle_radii)
        ):
            if obstacle_index in active_indices:
                continue
            separated = any(
                plane.normal_to_obstacle @ center - radius >= plane.offset - 1.0e-9
                for plane in active
            )
            self.assertTrue(separated, f"erased sphere {obstacle_index} lacks a separating plane")

    def test_one_qp_step_satisfies_joint_and_linearized_obstacle_constraints(self) -> None:
        qdot, metrics = self.controller.solve(np.asarray(self.scene.waypoints[0]))
        self.assertTrue(metrics.status.startswith("solved"))
        lower, upper = self.controller.joint_velocity_bounds()
        self.assertTrue(np.all(qdot >= lower - 1.0e-5))
        self.assertTrue(np.all(qdot <= upper + 1.0e-5))

        positions, jacobians, radii = certificate_world_state(
            self.model, self.data, self.certificate
        )
        worst = -np.inf
        for position, jacobian, radius in zip(positions, jacobians, radii):
            for plane in self.controller.prune_redundant_obstacle_spheres(
                position, float(radius)
            ):
                predicted = plane.normal_to_obstacle @ (position + jacobian[:, :6] @ qdot * DT)
                allowed = plane.offset - radius - self.controller.safety_margin
                worst = max(worst, float(predicted - allowed))
        self.assertLessEqual(worst, 5.0e-5)


if __name__ == "__main__":
    unittest.main()

