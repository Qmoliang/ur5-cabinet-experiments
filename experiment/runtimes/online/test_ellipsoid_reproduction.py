"""Numerical checks for the ellipsoid-support LiuQP extension."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from drawer_scene import drawer_scene
from ellipsoid_model import (
    build_robot_ellipsoid_certificate,
    closest_point_on_ellipsoid,
    closest_point_normal_on_ellipsoid,
    ellipsoid_world_state,
    optimal_support_separating_normal,
    sample_box_ellipsoid_tree,
    support_angular_gradient,
    support_radius,
)
from ellipsoid_qp_controller import EllipsoidLiuQPController
from model import build_model, set_configuration
from model import build_robot_certificate, sample_box_sphere_tree
from run_simulation import conservative_environment_clearance


class EllipsoidCertificateTests(unittest.TestCase):
    def test_safeguarded_newton_is_accurate_and_warm_starts(self) -> None:
        center = np.zeros(3)
        shape = np.diag(np.array([0.20, 0.025, 0.05]) ** 2)
        point = np.array([-0.25, 0.08, 0.02])
        cold = closest_point_on_ellipsoid(center, shape, point)
        warm = closest_point_on_ellipsoid(
            center,
            shape,
            point,
            initial_multiplier=cold.multiplier,
        )
        self.assertLess(cold.residual, 1.0e-10)
        self.assertLess(warm.residual, 1.0e-10)
        self.assertLessEqual(warm.newton_iterations, cold.newton_iterations)
        self.assertLessEqual(warm.bisection_iterations, cold.bisection_iterations)
        np.testing.assert_allclose(warm.normal, cold.normal, atol=1.0e-12)

    def test_general_optimal_support_normal_beats_centerline(self) -> None:
        robot_center = np.array([-0.12, 0.04, 0.02])
        obstacle_center = np.array([0.10, -0.03, 0.01])
        robot_shape = np.diag(np.array([0.03, 0.09, 0.04]) ** 2)
        obstacle_shape = np.diag(np.array([0.14, 0.025, 0.06]) ** 2)
        normal = optimal_support_separating_normal(
            robot_center, robot_shape, obstacle_center, obstacle_shape
        )
        delta = obstacle_center - robot_center
        centerline = delta / np.linalg.norm(delta)
        def gap(direction: np.ndarray) -> float:
            return float(
                direction @ delta
                - support_radius(robot_shape, direction)
                - support_radius(obstacle_shape, direction)
            )
        self.assertGreaterEqual(gap(normal), gap(centerline) - 1.0e-10)
        self.assertAlmostEqual(np.linalg.norm(normal), 1.0, places=12)

    def test_closest_point_normal_improves_anisotropic_support_gap(self) -> None:
        center = np.zeros(3)
        shape = np.diag(np.array([0.20, 0.025, 0.05]) ** 2)
        point = np.array([-0.25, 0.08, 0.02])
        optimal = closest_point_normal_on_ellipsoid(center, shape, point)
        centerline = (center - point) / np.linalg.norm(center - point)
        delta = center - point
        optimal_gap = float(optimal @ delta - support_radius(shape, optimal))
        centerline_gap = float(centerline @ delta - support_radius(shape, centerline))
        self.assertGreaterEqual(optimal_gap, centerline_gap - 1.0e-10)
        self.assertAlmostEqual(np.linalg.norm(optimal), 1.0, places=12)

    def test_closest_point_normal_reduces_to_centerline_for_sphere(self) -> None:
        center = np.array([0.2, -0.1, 0.3])
        point = np.array([-0.4, 0.25, -0.2])
        shape = np.eye(3) * 0.07**2
        normal = closest_point_normal_on_ellipsoid(center, shape, point)
        expected = (center - point) / np.linalg.norm(center - point)
        np.testing.assert_allclose(normal, expected, atol=1.0e-11)

    def test_drawer_start_is_valid_for_sphere_baseline_and_exact_geometry(self) -> None:
        scene = drawer_scene()
        model = build_model(scene)
        data = mujoco.MjData(model)
        set_configuration(model, data, np.asarray(scene.q0))
        robot = build_robot_certificate(model)
        centers, radii, _ = sample_box_sphere_tree(scene.boxes)
        clearance = conservative_environment_clearance(
            model, data, robot, centers, radii, 0.006
        )
        self.assertGreater(clearance, 0.0)
        self.assertEqual(data.ncon, 0)

    def test_box_cell_corners_are_inside_enclosing_ellipsoid(self) -> None:
        scene = drawer_scene()
        centers, shapes, _ = sample_box_ellipsoid_tree((scene.boxes[0],), cell_size=1.0)
        self.assertEqual(len(centers), 1)
        half = np.asarray(scene.boxes[0].half_size)
        inverse = np.linalg.inv(shapes[0])
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    offset = half * np.array([sx, sy, sz])
                    self.assertLessEqual(float(offset @ inverse @ offset), 1.0 + 1.0e-12)

    def test_support_angular_gradient_matches_finite_difference(self) -> None:
        axes = np.array([0.08, 0.035, 0.16])
        rotation = Rotation.from_euler("xyz", [0.3, -0.2, 0.5]).as_matrix()
        shape = rotation @ np.diag(axes**2) @ rotation.T
        normal = np.array([0.4, -0.7, 0.2])
        normal /= np.linalg.norm(normal)
        omega = np.array([0.6, -0.1, 0.35])
        epsilon = 1.0e-7
        delta = Rotation.from_rotvec(omega * epsilon).as_matrix()
        plus = support_radius(delta @ shape @ delta.T, normal)
        delta_minus = Rotation.from_rotvec(-omega * epsilon).as_matrix()
        minus = support_radius(delta_minus @ shape @ delta_minus.T, normal)
        finite = (plus - minus) / (2.0 * epsilon)
        analytic = float(support_angular_gradient(shape, normal) @ omega)
        self.assertAlmostEqual(analytic, finite, places=7)

    def test_world_shapes_are_positive_definite_and_finite(self) -> None:
        scene = drawer_scene()
        model = build_model(scene)
        data = mujoco.MjData(model)
        set_configuration(model, data, np.asarray(scene.q0))
        certificate = build_robot_ellipsoid_certificate(model)
        positions, jac_pos, jac_rot, shapes = ellipsoid_world_state(model, data, certificate)
        self.assertGreater(len(certificate), 40)
        self.assertTrue(np.all(np.isfinite(positions)))
        self.assertTrue(np.all(np.isfinite(jac_pos)))
        self.assertTrue(np.all(np.isfinite(jac_rot)))
        self.assertTrue(np.all(np.linalg.eigvalsh(shapes) > 0.0))

    def test_first_drawer_qp_is_solved(self) -> None:
        scene = drawer_scene()
        model = build_model(scene)
        data = mujoco.MjData(model)
        set_configuration(model, data, np.asarray(scene.q0))
        robot = build_robot_ellipsoid_certificate(model)
        centers, shapes, _ = sample_box_ellipsoid_tree(scene.boxes)
        controller = EllipsoidLiuQPController(model, data, scene, robot, centers, shapes)
        qdot, metrics = controller.solve(np.asarray(scene.waypoints[-1]))
        self.assertTrue(metrics.status.startswith("solved"), metrics.status)
        self.assertTrue(np.all(np.isfinite(qdot)))


if __name__ == "__main__":
    unittest.main()
