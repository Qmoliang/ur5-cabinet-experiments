"""Protocol-v3 invariants that must hold before formal experiment runs."""

from __future__ import annotations

import unittest
import hashlib
from unittest import mock
from types import SimpleNamespace

import mujoco
import numpy as np

from ellipsoid_model import closest_point_on_ellipsoid
from model import (
    attachment_position,
    build_model,
    build_robot_certificate,
    certificate_world_state,
    set_configuration,
)
from protocol_drawer_scene import (
    CONTACT_DISTANCE,
    DRAWER_Z,
    FRONT_X,
    KNOWN_PROXY_CELL_SIZE,
    NEAR_DISTANCE,
    SAFETY_MARGIN,
    formal_protocol_scene,
    protocol_drawer_v9_scene,
    protocol_drawer_v10_scene,
    protocol_drawer_v11_scene,
    protocol_drawer_v6s2_scene,
    protocol_drawer_v6s3_scene,
    protocol_drawer_v6s4_scene,
)
from protocol_known_proxies import (
    audit_known_proxy_cell_coverage,
    build_known_matched_proxy_tree,
)
from protocol_liuqp_controller import (
    ObstacleState,
    ProtocolLiuQPController,
    classify_clearance,
)
from run_protocol_v3_online_ablation import _build_mvt_and_audit, _build_mvt_only


class ProtocolV3Tests(unittest.TestCase):
    def test_streamed_qp_row_hash_matches_frozen_concatenation(self) -> None:
        rng = np.random.default_rng(1701)
        matrix = np.ascontiguousarray(rng.normal(size=(37, 6)))
        lower = np.ascontiguousarray(rng.normal(size=37))
        upper = np.ascontiguousarray(rng.normal(size=37))
        frozen_payload = np.round(matrix, 12).tobytes()
        frozen_payload += np.round(lower, 12).tobytes()
        frozen_payload += np.round(upper, 12).tobytes()
        expected = hashlib.sha256(frozen_payload).hexdigest()
        self.assertEqual(
            ProtocolLiuQPController._rows_hash(matrix, lower, upper),
            expected,
        )

    def test_osqp_dual_warm_start_is_remapped_by_semantic_row_identity(
        self,
    ) -> None:
        previous_identity = (
            ("joint_velocity", 0, -1),
            ("workspace", 4, 2),
            ("collision", 7, 101),
            ("collision", 7, 202),
        )
        previous_dual = np.array([1.0, 2.0, 3.0, 4.0, 99.0])
        current_identity = (
            ("joint_velocity", 0, -1),
            ("collision", 7, 202),
            ("collision", 7, 303),
            ("workspace", 4, 2),
        )
        mapped, reused = (
            ProtocolLiuQPController._remap_dual_by_row_identity(
                previous_identity,
                previous_dual,
                current_identity,
                capacity=7,
            )
        )
        self.assertEqual(reused, 3)
        np.testing.assert_allclose(
            mapped,
            np.array([1.0, 4.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
        )

    def test_formal_mvt_outward_rounding_has_no_float64_boundary_misses(self):
        robot_radius = 0.03
        query_radius = robot_radius + NEAR_DISTANCE + SAFETY_MARGIN
        half = np.array(
            [[0.007000003, 0.003000007, 0.002000009]] * 129, dtype=float
        )
        centers = np.zeros((len(half), 3), dtype=float)
        perturbations = np.linspace(-2.0e-9, 0.0, len(half))
        centers[:, 0] = query_radius + half[:, 0] + perturbations
        shapes = np.asarray([np.diag(item**2) for item in half])
        proxies = SimpleNamespace(
            centers=centers,
            proxy_offset_radii=np.zeros(len(centers)),
            proxy_uncertainty_shapes=np.zeros_like(shapes),
            ellipsoid_outer_shapes=shapes,
            base_ellipsoid_shapes=shapes,
        )
        table, missing = _build_mvt_and_audit(
            proxies,
            "ellipsoid",
            np.zeros((1, 3)),
            np.array([robot_radius]),
            simd=True,
        )
        self.addCleanup(table.close)
        self.assertEqual(missing, 0)

    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = formal_protocol_scene()
        cls.model = build_model(cls.scene)
        cls.data = mujoco.MjData(cls.model)
        cls.robot = build_robot_certificate(cls.model)
        cls.proxies = build_known_matched_proxy_tree(
            cls.scene.boxes,
            cell_size=KNOWN_PROXY_CELL_SIZE,
        )

    def setUp(self) -> None:
        set_configuration(self.model, self.data, np.asarray(self.scene.q0))

    def _controller(self, representation: str, **options) -> ProtocolLiuQPController:
        common = dict(
            model=self.model,
            data=self.data,
            scene=self.scene,
            robot_spheres=self.robot,
            obstacle_centers=self.proxies.centers,
            representation=representation,
            proxy_ids=self.proxies.proxy_ids,
            safety_margin=SAFETY_MARGIN,
            near_distance=NEAR_DISTANCE,
            contact_distance=CONTACT_DISTANCE,
        )
        if representation == "sphere":
            return ProtocolLiuQPController(
                **common,
                obstacle_radii=self.proxies.sphere_radii,
                **options,
            )
        return ProtocolLiuQPController(
            **common,
            obstacle_shapes=self.proxies.ellipsoid_shapes,
            **options,
        )

    def test_scene_has_two_physical_moving_views_and_no_tool(self) -> None:
        self.assertEqual(self.model.ncam, 2)
        camera_names = {
            self.model.camera(index).name for index in range(self.model.ncam)
        }
        self.assertEqual(
            camera_names,
            {
                "ur5_depth_wrist",
                "ur5_depth_forearm",
            },
        )
        names = {self.model.geom(index).name for index in range(self.model.ngeom)}
        self.assertIn("formal_d405_wrist_housing", names)
        self.assertIn("formal_d405_forearm_housing", names)
        self.assertEqual(sum("d405" in name.lower() for name in names), 2)
        np.testing.assert_allclose(self.model.cam_fovy, [58.0, 58.0])
        forbidden = ("gripper", "tool", "probe")
        self.assertFalse(any(any(token in name for token in forbidden) for name in names))

    def test_initial_robot_is_outside_below_and_collision_free(self) -> None:
        tip = attachment_position(self.model, self.data)
        centers, _, _ = certificate_world_state(
            self.model,
            self.data,
            self.robot,
        )
        self.assertLess(tip[0], FRONT_X)
        self.assertLess(tip[2], DRAWER_Z - 0.09)
        self.assertLess(float(np.max(centers[:, 0])), FRONT_X)
        self.assertEqual(self.data.ncon, 0)

    def test_v9_outward_camera_candidate_is_collision_free(self) -> None:
        scene = protocol_drawer_v9_scene()
        model = build_model(scene)
        data = mujoco.MjData(model)
        set_configuration(model, data, np.asarray(scene.q0))
        self.assertEqual(data.ncon, 0)
        left = model.camera("ur5_depth_wrist").id
        right = model.camera("ur5_depth_wrist_right").id
        left_position = data.cam_xpos[left]
        right_position = data.cam_xpos[right]
        left_optical = -data.cam_xmat[left].reshape(3, 3)[:, 2]
        right_optical = -data.cam_xmat[right].reshape(3, 3)[:, 2]
        self.assertLess(float(left_optical @ (right_position - left_position)), 0.0)
        self.assertLess(float(right_optical @ (left_position - right_position)), 0.0)

    def test_v10_parallel_camera_candidate_is_collision_free(self) -> None:
        scene = protocol_drawer_v10_scene()
        model = build_model(scene)
        data = mujoco.MjData(model)
        set_configuration(model, data, np.asarray(scene.q0))
        self.assertEqual(data.ncon, 0)
        left = model.camera("ur5_depth_wrist").id
        right = model.camera("ur5_depth_wrist_right").id
        left_optical = -data.cam_xmat[left].reshape(3, 3)[:, 2]
        right_optical = -data.cam_xmat[right].reshape(3, 3)[:, 2]
        np.testing.assert_allclose(left_optical, right_optical, atol=1.0e-12)

    def test_v11_camera_optical_centers_are_on_housing_front_faces(self) -> None:
        scene = protocol_drawer_v11_scene()
        model = build_model(scene)
        data = mujoco.MjData(model)
        set_configuration(model, data, np.asarray(scene.q0))
        self.assertEqual(data.ncon, 0)
        pairs = (
            ("ur5_depth_wrist", "protocol_d405_camera_housing"),
            (
                "ur5_depth_wrist_right",
                "protocol_d405_camera_housing_right",
            ),
        )
        for camera_name, housing_name in pairs:
            camera_id = model.camera(camera_name).id
            housing_id = model.geom(housing_name).id
            optical = -data.cam_xmat[camera_id].reshape(3, 3)[:, 2]
            offset = data.cam_xpos[camera_id] - data.geom_xpos[housing_id]
            axial = float(optical @ offset)
            lateral = offset - axial * optical
            self.assertAlmostEqual(axial, 0.011575, places=9)
            np.testing.assert_allclose(lateral, np.zeros(3), atol=1.0e-10)

    def test_v6s2_mirrored_shoulder_camera_is_physical_and_collision_free(self) -> None:
        scene = protocol_drawer_v6s2_scene()
        model = build_model(scene)
        data = mujoco.MjData(model)
        set_configuration(model, data, np.asarray(scene.q0))
        self.assertEqual(data.ncon, 0)
        self.assertEqual(model.ncam, 4)
        self.assertGreaterEqual(
            model.geom("protocol_d405_shoulder_housing_right").id, 0
        )

    def test_v6s3_parallel_shoulder_pair_is_collision_free(self) -> None:
        scene = protocol_drawer_v6s3_scene()
        model = build_model(scene)
        data = mujoco.MjData(model)
        set_configuration(model, data, np.asarray(scene.q0))
        self.assertEqual(data.ncon, 0)
        left = model.camera("ur5_depth_shoulder").id
        right = model.camera("ur5_depth_shoulder_right").id
        left_optical = -data.cam_xmat[left].reshape(3, 3)[:, 2]
        right_optical = -data.cam_xmat[right].reshape(3, 3)[:, 2]
        np.testing.assert_allclose(left_optical, right_optical, atol=1.0e-12)

    def test_v6s4_second_shoulder_lens_is_just_ahead_of_front_face(self) -> None:
        scene = protocol_drawer_v6s4_scene()
        model = build_model(scene)
        data = mujoco.MjData(model)
        set_configuration(model, data, np.asarray(scene.q0))
        self.assertEqual(data.ncon, 0)
        camera_id = model.camera("ur5_depth_shoulder_right").id
        housing_id = model.geom("protocol_d405_shoulder_housing_right").id
        optical = -data.cam_xmat[camera_id].reshape(3, 3)[:, 2]
        offset = data.cam_xpos[camera_id] - data.geom_xpos[housing_id]
        axial = float(optical @ offset)
        lateral = offset - axial * optical
        # Match the primary shoulder sensor: 1 mm lens protrusion beyond the
        # 11.575 mm housing half-thickness, never inside the opaque housing.
        self.assertAlmostEqual(axial, 0.012575, places=7)
        np.testing.assert_allclose(lateral, np.zeros(3), atol=1.0e-8)

    def test_known_sphere_and_ellipsoid_cells_are_matched_and_cover(self) -> None:
        audit = audit_known_proxy_cell_coverage(self.proxies)
        self.assertTrue(audit["matched_centers_and_count"])
        self.assertTrue(audit["sphere_covers_all_cells"])
        self.assertTrue(audit["ellipsoid_covers_all_cells"])
        self.assertEqual(len(self.proxies.proxy_ids), len(np.unique(self.proxies.proxy_ids)))
        self.assertEqual(self.proxies.centers.shape[0], 200)

    def test_complete_three_state_boundaries(self) -> None:
        kwargs = dict(
            near_distance=NEAR_DISTANCE,
            contact_distance=CONTACT_DISTANCE,
        )
        self.assertIs(
            classify_clearance(NEAR_DISTANCE, **kwargs),
            ObstacleState.NORMAL,
        )
        self.assertIs(
            classify_clearance(np.nextafter(NEAR_DISTANCE, -np.inf), **kwargs),
            ObstacleState.NEAR,
        )
        self.assertIs(
            classify_clearance(np.nextafter(CONTACT_DISTANCE, np.inf), **kwargs),
            ObstacleState.NEAR,
        )
        self.assertIs(
            classify_clearance(CONTACT_DISTANCE, **kwargs),
            ObstacleState.CONTACT_RECOVERY,
        )
        # A positive raw surface gap remains NEAR even when it lies inside the
        # independent 6 mm hard safety margin (h = c - d_safe < 0).
        self.assertIs(
            classify_clearance(0.0019566, **kwargs),
            ObstacleState.NEAR,
        )

    def test_three_states_create_the_required_mutually_exclusive_terms(self) -> None:
        robot = [self.robot[0]]
        position, _, radii = certificate_world_state(self.model, self.data, robot)
        obstacle_radius = 0.010
        target = attachment_position(self.model, self.data)
        cases = (
            (NEAR_DISTANCE + 0.010, ObstacleState.NORMAL, False, False),
            (
                0.5 * (NEAR_DISTANCE + CONTACT_DISTANCE),
                ObstacleState.NEAR,
                True,
                False,
            ),
            (
                CONTACT_DISTANCE - 0.001,
                ObstacleState.CONTACT_RECOVERY,
                False,
                True,
            ),
        )
        for surface_clearance, state, near_term, recovery_row in cases:
            distance = radii[0] + obstacle_radius + surface_clearance
            center = position[0] + np.array([distance, 0.0, 0.0])
            controller = ProtocolLiuQPController(
                self.model,
                self.data,
                self.scene,
                robot,
                center[None, :],
                representation="sphere",
                obstacle_radii=np.array([obstacle_radius]),
                proxy_ids=np.array([1001]),
                safety_margin=SAFETY_MARGIN,
                near_distance=NEAR_DISTANCE,
                contact_distance=CONTACT_DISTANCE,
            )
            _, metrics = controller.solve(target)
            self.assertEqual(len(controller.last_pair_records), 1)
            record = controller.last_pair_records[0]
            self.assertEqual(record.state, state.value)
            self.assertAlmostEqual(record.surface_clearance, surface_clearance)
            self.assertAlmostEqual(
                record.clearance, surface_clearance - SAFETY_MARGIN
            )
            self.assertEqual(record.near_penalty, near_term)
            self.assertEqual(record.contact_recovery_row, recovery_row)
            self.assertEqual(metrics.near_penalty_terms, int(near_term))
            self.assertEqual(metrics.contact_repulsion_rows, int(recovery_row))

    def test_qp_failure_fallback_is_exactly_zero_velocity(self) -> None:
        controller = self._controller("sphere")

        class FailedSolver:
            def setup(self, **_kwargs) -> None:
                return None

            def warm_start(self, **_kwargs) -> None:
                return None

            def solve(self, **_kwargs):
                return SimpleNamespace(
                    x=None,
                    info=SimpleNamespace(status="primal infeasible"),
                )

        with mock.patch("protocol_liuqp_controller.osqp.OSQP", return_value=FailedSolver()):
            qdot, metrics = controller.solve(np.asarray(self.scene.waypoints[-1]))
        np.testing.assert_array_equal(qdot, np.zeros(6))
        self.assertEqual(metrics.status, "primal infeasible")

    def test_exact_closest_point_plane_and_multiplier_warm_start(self) -> None:
        ellipsoid = self._controller("ellipsoid")
        robot_center = np.array([0.31, 0.41, 0.56])
        robot_radius = 0.025
        obstacle_index = 0
        cold = ellipsoid.separating_plane(
            0,
            robot_center,
            robot_radius,
            obstacle_index,
        )
        warm = ellipsoid.separating_plane(
            0,
            robot_center,
            robot_radius,
            obstacle_index,
        )
        direct = closest_point_on_ellipsoid(
            self.proxies.centers[obstacle_index],
            self.proxies.ellipsoid_shapes[obstacle_index],
            robot_center,
        )
        np.testing.assert_allclose(cold.surface_point, direct.surface_point, atol=1.0e-11)
        self.assertLess(cold.closest_point_residual, 1.0e-10)
        self.assertLess(warm.closest_point_residual, 1.0e-10)
        self.assertLessEqual(warm.newton_iterations, cold.newton_iterations)
        self.assertLessEqual(warm.bisection_iterations, cold.bisection_iterations)
        self.assertAlmostEqual(
            float(cold.normal_to_obstacle @ cold.surface_point),
            cold.offset,
            places=12,
        )
        # Dense surface samples cannot cross to the robot side of the tangent.
        eigenvalues, rotation = np.linalg.eigh(
            self.proxies.ellipsoid_shapes[obstacle_index]
        )
        axes = np.sqrt(eigenvalues)
        theta = np.linspace(0.0, 2.0 * np.pi, 181)
        phi = np.linspace(0.0, np.pi, 91)
        points = []
        for polar in phi:
            local = np.column_stack(
                (
                    axes[0] * np.sin(polar) * np.cos(theta),
                    axes[1] * np.sin(polar) * np.sin(theta),
                    np.full_like(theta, axes[2] * np.cos(polar)),
                )
            )
            points.append(
                self.proxies.centers[obstacle_index] + local @ rotation.T
            )
        samples = np.vstack(points)
        minimum_support = np.min(samples @ cold.normal_to_obstacle)
        self.assertGreaterEqual(minimum_support, cold.offset - 1.0e-8)

    def test_scalar_measurement_offset_expands_both_matched_certificates(self) -> None:
        center = np.array([[0.0, 0.0, 0.0]])
        robot_center = np.array([0.20, 0.0, 0.0])
        robot_radius = 0.03
        measurement_offset = 0.007
        sphere = ProtocolLiuQPController(
            self.model,
            self.data,
            self.scene,
            self.robot,
            center,
            representation="sphere",
            obstacle_radii=np.array([0.05]),
            obstacle_offsets=np.array([measurement_offset]),
            safety_margin=0.0,
            redundant_plane_pruning=False,
        )
        sphere_plane = sphere.separating_plane(
            0, robot_center, robot_radius, 0
        )
        self.assertAlmostEqual(
            sphere_plane.clearance,
            0.20 - 0.05 - measurement_offset - robot_radius,
            places=12,
        )
        ellipsoid = ProtocolLiuQPController(
            self.model,
            self.data,
            self.scene,
            self.robot,
            center,
            representation="ellipsoid",
            obstacle_shapes=np.array(
                [np.diag([0.05**2, 0.03**2, 0.02**2])]
            ),
            obstacle_offsets=np.array([measurement_offset]),
            safety_margin=0.0,
            redundant_plane_pruning=False,
        )
        ellipsoid_plane = ellipsoid.separating_plane(
            0, robot_center, robot_radius, 0
        )
        self.assertAlmostEqual(
            ellipsoid_plane.clearance,
            0.20 - 0.05 - measurement_offset - robot_radius,
            places=10,
        )
        np.testing.assert_allclose(
            ellipsoid_plane.surface_point,
            np.array([0.05 + measurement_offset, 0.0, 0.0]),
            atol=1.0e-10,
        )

    def test_first_sphere_and_ellipsoid_qps_use_same_robot_chain_and_solve(self) -> None:
        sphere = self._controller("sphere")
        ellipsoid = self._controller("ellipsoid")
        self.assertIs(sphere.robot_spheres, ellipsoid.robot_spheres)
        target = np.asarray(self.scene.waypoints[-1])
        sphere_qdot, sphere_metrics = sphere.solve(target)
        ellipsoid_qdot, ellipsoid_metrics = ellipsoid.solve(target)
        self.assertTrue(sphere_metrics.status.startswith("solved"), sphere_metrics.status)
        self.assertTrue(
            ellipsoid_metrics.status.startswith("solved"),
            ellipsoid_metrics.status,
        )
        self.assertTrue(np.all(np.isfinite(sphere_qdot)))
        self.assertTrue(np.all(np.isfinite(ellipsoid_qdot)))
        self.assertGreater(sphere_metrics.geometry_ms, sphere_metrics.prune_ms)
        self.assertGreater(ellipsoid_metrics.geometry_ms, ellipsoid_metrics.prune_ms)

    def test_native_exact_batch_preserves_python_qp_rows_and_velocity(self) -> None:
        target = np.asarray(self.scene.waypoints[-1])
        native = self._controller("ellipsoid", native_exact_batch=True)
        reference = self._controller("ellipsoid", native_exact_batch=False)
        native_qdot, native_metrics = native.solve(target)
        reference_qdot, reference_metrics = reference.solve(target)
        self.assertEqual(native_metrics.qp_row_sha256, reference_metrics.qp_row_sha256)
        np.testing.assert_allclose(native_qdot, reference_qdot, atol=2.0e-8, rtol=2.0e-8)
        self.assertEqual(
            [(item.proxy_id, item.state) for item in native.last_pair_records],
            [(item.proxy_id, item.state) for item in reference.last_pair_records],
        )

    def test_native_sphere_fusion_preserves_python_qp_rows_and_velocity(self) -> None:
        target = np.asarray(self.scene.waypoints[-1])
        robot_radii = np.asarray(
            [sphere.radius for sphere in self.robot], dtype=float
        )
        indexed_proxies = SimpleNamespace(
            centers=self.proxies.centers,
            sphere_radii=self.proxies.sphere_radii,
            proxy_offset_radii=np.zeros(len(self.proxies.centers)),
            proxy_uncertainty_shapes=None,
            base_ellipsoid_shapes=self.proxies.ellipsoid_shapes,
            ellipsoid_outer_shapes=self.proxies.ellipsoid_shapes,
        )
        table = _build_mvt_only(
            indexed_proxies, "sphere", robot_radii, simd=True
        )
        self.addCleanup(table.close)
        native = self._controller(
            "sphere",
            obstacle_index=table,
            native_exact_batch=True,
            native_exact_pair_batch=True,
        )
        reference = self._controller(
            "sphere",
            obstacle_index=table,
            native_exact_batch=False,
        )
        native_qdot, native_metrics = native.solve(target)
        reference_qdot, reference_metrics = reference.solve(target)
        self.assertEqual(
            native_metrics.broadphase_candidate_pairs,
            reference_metrics.broadphase_candidate_pairs,
        )
        self.assertEqual(
            native_metrics.qp_row_sha256, reference_metrics.qp_row_sha256
        )
        np.testing.assert_allclose(
            native_qdot, reference_qdot, atol=2.0e-8, rtol=2.0e-8
        )
        self.assertEqual(
            [
                (item.proxy_id, item.state)
                for item in native.last_pair_records
            ],
            [
                (item.proxy_id, item.state)
                for item in reference.last_pair_records
            ],
        )

    def test_persistent_qp_workspace_matches_cycle_rebuild(self) -> None:
        target = np.asarray(self.scene.waypoints[-1])
        persistent = self._controller(
            "ellipsoid", persistent_qp_workspace=True
        )
        rebuilt = self._controller(
            "ellipsoid", persistent_qp_workspace=False
        )
        for _ in range(2):
            persistent_qdot, persistent_metrics = persistent.solve(target)
            rebuilt_qdot, rebuilt_metrics = rebuilt.solve(target)
            self.assertEqual(
                persistent_metrics.qp_row_sha256,
                rebuilt_metrics.qp_row_sha256,
            )
            np.testing.assert_allclose(
                persistent_qdot,
                rebuilt_qdot,
                atol=1.0e-7,
                rtol=1.0e-7,
            )
            self.assertEqual(
                [
                    (item.proxy_id, item.state)
                    for item in persistent.last_pair_records
                ],
                [
                    (item.proxy_id, item.state)
                    for item in rebuilt.last_pair_records
                ],
            )

    def test_exact_pair_batch_preserves_per_robot_qp_rows_and_velocity(self) -> None:
        target = np.asarray(self.scene.waypoints[-1])
        pair_batch = self._controller(
            "ellipsoid",
            native_exact_batch=True,
            native_exact_pair_batch=True,
        )
        per_robot = self._controller(
            "ellipsoid",
            native_exact_batch=True,
            native_exact_pair_batch=False,
        )
        pair_qdot, pair_metrics = pair_batch.solve(target)
        robot_qdot, robot_metrics = per_robot.solve(target)
        self.assertEqual(
            pair_metrics.qp_row_sha256, robot_metrics.qp_row_sha256
        )
        np.testing.assert_allclose(
            pair_qdot, robot_qdot, atol=2.0e-8, rtol=2.0e-8
        )
        self.assertEqual(
            [(item.proxy_id, item.state) for item in pair_batch.last_pair_records],
            [(item.proxy_id, item.state) for item in per_robot.last_pair_records],
        )

    def test_directional_pair_batch_preserves_qp_rows_and_velocity(self) -> None:
        target = np.asarray(self.scene.waypoints[-1])
        uncertainty = np.repeat(
            np.diag(np.square([0.0010, 0.0015, 0.0008]))[None, :, :],
            len(self.proxies.centers),
            axis=0,
        )
        common = dict(
            model=self.model,
            data=self.data,
            scene=self.scene,
            robot_spheres=self.robot,
            obstacle_centers=self.proxies.centers,
            representation="ellipsoid",
            obstacle_shapes=self.proxies.ellipsoid_shapes,
            obstacle_uncertainty_shapes=uncertainty,
            proxy_ids=self.proxies.proxy_ids,
            safety_margin=SAFETY_MARGIN,
            near_distance=NEAR_DISTANCE,
            contact_distance=CONTACT_DISTANCE,
        )
        native = ProtocolLiuQPController(
            **common, native_directional_pair_batch=True
        )
        reference = ProtocolLiuQPController(
            **common, native_directional_pair_batch=False
        )
        native_qdot, native_metrics = native.solve(target)
        reference_qdot, reference_metrics = reference.solve(target)
        self.assertEqual(
            native_metrics.qp_row_sha256, reference_metrics.qp_row_sha256
        )
        np.testing.assert_allclose(
            native_qdot, reference_qdot, atol=2.0e-8, rtol=2.0e-8
        )
        self.assertEqual(
            [(item.proxy_id, item.state) for item in native.last_pair_records],
            [(item.proxy_id, item.state) for item in reference.last_pair_records],
        )


if __name__ == "__main__":
    unittest.main()
