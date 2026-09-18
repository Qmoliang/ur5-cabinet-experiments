from __future__ import annotations

import unittest

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from depth_camera_perception import (
    UR5MountedDepthCamera,
    center_voxel_filter_with_residual,
    fuse_depth_observations,
)
from model import SCENES, build_model, set_configuration


class UR5DepthCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = SCENES["shelf"]
        cls.model = build_model(cls.scene)
        cls.data = mujoco.MjData(cls.model)

    def setUp(self) -> None:
        set_configuration(self.model, self.data, np.asarray(self.scene.q0))

    def test_camera_is_body_mounted_and_wrist_changes_view_direction(self) -> None:
        camera_id = self.model.camera("ur5_depth_wrist").id
        initial_position = self.data.cam_xpos[camera_id].copy()
        initial_direction = -self.data.cam_xmat[camera_id].reshape(3, 3)[:, 2]
        q = np.asarray(self.scene.q0).copy()
        q[4] -= 1.1
        set_configuration(self.model, self.data, q)
        moved_position = self.data.cam_xpos[camera_id].copy()
        moved_direction = -self.data.cam_xmat[camera_id].reshape(3, 3)[:, 2]
        self.assertGreater(np.linalg.norm(moved_position - initial_position), 0.01)
        self.assertLess(float(initial_direction @ moved_direction), 0.75)

    def test_forward_scan_observes_shelf_without_robot_points(self) -> None:
        q = np.asarray(self.scene.q0).copy()
        q[4] -= 1.4
        set_configuration(self.model, self.data, q)
        accepted = {self.model.geom(box.name).id for box in self.scene.boxes}
        with UR5MountedDepthCamera(
            self.model, width=96, height=72, pixel_stride=3
        ) as camera:
            observation = camera.capture(self.data)[0]
        observed = set(int(value) for value in observation.geom_ids)
        self.assertNotIn(-1, observed)
        self.assertTrue(observed & accepted)
        robot_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) != 0
        }
        self.assertFalse(observed & robot_geom_ids)

    def test_native_depth_points_are_exact_first_visible_ray_hits(self) -> None:
        q = np.asarray(self.scene.q0).copy()
        q[4] -= 1.4
        set_configuration(self.model, self.data, q)
        with UR5MountedDepthCamera(
            self.model, width=96, height=72, pixel_stride=3,
            native_raycast=True,
        ) as camera:
            observation = camera.capture(self.data)[0]
            geomgroup = np.asarray(camera.scene_option.geomgroup, dtype=np.uint8)
        self.assertGreater(len(observation.points), 20)
        # Audit a deterministic spread of returned samples with the scalar
        # reference query. Runtime proxy construction never receives this
        # simulation-only geometry identity.
        sample = np.linspace(
            0, len(observation.points) - 1, min(64, len(observation.points)),
            dtype=int,
        )
        for index in sample:
            displacement = observation.points[index] - observation.camera_position
            observed_range = float(np.linalg.norm(displacement))
            direction = displacement / observed_range
            geom_id = np.array([-1], dtype=np.int32)
            reference_range = mujoco.mj_ray(
                self.model,
                self.data,
                observation.camera_position,
                direction,
                geomgroup,
                True,
                -1,
                geom_id,
            )
            self.assertAlmostEqual(observed_range, float(reference_range), places=11)
            self.assertEqual(int(observation.geom_ids[index]), int(geom_id[0]))

    def test_centervox_residual_covers_every_input_point(self) -> None:
        points = np.array(
            [
                [0.001, 0.001, 0.001],
                [0.008, 0.008, 0.008],
                [0.021, 0.001, 0.001],
            ]
        )
        filtered, residuals = center_voxel_filter_with_residual(points, 0.02)
        tree = cKDTree(filtered)
        distances, _ = tree.query(points, k=1)
        self.assertLessEqual(float(np.max(distances)), float(np.max(residuals)) + 1e-12)

    def test_fused_cloud_reports_conservative_observed_surface_radius(self) -> None:
        accepted = {self.model.geom(box.name).id for box in self.scene.boxes}
        observations = []
        with UR5MountedDepthCamera(
            self.model, width=96, height=72, pixel_stride=3
        ) as camera:
            for wrist_1_offset in (-0.25, 0.0, 0.25):
                q = np.asarray(self.scene.q0).copy()
                q[3] += wrist_1_offset
                q[4] -= 1.4
                set_configuration(self.model, self.data, q)
                self.assertEqual(self.data.ncon, 0)
                observations.extend(camera.capture(self.data))
        fused = fuse_depth_observations(
            observations,
            voxel_size=0.012,
            calibrated_depth_bound=0.003,
            workspace_lower=np.array([0.0, -0.7, 0.05]),
            workspace_upper=np.array([0.9, 0.7, 1.2]),
            accepted_geom_ids=accepted,
        )
        self.assertGreater(len(fused.raw_points), 100)
        self.assertGreater(len(fused.filtered_points), 50)
        tree = cKDTree(fused.filtered_points)
        distances, _ = tree.query(fused.raw_points, k=1)
        self.assertLessEqual(float(np.max(distances)), fused.cover_radius + 1e-12)
        self.assertGreater(fused.cover_radius, 0.003)


if __name__ == "__main__":
    unittest.main()
