from __future__ import annotations

import unittest

import mujoco
import numpy as np

from depth_camera_perception import UR5MountedDepthCamera
from incremental_drawer_scene import DRAWER_Z, FRONT_X, incremental_drawer_scene
from model import (
    attachment_position,
    build_model,
    build_robot_certificate,
    certificate_world_state,
    set_configuration,
)


class IncrementalDrawerProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = incremental_drawer_scene()
        cls.model = build_model(cls.scene)
        cls.data = mujoco.MjData(cls.model)

    def setUp(self) -> None:
        set_configuration(self.model, self.data, np.asarray(self.scene.q0))

    def test_initial_robot_is_below_and_outside_drawer(self) -> None:
        tip = attachment_position(self.model, self.data)
        centers, _, _ = certificate_world_state(
            self.model, self.data, build_robot_certificate(self.model)
        )
        self.assertLess(tip[0], FRONT_X)
        self.assertLess(tip[2], DRAWER_Z - 0.09)
        self.assertLess(float(np.max(centers[:, 0])), FRONT_X)
        self.assertEqual(self.data.ncon, 0)

    def test_compact_gripper_and_camera_housing_are_certified(self) -> None:
        names = {sphere.body_name for sphere in build_robot_certificate(self.model)}
        self.assertIn("wrist_3_link", names)
        geom_names = {self.model.geom(i).name for i in range(self.model.ngeom)}
        self.assertIn("compact_gripper_left", geom_names)
        self.assertIn("compact_gripper_right", geom_names)
        self.assertIn("d405_camera_housing", geom_names)
        self.assertNotIn("drawer_retrieval_tool", geom_names)

    def test_occluding_self_filter_keeps_robot_out_of_cloud(self) -> None:
        with UR5MountedDepthCamera(
            self.model,
            width=96,
            height=72,
            pixel_stride=2,
            occluding_self_filter=True,
        ) as camera:
            observation = camera.capture(self.data)[0]
            self.assertTrue(
                all(int(geom_id) not in camera.robot_geom_ids for geom_id in observation.geom_ids)
            )
            self.assertGreater(len(observation.points), 0)


if __name__ == "__main__":
    unittest.main()

