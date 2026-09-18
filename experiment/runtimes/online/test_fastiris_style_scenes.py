"""Regression checks for the transparent FastIRIS-style UR5e adaptations."""

import unittest

import mujoco
import numpy as np

from fastiris_style_scenes import FASTIRIS_STYLE_SCENES
from model import build_model, set_configuration
from pointcloud_proxy import build_box_scene_proxy_set
from run_simulation import minimum_mujoco_world_contact


class FastIRISStyleSceneTests(unittest.TestCase):
    def test_starts_are_exact_collision_free_and_continuously_covered(self) -> None:
        for name, scene in FASTIRIS_STYLE_SCENES.items():
            with self.subTest(scene=name):
                model = build_model(scene)
                data = mujoco.MjData(model)
                set_configuration(model, data, np.asarray(scene.q0))
                self.assertGreaterEqual(
                    minimum_mujoco_world_contact(data, model), -1.0e-8
                )
                proxies = build_box_scene_proxy_set(
                    scene.boxes,
                    point_spacing=0.006,
                    filter_size=0.006,
                    cluster_size=0.100,
                    maximum_aabb_overshoot=0.025,
                    validation_spacing=0.0015,
                )
                self.assertGreater(proxies.surface_cover_radius, 0.0)
                self.assertEqual(len(proxies.centers), len(proxies.sphere_radii))
                self.assertEqual(len(proxies.centers), len(proxies.ellipsoid_shapes))
                self.assertLessEqual(proxies.maximum_ellipsoid_overshoot, 0.0251)


if __name__ == "__main__":
    unittest.main()
