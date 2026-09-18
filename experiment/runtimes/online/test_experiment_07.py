from __future__ import annotations

import unittest

import numpy as np

from experiment_07_scenes import experiment_07_drawer_scene
from pointcloud_proxy import fit_matched_voxel_proxies
from protocol_drawer_scene import formal_drawer_camera_v5_balanced_scene


class Experiment07Test(unittest.TestCase):
    def test_drawer_inherits_frozen_geometry_and_adds_ground_supports(self):
        frozen = formal_drawer_camera_v5_balanced_scene()
        scene = experiment_07_drawer_scene()
        self.assertEqual(scene.q0, frozen.q0)
        self.assertEqual(scene.waypoints, frozen.waypoints)
        self.assertEqual(len(scene.boxes), len(frozen.boxes) + 2)
        for old, new in zip(frozen.boxes, scene.boxes):
            self.assertEqual(old.name, new.name)
            np.testing.assert_array_equal(old.center, new.center)
            np.testing.assert_array_equal(old.half_size, new.half_size)
        self.assertEqual(
            min(box.center[2] - box.half_size[2] for box in scene.boxes), 0.0
        )

    def test_minimum_core_axis_preserves_planar_coverage(self):
        grid = np.linspace(-0.02, 0.02, 5)
        points = np.asarray([(x, y, 0.0) for x in grid for y in grid], dtype=float)
        proxies = fit_matched_voxel_proxies(
            points,
            filter_size=0.001,
            cluster_size=0.1,
            maximum_aabb_overshoot=0.1,
            already_filtered=True,
            minimum_core_semi_axis=0.0075,
        )
        minimum_axes = np.sqrt(
            np.maximum(np.linalg.eigvalsh(proxies.ellipsoid_shapes), 0.0)
        )[:, 0]
        self.assertTrue(np.all(minimum_axes >= 0.0075 - 1.0e-12))
        for point_index, point in enumerate(proxies.filtered_points):
            proxy_index = proxies.filtered_cluster_indices[point_index]
            offset = point - proxies.centers[proxy_index]
            value = offset @ np.linalg.solve(
                proxies.ellipsoid_shapes[proxy_index], offset
            )
            self.assertLessEqual(value, 1.0 + 1.0e-10)

    def test_negative_minimum_core_axis_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            fit_matched_voxel_proxies(
                np.zeros((1, 3)),
                already_filtered=True,
                minimum_core_semi_axis=-0.001,
            )


if __name__ == "__main__":
    unittest.main()
