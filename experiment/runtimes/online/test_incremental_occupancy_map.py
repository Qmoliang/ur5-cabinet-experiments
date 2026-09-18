from __future__ import annotations

import unittest

import numpy as np

from depth_camera_perception import DepthObservation
from incremental_occupancy_map import FREE, OCCUPIED, UNKNOWN, IncrementalOccupancyMap


def observation(
    endpoint=(0.30, 0.0, 0.0), radius=0.006, endpoint_occupied=True
):
    return DepthObservation(
        camera_name="test",
        camera_position=np.zeros(3),
        camera_rotation=np.eye(3),
        points=np.asarray([endpoint], dtype=float),
        geom_ids=np.asarray([1], dtype=np.int32),
        optical_depths=np.asarray([endpoint[0]], dtype=float),
        sample_radii=np.asarray([radius], dtype=float),
        maximum_pixel_radius=radius,
        endpoint_is_occupied=np.asarray([endpoint_occupied], dtype=bool),
    )


class IncrementalOccupancyMapTests(unittest.TestCase):
    def test_vectorized_ball_keys_equal_brute_force_definition(self):
        occupancy = IncrementalOccupancyMap(voxel_size=0.017)
        center = np.array([0.031, -0.024, 0.413])
        radius = 0.058
        actual = set(occupancy._ball_keys(center, radius))
        half_diagonal = np.sqrt(3.0) * 0.5 * occupancy.voxel_size
        lower = np.floor(
            (center - radius - half_diagonal) / occupancy.voxel_size
        ).astype(int)
        upper = np.floor(
            (center + radius + half_diagonal) / occupancy.voxel_size
        ).astype(int)
        expected = set()
        for x in range(lower[0], upper[0] + 1):
            for y in range(lower[1], upper[1] + 1):
                for z in range(lower[2], upper[2] + 1):
                    key = (x, y, z)
                    if (
                        np.linalg.norm(occupancy.center(key) - center)
                        <= radius + half_diagonal
                    ):
                        expected.add(key)
        self.assertEqual(actual, expected)

    def test_query_snapshot_does_not_change_after_worker_update(self):
        occupancy = IncrementalOccupancyMap(voxel_size=0.02)
        occupancy._scores[(0, 0, 0)] = occupancy.maximum_score
        snapshot = occupancy.clone_for_query()
        occupancy._scores[(0, 0, 0)] = occupancy.minimum_score
        self.assertEqual(snapshot.state_key((0, 0, 0)), OCCUPIED)
        self.assertEqual(occupancy.state_key((0, 0, 0)), FREE)

    def test_ray_is_free_endpoint_occupied_and_behind_unknown(self):
        grid = IncrementalOccupancyMap(0.02)
        grid.integrate([observation()])
        self.assertEqual(grid.state(np.array([0.10, 0.0, 0.0])), FREE)
        self.assertEqual(grid.state(np.array([0.30, 0.0, 0.0])), OCCUPIED)
        self.assertEqual(grid.state(np.array([0.40, 0.0, 0.0])), UNKNOWN)

    def test_self_return_keeps_free_prefix_without_environment_endpoint(self):
        grid = IncrementalOccupancyMap(0.02)
        stats = grid.integrate(
            [observation(endpoint=(0.30, 0.0, 0.0), endpoint_occupied=False)]
        )
        self.assertEqual(grid.state(np.array([0.10, 0.0, 0.0])), FREE)
        self.assertNotEqual(grid.state(np.array([0.30, 0.0, 0.0])), OCCUPIED)
        self.assertEqual(stats.occupied_updates, 0)

    def test_frames_are_causal_and_accumulate(self):
        grid = IncrementalOccupancyMap(0.02)
        first = grid.integrate([observation((0.30, 0.0, 0.0))])
        second = grid.integrate([observation((0.30, 0.10, 0.0))])
        self.assertEqual(first.frame_index, 0)
        self.assertEqual(second.frame_index, 1)
        self.assertGreater(second.free_voxels, first.free_voxels)

    def test_unknown_hard_guard_rejects_unobserved_sweep(self):
        grid = IncrementalOccupancyMap(0.02)
        grid.integrate([observation()])
        # The current robot volume is known safe by induction; a single depth
        # ray alone must not be promoted to a finite-width free tunnel.
        grid.mark_current_robot_free(
            np.asarray([[0.06, 0.0, 0.0]]), np.asarray([0.018])
        )
        safe = grid.certify_swept_spheres(
            np.asarray([[0.06, 0.0, 0.0]]),
            np.asarray([[0.065, 0.0, 0.0]]),
            np.asarray([0.005]),
        )
        blocked = grid.certify_swept_spheres(
            np.asarray([[0.06, 0.08, 0.0]]),
            np.asarray([[0.12, 0.08, 0.0]]),
            np.asarray([0.005]),
        )
        self.assertTrue(safe.safe)
        self.assertFalse(blocked.safe)
        self.assertGreater(blocked.unknown_voxels, 0)


if __name__ == "__main__":
    unittest.main()
