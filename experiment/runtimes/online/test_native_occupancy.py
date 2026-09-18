from __future__ import annotations

import unittest

import numpy as np

from depth_camera_perception import DepthObservation
from incremental_occupancy_map import IncrementalOccupancyMap
from native_occupancy import NativeIncrementalOccupancyMap, NativeOccupancySnapshot


class NativeOccupancyTests(unittest.TestCase):
    def test_dirty_state_deltas_reconstruct_every_known_voxel(self):
        rng = np.random.default_rng(2031)
        live = NativeIncrementalOccupancyMap(
            voxel_size=0.012, track_state_deltas=True
        )
        self.addCleanup(live.close)
        replay: dict[tuple[int, int, int], int] = {}
        origin = np.array([0.0, 0.0, 0.42])
        for frame in range(3):
            points = rng.uniform(
                [0.10, -0.12, 0.28], [0.52, 0.12, 0.70], size=(80, 3)
            )
            radii = rng.uniform(0.001, 0.006, size=len(points))
            live.integrate(
                [
                    DepthObservation(
                        camera_name=f"delta_{frame}",
                        camera_position=origin,
                        camera_rotation=np.eye(3),
                        points=points,
                        geom_ids=np.zeros(len(points), dtype=np.int32),
                        optical_depths=np.linalg.norm(points - origin, axis=1),
                        sample_radii=radii,
                        maximum_pixel_radius=float(np.max(radii)),
                    )
                ]
            )
            delta_keys, delta_states = live.snapshot_delta_arrays()
            for key, state in zip(delta_keys, delta_states):
                packed = tuple(map(int, key))
                if int(state) == 0:
                    replay.pop(packed, None)
                else:
                    replay[packed] = int(state)
            full_keys, full_states = live.snapshot_arrays()
            expected = {
                tuple(map(int, key)): int(state)
                for key, state in zip(full_keys, full_states)
                if int(state) != 0
            }
            self.assertEqual(replay, expected)
            empty_keys, empty_states = live.snapshot_delta_arrays()
            self.assertEqual(len(empty_keys), 0)
            self.assertEqual(len(empty_states), 0)

    def test_directional_endpoint_integration_uses_ellipsoid_not_outer_sphere(self):
        voxel = 0.01
        center = np.array([[0.20, 0.20, 0.20]])
        shape = np.diag([0.055**2, 0.003**2, 0.003**2])[None, :, :]
        radius = np.array([0.055])

        def observation(directional: bool) -> DepthObservation:
            return DepthObservation(
                camera_name="directional_endpoint",
                camera_position=np.array([0.0, 0.20, 0.20]),
                camera_rotation=np.eye(3),
                points=center,
                geom_ids=np.zeros(1, dtype=np.int32),
                optical_depths=np.array([0.20]),
                sample_radii=radius,
                maximum_pixel_radius=float(radius[0]),
                sample_uncertainty_shapes=shape if directional else None,
            )

        ellipsoid_map = NativeIncrementalOccupancyMap(voxel_size=voxel)
        sphere_map = NativeIncrementalOccupancyMap(voxel_size=voxel)
        self.addCleanup(ellipsoid_map.close)
        self.addCleanup(sphere_map.close)
        ellipsoid_map.integrate([observation(True)])
        sphere_map.integrate([observation(False)])
        ellipsoid_keys, ellipsoid_states = ellipsoid_map.snapshot_arrays()
        sphere_keys, sphere_states = sphere_map.snapshot_arrays()
        ellipsoid = {
            tuple(key): int(state)
            for key, state in zip(ellipsoid_keys, ellipsoid_states)
        }
        sphere = {
            tuple(key): int(state) for key, state in zip(sphere_keys, sphere_states)
        }
        # Both certificates include the endpoint and the ellipsoid major axis.
        self.assertEqual(ellipsoid[(20, 20, 20)], 2)
        self.assertEqual(ellipsoid[(24, 20, 20)], 2)
        # A voxel 4 cm along the narrow axis belongs to the outer sphere but
        # cannot intersect the 3 mm ellipsoid cross-section.
        self.assertNotEqual(ellipsoid.get((20, 24, 20), 0), 2)
        self.assertEqual(sphere[(20, 24, 20)], 2)

    def test_self_return_marks_free_prefix_but_not_occupied_endpoint(self):
        live = NativeIncrementalOccupancyMap(voxel_size=0.01)
        self.addCleanup(live.close)
        observation = DepthObservation(
            camera_name="self_test",
            camera_position=np.array([0.0, 0.0, 0.30]),
            camera_rotation=np.eye(3),
            points=np.array([[0.12, 0.0, 0.30]]),
            geom_ids=np.array([1], dtype=np.int32),
            optical_depths=np.array([0.12]),
            sample_radii=np.array([0.002]),
            maximum_pixel_radius=0.002,
            endpoint_is_occupied=np.array([False]),
        )
        stats = live.integrate([observation])
        keys, states = live.snapshot_arrays()
        state_by_key = {tuple(key): int(state) for key, state in zip(keys, states)}
        self.assertEqual(state_by_key[(5, 0, 30)], 1)
        self.assertNotEqual(state_by_key.get((12, 0, 30), 0), 2)
        self.assertEqual(stats.occupied_updates, 0)

    def test_calibrated_staging_is_free_but_occupied_evidence_wins(self):
        live = NativeIncrementalOccupancyMap(voxel_size=0.01)
        self.addCleanup(live.close)
        live.set_calibrated_free_aabb(
            np.array([-0.20, -0.20, 0.0]), np.array([0.30, 0.20, 0.60])
        )
        inside = live.certify_swept_spheres(
            np.array([[0.10, 0.0, 0.30]]),
            np.array([[0.12, 0.0, 0.30]]),
            np.array([0.02]),
        )
        outside = live.certify_swept_spheres(
            np.array([[0.31, 0.0, 0.30]]),
            np.array([[0.32, 0.0, 0.30]]),
            np.array([0.005]),
        )
        self.assertTrue(inside.safe)
        self.assertGreater(outside.unknown_voxels, 0)
        endpoint = np.array([[0.12, 0.0, 0.30]])
        live.integrate(
            [
                DepthObservation(
                    camera_name="test",
                    camera_position=np.array([0.0, 0.0, 0.30]),
                    camera_rotation=np.eye(3),
                    points=endpoint,
                    geom_ids=np.zeros(1, dtype=np.int32),
                    optical_depths=np.array([0.12]),
                    sample_radii=np.array([0.002]),
                    maximum_pixel_radius=0.002,
                )
            ]
        )
        occupied = live.certify_swept_spheres(
            endpoint, endpoint, np.array([0.003])
        )
        self.assertGreater(occupied.occupied_voxels, 0)
        self.assertFalse(occupied.safe)

    def test_ellipsoid_voxel_query_does_not_fall_back_to_its_aabb(self):
        voxel = 0.02
        keys = np.asarray(
            [(x, y, z) for x in range(-5, 6) for y in range(-5, 6) for z in range(-2, 3)],
            dtype=np.int32,
        )
        states = np.ones(len(keys), dtype=np.int8)
        occupied_key = (2, -3, 0)
        occupied_index = np.flatnonzero(np.all(keys == occupied_key, axis=1))[0]
        states[occupied_index] = 2
        native = NativeOccupancySnapshot(keys, states, voxel)
        self.addCleanup(native.close)
        angle = np.pi / 4.0
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0],
             [np.sin(angle), np.cos(angle), 0.0],
             [0.0, 0.0, 1.0]]
        )
        slender = rotation @ np.diag([0.08**2, 0.006**2, 0.006**2]) @ rotation.T
        exact = native.certify_ellipsoids(np.zeros((1, 3)), slender[None, :, :])
        aabb_fallback = native.certify_ellipsoids(
            np.zeros((1, 3)), (np.eye(3) * 0.08**2)[None, :, :]
        )
        self.assertTrue(exact.safe)
        self.assertEqual(exact.occupied_voxels, 0)
        self.assertGreater(aabb_fallback.occupied_voxels, 0)

    def test_native_incremental_updates_match_python_reference(self):
        rng = np.random.default_rng(20260829)
        reference = IncrementalOccupancyMap(voxel_size=0.017)
        native = NativeIncrementalOccupancyMap(voxel_size=0.017)
        self.addCleanup(native.close)
        for frame in range(4):
            observations = []
            for camera in range(2):
                origin = np.array([0.02 * camera, -0.04, 0.42])
                points = rng.uniform(
                    [0.12, -0.16, 0.32], [0.55, 0.18, 0.72], size=(45, 3)
                )
                radii = rng.uniform(0.0015, 0.009, size=len(points))
                endpoint_mask = rng.random(len(points)) > 0.25
                observations.append(
                    DepthObservation(
                        camera_name=f"camera_{camera}",
                        camera_position=origin,
                        camera_rotation=np.eye(3),
                        points=points,
                        geom_ids=np.zeros(len(points), dtype=np.int32),
                        optical_depths=np.linalg.norm(points - origin, axis=1),
                        sample_radii=radii,
                        maximum_pixel_radius=float(np.max(radii)),
                        endpoint_is_occupied=endpoint_mask,
                    )
                )
            expected_stats = reference.integrate(observations)
            actual_stats = native.integrate(observations)
            self.assertEqual(actual_stats, expected_stats)
            centers = rng.uniform([0.0, -0.1, 0.35], [0.3, 0.1, 0.65], size=(5, 3))
            robot_radii = rng.uniform(0.012, 0.04, size=5)
            reference.mark_current_robot_free(centers, robot_radii)
            native.mark_current_robot_free(centers, robot_radii)
            expected_keys, expected_states = reference.snapshot_arrays()
            actual_keys, actual_states = native.snapshot_arrays()
            expected = {
                tuple(map(int, key)): int(state)
                for key, state in zip(expected_keys, expected_states)
            }
            actual = {
                tuple(map(int, key)): int(state)
                for key, state in zip(actual_keys, actual_states)
            }
            self.assertEqual(actual, expected)

    def test_random_sweeps_match_python_counts(self):
        rng = np.random.default_rng(89)
        occupancy = IncrementalOccupancyMap(voxel_size=0.017)
        for key in rng.integers(-18, 19, size=(1800, 3)):
            occupancy._scores[tuple(map(int, key))] = (
                occupancy.maximum_score
                if rng.random() < 0.25
                else occupancy.minimum_score
            )
        keys, states = occupancy.snapshot_arrays()
        native = NativeOccupancySnapshot(keys, states, occupancy.voxel_size)
        self.addCleanup(native.close)
        for _ in range(60):
            starts = rng.uniform(-0.25, 0.25, size=(7, 3))
            ends = starts + rng.uniform(-0.025, 0.025, size=(7, 3))
            radii = rng.uniform(0.008, 0.045, size=7)
            expected = occupancy.certify_swept_spheres(
                starts, ends, radii, margin=0.003
            )
            actual = native.certify_swept_spheres(
                starts, ends, radii, margin=0.003
            )
            self.assertEqual(actual, expected)

    def test_live_incremental_sweep_matches_exported_snapshot(self):
        rng = np.random.default_rng(1337)
        live = NativeIncrementalOccupancyMap(voxel_size=0.016)
        self.addCleanup(live.close)
        points = rng.uniform([0.1, -0.2, 0.2], [0.6, 0.2, 0.8], size=(250, 3))
        origin = np.array([0.0, 0.0, 0.45])
        radii = rng.uniform(0.001, 0.006, size=len(points))
        observation = DepthObservation(
            camera_name="test",
            camera_position=origin,
            camera_rotation=np.eye(3),
            points=points,
            geom_ids=np.zeros(len(points), dtype=np.int32),
            optical_depths=np.linalg.norm(points - origin, axis=1),
            sample_radii=radii,
            maximum_pixel_radius=float(np.max(radii)),
        )
        live.integrate([observation])
        robot_centers = rng.uniform([0.0, -0.05, 0.35], [0.2, 0.05, 0.55], size=(8, 3))
        robot_radii = rng.uniform(0.01, 0.035, size=8)
        live.mark_current_robot_free(robot_centers, robot_radii)
        keys, states = live.snapshot_arrays()
        snapshot = NativeOccupancySnapshot(keys, states, live.voxel_size)
        self.addCleanup(snapshot.close)
        clone = live.clone_snapshot()
        self.addCleanup(clone.close)
        ends = robot_centers + rng.uniform(-0.012, 0.012, size=(8, 3))
        expected = snapshot.certify_swept_spheres(
            robot_centers, ends, robot_radii, margin=0.002
        )
        actual = live.certify_swept_spheres(
            robot_centers, ends, robot_radii, margin=0.002
        )
        cloned = clone.certify_swept_spheres(
            robot_centers, ends, robot_radii, margin=0.002
        )
        self.assertEqual(actual, expected)
        self.assertEqual(cloned, expected)

        # A cloned snapshot is immutable: later live-map integration and score
        # changes must not alter the certificate returned by the clone.
        extra_points = rng.uniform(
            [0.15, -0.08, 0.30], [0.45, 0.08, 0.65], size=(180, 3)
        )
        extra_radii = np.full(len(extra_points), 0.004, dtype=float)
        live.integrate(
            [
                DepthObservation(
                    camera_name="later_frame",
                    camera_position=origin,
                    camera_rotation=np.eye(3),
                    points=extra_points,
                    geom_ids=np.zeros(len(extra_points), dtype=np.int32),
                    optical_depths=np.linalg.norm(extra_points - origin, axis=1),
                    sample_radii=extra_radii,
                    maximum_pixel_radius=0.004,
                )
            ]
        )
        live.mark_current_robot_free(
            robot_centers + np.array([0.015, 0.0, 0.0]), robot_radii
        )
        self.assertEqual(
            clone.certify_swept_spheres(
                robot_centers, ends, robot_radii, margin=0.002
            ),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
