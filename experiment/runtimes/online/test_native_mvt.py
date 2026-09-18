from __future__ import annotations

import unittest

import numpy as np

from native_mvt import BruteForceAABBIndex, NativeMVT, NativeMultilevelMVT


class NativeMVTTests(unittest.TestCase):
    def test_random_queries_exactly_match_naive_aabb_oracle(self):
        rng = np.random.default_rng(42)
        centers = rng.uniform([-0.2, -0.4, 0.1], [0.9, 0.8, 1.1], size=(513, 3))
        half = rng.uniform(0.005, 0.08, size=(513, 3))
        queries = [
            (
                rng.uniform([-0.4, -0.6, 0.0], [1.1, 1.0, 1.3]),
                rng.uniform(0.001, 0.16, size=3),
            )
            for _ in range(250)
        ]
        for simd in (False, True):
            table = NativeMVT(centers, half, voxel_size=0.07, simd=simd)
            self.addCleanup(table.close)
            self.assertEqual(table.stats.simd_width, 8)
            self.assertGreater(table.stats.occupied_cells, 0)
            for query_center, query_half in queries:
                expected = np.flatnonzero(
                    np.all(centers - half <= query_center + query_half, axis=1)
                    & np.all(centers + half >= query_center - query_half, axis=1)
                )
                actual = table.query_aabb(query_center, query_half)
                np.testing.assert_array_equal(actual, expected)

    def test_sphere_and_ellipsoid_constructors(self):
        centers = np.array([[0.0, 0.0, 0.2], [0.2, 0.0, 0.2]])
        spheres = NativeMVT.from_spheres(centers, np.array([0.03, 0.04]), 0.05)
        self.addCleanup(spheres.close)
        np.testing.assert_array_equal(spheres.query_sphere(np.zeros(3), 0.25), [0, 1])
        shapes = np.array([np.diag([0.01, 0.0025, 0.0004])] * 2)
        ellipsoids = NativeMVT.from_ellipsoids(centers, shapes, np.array([0.005, 0.005]), 0.05)
        self.addCleanup(ellipsoids.close)
        self.assertEqual(ellipsoids.stats.proxy_count, 2)


class NativeMultilevelMVTTests(unittest.TestCase):
    def test_ball_aabb_rejects_query_cube_corners(self):
        centers = np.array([[0.09, 0.09, 0.0], [0.07, 0.07, 0.0]])
        half = np.full((2, 3), 0.005)
        table = NativeMultilevelMVT(
            centers,
            half,
            base_voxel_size=0.025,
            maximum_query_half_extent=0.10,
        )
        self.addCleanup(table.close)
        # Both boxes overlap the query cube, but only the second intersects
        # the radius-0.1 ball.
        np.testing.assert_array_equal(
            table.query_sphere(np.zeros(3), 0.10), [1]
        )

    def test_bruteforce_oracle_matches_scalar_and_simd_candidate_ids(self):
        rng = np.random.default_rng(20260830)
        centers = rng.uniform(-0.8, 0.8, size=(257, 3))
        half = rng.uniform(0.002, 0.12, size=(257, 3))
        padding = 0.047
        maximum_query_half = 0.18
        oracle = BruteForceAABBIndex(
            centers, half, query_padding=padding
        )
        scalar = NativeMultilevelMVT(
            centers,
            half,
            base_voxel_size=0.015,
            maximum_query_half_extent=maximum_query_half,
            query_padding=padding,
            simd=False,
        )
        simd = NativeMultilevelMVT(
            centers,
            half,
            base_voxel_size=0.015,
            maximum_query_half_extent=maximum_query_half,
            query_padding=padding,
            simd=True,
        )
        try:
            query_centers = rng.uniform(-0.7, 0.7, size=(64, 3))
            radii = rng.uniform(0.005, maximum_query_half - padding, size=64)
            expected_rows = oracle.query_spheres(query_centers, radii)
            scalar_rows = scalar.query_spheres(query_centers, radii)
            simd_rows = simd.query_spheres(query_centers, radii)
            unordered_rows = simd.query_spheres_unordered(
                query_centers, radii
            )
            for expected, scalar_row, simd_row, unordered_row in zip(
                expected_rows, scalar_rows, simd_rows, unordered_rows
            ):
                np.testing.assert_array_equal(scalar_row, expected)
                np.testing.assert_array_equal(simd_row, expected)
                np.testing.assert_array_equal(
                    np.sort(unordered_row), expected
                )
                self.assertEqual(
                    len(unordered_row), len(np.unique(unordered_row))
                )
        finally:
            scalar.close()
            simd.close()

    def test_random_queries_exactly_match_full_scan_oracle(self):
        rng = np.random.default_rng(20260829)
        centers = rng.uniform([-0.8, -0.9, -0.2], [1.1, 0.9, 1.4], size=(777, 3))
        half = rng.uniform(0.001, 0.12, size=(777, 3))
        maximum_query_half = 0.19
        queries = [
            (
                rng.uniform([-1.0, -1.1, -0.4], [1.3, 1.1, 1.6]),
                rng.uniform(0.0, maximum_query_half, size=3),
            )
            for _ in range(400)
        ]
        for simd in (False, True):
            table = NativeMultilevelMVT(
                centers,
                half,
                base_voxel_size=0.0125,
                maximum_query_half_extent=maximum_query_half,
                simd=simd,
            )
            self.addCleanup(table.close)
            self.assertEqual(table.stats.simd_width, 8)
            self.assertEqual(table.stats.index_references, len(centers))
            self.assertEqual(sum(table.stats.level_proxy_counts), len(centers))
            self.assertEqual(
                table.stats.cell_lookups_per_query,
                27 * table.stats.level_count,
            )
            for query_center, query_half in queries:
                expected = np.flatnonzero(
                    np.all(centers - half <= query_center + query_half, axis=1)
                    & np.all(centers + half >= query_center - query_half, axis=1)
                )
                actual = table.query_aabb(query_center, query_half)
                np.testing.assert_array_equal(actual, expected)

    def test_boundary_queries_and_maximum_extent_guard(self):
        centers = np.array(
            [[-0.05000001, 0.0, 0.0], [0.05000001, 0.0, 0.0], [0.30, 0.0, 0.0]]
        )
        half = np.array([[0.010, 0.020, 0.015]] * 3)
        table = NativeMultilevelMVT(
            centers,
            half,
            base_voxel_size=0.025,
            maximum_query_half_extent=0.08,
        )
        self.addCleanup(table.close)
        expected = np.flatnonzero(
            np.all(centers - half <= np.array([0.0, 0.0, 0.0]) + 0.08, axis=1)
            & np.all(centers + half >= np.array([0.0, 0.0, 0.0]) - 0.08, axis=1)
        )
        np.testing.assert_array_equal(
            table.query_aabb(np.zeros(3), np.full(3, 0.08)), expected
        )
        with self.assertRaises(ValueError):
            table.query_aabb(np.zeros(3), np.array([0.081, 0.01, 0.01]))

    def test_batch_queries_exactly_match_individual_queries(self):
        rng = np.random.default_rng(20260830)
        centers = rng.uniform(-0.7, 0.9, size=(421, 3))
        half = rng.uniform(0.002, 0.095, size=(421, 3))
        query_centers = rng.uniform(-0.9, 1.1, size=(65, 3))
        query_half = rng.uniform(0.001, 0.18, size=(65, 3))
        for simd in (False, True):
            table = NativeMultilevelMVT(
                centers,
                half,
                base_voxel_size=0.0125,
                maximum_query_half_extent=0.18,
                simd=simd,
            )
            self.addCleanup(table.close)
            actual = table.query_aabbs(query_centers, query_half)
            expected = tuple(
                table.query_aabb(center, extent)
                for center, extent in zip(query_centers, query_half)
            )
            self.assertEqual(len(actual), len(expected))
            for batch_row, individual_row in zip(actual, expected):
                np.testing.assert_array_equal(batch_row, individual_row)

    def test_multilevel_sphere_and_ellipsoid_constructors(self):
        centers = np.array([[0.0, 0.0, 0.2], [0.2, 0.0, 0.2]])
        spheres = NativeMultilevelMVT.from_spheres(
            centers,
            np.array([0.03, 0.04]),
            0.02,
            0.30,
        )
        self.addCleanup(spheres.close)
        np.testing.assert_array_equal(
            spheres.query_sphere(np.zeros(3), 0.25), [0, 1]
        )
        shapes = np.array([np.diag([0.01, 0.0025, 0.0004])] * 2)
        ellipsoids = NativeMultilevelMVT.from_ellipsoids(
            centers,
            shapes,
            np.array([0.005, 0.005]),
            0.02,
            0.30,
        )
        self.addCleanup(ellipsoids.close)
        self.assertEqual(ellipsoids.stats.proxy_count, 2)
        self.assertEqual(ellipsoids.stats.index_references, 2)


if __name__ == "__main__":
    unittest.main()
