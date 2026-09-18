from __future__ import annotations

import unittest

import numpy as np

from ellipsoid_model import (
    closest_point_on_ellipsoid,
    optimal_support_separating_normal,
    optimal_support_sum_separating_normal,
)
from native_ellipsoid_support import NativeEllipsoidSupport
from native_mvt import NativeMultilevelMVT


class NativeEllipsoidSupportTests(unittest.TestCase):
    def test_fused_support_sum_lazy_pruning_matches_eager_exact_rows(self):
        rng = np.random.default_rng(20260901)
        obstacle_count = 900
        centers = rng.uniform(
            [-0.35, -0.45, 0.05], [0.85, 0.65, 1.05],
            size=(obstacle_count, 3),
        )
        rotations = np.empty((obstacle_count, 3, 3), dtype=float)
        shapes = np.empty((obstacle_count, 3, 3), dtype=float)
        uncertainty = np.empty_like(shapes)
        for index in range(obstacle_count):
            basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            rotations[index] = basis
            axes = rng.uniform(0.006, 0.055, 3)
            shapes[index] = basis @ np.diag(axes**2) @ basis.T
            error_basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            error_axes = rng.uniform(0.0005, 0.009, 3)
            uncertainty[index] = (
                error_basis @ np.diag(error_axes**2) @ error_basis.T
            )
        eigenvalues = np.linalg.eigvalsh(shapes)
        uncertainty_eigenvalues = np.linalg.eigvalsh(uncertainty)
        offsets = rng.uniform(0.0, 0.004, obstacle_count)
        half = (
            np.sqrt(np.maximum(np.diagonal(shapes, axis1=1, axis2=2), 0.0))
            + np.sqrt(
                np.maximum(
                    np.diagonal(uncertainty, axis1=1, axis2=2), 0.0
                )
            )
            + offsets[:, None]
        )
        robot_centers = rng.uniform(
            [-0.20, -0.30, 0.15], [0.70, 0.50, 0.90], size=(17, 3)
        )
        robot_radii = rng.uniform(0.018, 0.055, len(robot_centers))
        query_padding = 0.052
        table = NativeMultilevelMVT(
            centers,
            half,
            base_voxel_size=0.0125,
            maximum_query_half_extent=0.12,
            query_padding=query_padding,
            simd=True,
        )
        self.addCleanup(table.close)
        kernel = NativeEllipsoidSupport()
        dense = np.zeros(
            (len(robot_centers), obstacle_count, 3), dtype=np.float64
        )
        (
            actual_active,
            actual_normals,
            _actual_surfaces,
            _actual_iterations,
            actual_residuals,
            actual_computed,
            _actual_warm,
            actual_obstacles,
            actual_groups,
        ) = kernel.support_sum_prune_pairs_mvt_warm(
            table.native_handle,
            table.query_padding,
            robot_centers,
            robot_radii,
            centers,
            shapes,
            uncertainty,
            eigenvalues,
            uncertainty_eigenvalues,
            offsets,
            np.arange(obstacle_count, dtype=np.int64),
            dense,
            contact_distance=0.0,
        )

        total_candidates = 0
        total_computed = 0
        for group, (robot_center, robot_radius) in enumerate(
            zip(robot_centers, robot_radii)
        ):
            candidates = table.query_sphere(robot_center, float(robot_radius))
            lower = (
                np.linalg.norm(centers[candidates] - robot_center, axis=1)
                - np.sqrt(np.max(eigenvalues[candidates], axis=1))
                - np.sqrt(np.max(uncertainty_eigenvalues[candidates], axis=1))
                - offsets[candidates]
                - robot_radius
            )
            ordered = candidates[np.lexsort((candidates, lower))].astype(
                np.int32
            )
            begin = int(actual_groups[group])
            end = int(actual_groups[group + 1])
            np.testing.assert_array_equal(
                actual_obstacles[begin:end], ordered
            )
            pair_centers = np.repeat(robot_center[None, :], len(ordered), axis=0)
            pair_shapes = np.repeat(
                (np.eye(3) * robot_radius**2)[None, :, :],
                len(ordered),
                axis=0,
            )
            normals, iterations, residuals = kernel.normals_sum_pairs_newton_warm(
                pair_centers,
                pair_shapes,
                centers[ordered],
                shapes[ordered],
                uncertainty[ordered],
                np.zeros((len(ordered), 3)),
                max_iterations=16,
            )
            retry = residuals > 1.0e-7
            if np.any(retry):
                retry_normals, retry_iterations, retry_residuals = (
                    kernel.normals_sum_pairs_newton_warm(
                        pair_centers[retry],
                        pair_shapes[retry],
                        centers[ordered[retry]],
                        shapes[ordered[retry]],
                        uncertainty[ordered[retry]],
                        np.zeros((int(np.count_nonzero(retry)), 3)),
                        max_iterations=64,
                    )
                )
                normals[retry] = retry_normals
                iterations[retry] += retry_iterations
                residuals[retry] = retry_residuals
            self.assertLessEqual(float(np.max(residuals)), 1.0e-7)
            obstacle_extent = np.sqrt(
                np.einsum("ni,nij,nj->n", normals, shapes[ordered], normals)
            )
            uncertainty_extent = np.sqrt(
                np.einsum(
                    "ni,nij,nj->n",
                    normals,
                    uncertainty[ordered],
                    normals,
                )
            )
            surfaces = (
                centers[ordered]
                - np.einsum("nij,nj->ni", shapes[ordered], normals)
                / obstacle_extent[:, None]
                - np.einsum("nij,nj->ni", uncertainty[ordered], normals)
                / uncertainty_extent[:, None]
                - offsets[ordered, None] * normals
            )
            plane_offsets = np.einsum("ni,ni->n", normals, surfaces)
            expected_active = list(
                map(
                    int,
                    kernel.prune_planes_sum(
                        centers[ordered],
                        shapes[ordered],
                        uncertainty[ordered],
                        offsets[ordered],
                        normals,
                        plane_offsets,
                    ),
                )
            )
            clearance = (
                np.einsum(
                    "ni,ni->n", normals, centers[ordered] - robot_center
                )
                - robot_radius
                - obstacle_extent
                - uncertainty_extent
                - offsets[ordered]
            )
            active_set = set(expected_active)
            for local in np.flatnonzero(clearance <= 0.0):
                if int(local) not in active_set:
                    expected_active.append(int(local))
                    active_set.add(int(local))
            np.testing.assert_array_equal(
                actual_active[group], expected_active
            )
            selected = begin + np.asarray(expected_active, dtype=np.int64)
            np.testing.assert_allclose(
                actual_normals[selected],
                normals[expected_active],
                rtol=0.0,
                atol=2.0e-12,
            )
            self.assertLessEqual(
                float(np.max(actual_residuals[selected])), 1.0e-7
            )
            total_candidates += len(ordered)
            total_computed += int(np.count_nonzero(actual_computed[begin:end]))
        self.assertLess(total_computed, total_candidates)

    def test_fused_sphere_mvt_pruning_matches_python_reference(self):
        rng = np.random.default_rng(20260831)
        obstacle_count = 1200
        centers = rng.uniform(
            [-0.35, -0.45, 0.05], [0.85, 0.65, 1.05],
            size=(obstacle_count, 3),
        )
        radii = rng.uniform(0.004, 0.035, obstacle_count)
        offsets = rng.uniform(0.0, 0.004, obstacle_count)
        robot_centers = rng.uniform(
            [-0.20, -0.30, 0.15], [0.70, 0.50, 0.90],
            size=(17, 3),
        )
        robot_radii = rng.uniform(0.018, 0.055, len(robot_centers))
        query_padding = 0.045
        table = NativeMultilevelMVT.from_spheres(
            centers,
            radii + offsets,
            base_voxel_size=0.0125,
            maximum_query_half_extent=0.11,
            query_padding=query_padding,
            simd=True,
        )
        self.addCleanup(table.close)
        (
            active_groups,
            normals,
            surfaces,
            ordered_obstacles,
            groups,
        ) = NativeEllipsoidSupport().sphere_prune_pairs_mvt(
            table.native_handle,
            table.query_padding,
            robot_centers,
            robot_radii,
            centers,
            radii,
            offsets,
            contact_distance=0.0,
        )

        for group, (point, robot_radius) in enumerate(
            zip(robot_centers, robot_radii)
        ):
            candidates = table.query_sphere(point, float(robot_radius))
            lower = (
                np.linalg.norm(centers[candidates] - point, axis=1)
                - radii[candidates]
                - offsets[candidates]
            )
            expected_order = candidates[
                np.lexsort((candidates, lower))
            ].astype(np.int32)
            begin = int(groups[group])
            end = int(groups[group + 1])
            np.testing.assert_array_equal(
                ordered_obstacles[begin:end], expected_order
            )
            delta = centers[expected_order] - point
            distance = np.linalg.norm(delta, axis=1)
            expected_normals = np.zeros_like(delta)
            expected_normals[:, 0] = 1.0
            nonzero = distance > 1.0e-12
            expected_normals[nonzero] = (
                delta[nonzero] / distance[nonzero, None]
            )
            expected_surfaces = (
                centers[expected_order]
                - radii[expected_order, None] * expected_normals
            )
            np.testing.assert_allclose(
                normals[begin:end], expected_normals, rtol=0.0, atol=2.0e-15
            )
            np.testing.assert_allclose(
                surfaces[begin:end], expected_surfaces, rtol=0.0, atol=2.0e-15
            )

            remaining = list(range(len(expected_order)))
            expected_active = []
            while remaining:
                nearest = remaining[0]
                expected_active.append(nearest)
                normal = expected_normals[nearest]
                obstacle = expected_order[nearest]
                plane_offset = (
                    centers[obstacle] @ normal
                    - radii[obstacle]
                    - offsets[obstacle]
                )
                local = np.asarray(remaining, dtype=np.int64)
                local_obstacles = expected_order[local]
                minimum_support = (
                    centers[local_obstacles] @ normal
                    - radii[local_obstacles]
                    - offsets[local_obstacles]
                )
                remove = minimum_support >= plane_offset - 1.0e-10
                remaining = [
                    item
                    for item, removed in zip(remaining, remove)
                    if item != nearest and not bool(removed)
                ]
            contact = np.flatnonzero(
                distance
                - radii[expected_order]
                - offsets[expected_order]
                - robot_radius
                <= 0.0
            )
            expected_active.extend(
                int(item)
                for item in contact
                if int(item) not in set(expected_active)
            )
            np.testing.assert_array_equal(
                active_groups[group], expected_active
            )

    def test_pair_thread_configuration_is_explicit_and_validated(self):
        kernel = NativeEllipsoidSupport()
        try:
            self.assertEqual(kernel.set_pair_threads(7), 7)
            self.assertEqual(kernel.pair_threads, 7)
            self.assertEqual(kernel.set_pair_affinity_mask(0x3F), 0x3F)
            self.assertEqual(kernel.pair_affinity_mask, 0x3F)
            with self.assertRaises(ValueError):
                kernel.set_pair_threads(0)
            with self.assertRaises(ValueError):
                kernel.set_pair_affinity_mask(-1)
        finally:
            kernel.set_pair_threads(8)
            kernel.set_pair_affinity_mask(0)

    def test_pairwise_closest_points_match_individual_exact_batches(self):
        rng = np.random.default_rng(1907)
        count = 257
        points = rng.uniform([-0.4, -0.4, 0.05], [0.4, 0.4, 0.8], (count, 3))
        centers = points + rng.normal(size=(count, 3)) * 0.12
        axes = rng.uniform(0.004, 0.06, (count, 3))
        rotations = np.empty((count, 3, 3), dtype=float)
        for index in range(count):
            rotations[index], _ = np.linalg.qr(rng.normal(size=(3, 3)))
        values = np.square(axes)
        initial = np.full(count, np.nan, dtype=float)
        kernel = NativeEllipsoidSupport()
        actual = kernel.closest_points_pairs_warm(
            points, centers, values, rotations, initial
        )
        expected_rows = [
            kernel.closest_points_warm(
                points[index],
                centers[index : index + 1],
                values[index : index + 1],
                rotations[index : index + 1],
                initial[index : index + 1],
            )
            for index in range(count)
        ]
        for field_index in range(6):
            expected = np.concatenate(
                [row[field_index] for row in expected_rows], axis=0
            )
            np.testing.assert_allclose(actual[field_index], expected, rtol=0.0, atol=0.0)

    def test_batched_closest_points_match_safeguarded_python_reference(self):
        rng = np.random.default_rng(907)
        point = np.array([0.42, -0.31, 0.57])
        centers = rng.uniform([-0.15, -0.12, 0.18], [0.12, 0.16, 0.40], (193, 3))
        shapes = []
        for _ in centers:
            basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            axes = rng.uniform(0.004, 0.075, 3)
            shapes.append(basis @ np.diag(axes**2) @ basis.T)
        shapes = np.asarray(shapes)
        eigenvalues, rotations = np.linalg.eigh(shapes)
        cold = [
            closest_point_on_ellipsoid(center, shape, point)
            for center, shape in zip(centers, shapes)
        ]
        initial = np.asarray([item.multiplier for item in cold]) * 1.013
        actual = NativeEllipsoidSupport().closest_points_warm(
            point, centers, eigenvalues, rotations, initial
        )
        normals, surfaces, multipliers, newton, bisection, residuals = actual
        expected = [
            closest_point_on_ellipsoid(
                center,
                shape,
                point,
                initial_multiplier=warm,
                eigenvalues=values,
                rotation=rotation,
            )
            for center, shape, warm, values, rotation in zip(
                centers, shapes, initial, eigenvalues, rotations
            )
        ]
        np.testing.assert_allclose(
            normals, np.asarray([item.normal for item in expected]),
            rtol=2.0e-10, atol=2.0e-10,
        )
        np.testing.assert_allclose(
            surfaces, np.asarray([item.surface_point for item in expected]),
            rtol=2.0e-10, atol=2.0e-10,
        )
        np.testing.assert_allclose(
            multipliers, np.asarray([item.multiplier for item in expected]),
            rtol=2.0e-10, atol=2.0e-12,
        )
        self.assertLessEqual(float(np.max(residuals)), 1.1e-12)
        self.assertTrue(np.all(newton <= 10))
        self.assertTrue(np.all(bisection <= 58))

    def test_batch_matches_python_reference(self):
        rng = np.random.default_rng(17)
        robot_center = np.array([0.1, -0.2, 0.4])
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        robot_shape = rotation @ np.diag(np.array([0.018, 0.027, 0.044]) ** 2) @ rotation.T
        centers = rng.uniform([-0.3, -0.5, 0.1], [0.7, 0.5, 0.9], size=(128, 3))
        shapes = []
        for _ in centers:
            basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            axes = rng.uniform(0.008, 0.09, size=3)
            shapes.append(basis @ np.diag(axes**2) @ basis.T)
        shapes = np.asarray(shapes)
        actual, iterations = NativeEllipsoidSupport().normals(
            robot_center, robot_shape, centers, shapes
        )
        expected = np.asarray(
            [
                optimal_support_separating_normal(
                    robot_center, robot_shape, center, shape
                )
                for center, shape in zip(centers, shapes)
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=2.0e-11)
        self.assertTrue(np.all(iterations <= 24))

    def test_native_plane_pruning_matches_python_sequence(self):
        rng = np.random.default_rng(23)
        count = 257
        centers = rng.normal(size=(count, 3)) * np.array([0.2, 0.3, 0.1])
        axes = rng.uniform(0.005, 0.05, size=(count, 3))
        shapes = np.asarray([np.diag(item**2) for item in axes])
        offsets = rng.uniform(0.0, 0.01, size=count)
        normals = rng.normal(size=(count, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        extents = np.sqrt(np.einsum("ni,nij,nj->n", normals, shapes, normals)) + offsets
        plane_offsets = np.einsum("ni,ni->n", normals, centers) - extents
        remaining = list(range(count))
        expected = []
        while remaining:
            nearest = remaining[0]
            expected.append(nearest)
            normal = normals[nearest]
            projected = (
                centers[remaining] @ normal
                - np.sqrt(np.einsum("i,nij,j->n", normal, shapes[remaining], normal))
                - offsets[remaining]
            )
            behind = projected >= plane_offsets[nearest] - 1.0e-10
            remaining = [item for item, remove in zip(remaining, behind) if not remove]
        actual = NativeEllipsoidSupport().prune_planes(
            centers, shapes, offsets, normals, plane_offsets
        )
        np.testing.assert_array_equal(actual, expected)

    def test_grouped_plane_pruning_matches_individual_native_calls(self):
        rng = np.random.default_rng(2308)
        counts = np.array([0, 17, 3, 65, 1, 84, 9], dtype=np.int32)
        groups = np.r_[0, np.cumsum(counts)].astype(np.int32)
        count = int(groups[-1])
        centers = rng.normal(size=(count, 3)) * 0.2
        axes = rng.uniform(0.004, 0.05, (count, 3))
        shapes = np.asarray([np.diag(item**2) for item in axes])
        offsets = rng.uniform(0.0, 0.01, count)
        normals = rng.normal(size=(count, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        plane_offsets = (
            np.einsum("ni,ni->n", normals, centers)
            - np.sqrt(np.einsum("ni,nij,nj->n", normals, shapes, normals))
            - offsets
        )
        kernel = NativeEllipsoidSupport()
        actual = kernel.prune_planes_grouped(
            centers, shapes, offsets, normals, plane_offsets, groups
        )
        expected = [
            kernel.prune_planes(
                centers[groups[index] : groups[index + 1]],
                shapes[groups[index] : groups[index + 1]],
                offsets[groups[index] : groups[index + 1]],
                normals[groups[index] : groups[index + 1]],
                plane_offsets[groups[index] : groups[index + 1]],
            )
            for index in range(len(counts))
        ]
        for grouped, individual in zip(actual, expected):
            np.testing.assert_array_equal(grouped, individual)

    def test_native_support_sum_plane_pruning_matches_python_sequence(self):
        rng = np.random.default_rng(2301)
        count = 257
        centers = rng.normal(size=(count, 3)) * np.array([0.2, 0.3, 0.1])
        axes = rng.uniform(0.005, 0.05, size=(count, 3))
        uncertainty_axes = rng.uniform(0.001, 0.018, size=(count, 3))
        shapes = np.asarray([np.diag(item**2) for item in axes])
        uncertainty = np.asarray(
            [np.diag(item**2) for item in uncertainty_axes]
        )
        offsets = rng.uniform(0.0, 0.01, size=count)
        normals = rng.normal(size=(count, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        extents = (
            np.sqrt(np.einsum("ni,nij,nj->n", normals, shapes, normals))
            + np.sqrt(
                np.einsum("ni,nij,nj->n", normals, uncertainty, normals)
            )
            + offsets
        )
        plane_offsets = np.einsum("ni,ni->n", normals, centers) - extents
        remaining = list(range(count))
        expected = []
        while remaining:
            nearest = remaining[0]
            expected.append(nearest)
            normal = normals[nearest]
            projected = (
                centers[remaining] @ normal
                - np.sqrt(
                    np.einsum(
                        "i,nij,j->n", normal, shapes[remaining], normal
                    )
                )
                - np.sqrt(
                    np.einsum(
                        "i,nij,j->n", normal, uncertainty[remaining], normal
                    )
                )
                - offsets[remaining]
            )
            behind = projected >= plane_offsets[nearest] - 1.0e-10
            remaining = [
                item for item, remove in zip(remaining, behind) if not remove
            ]
        actual = NativeEllipsoidSupport().prune_planes_sum(
            centers,
            shapes,
            uncertainty,
            offsets,
            normals,
            plane_offsets,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_support_sum_matches_python_reference(self):
        rng = np.random.default_rng(41)
        robot_center = np.array([0.1, -0.2, 0.4])
        robot_shape = np.diag(np.array([0.018, 0.031, 0.047]) ** 2)
        centers = rng.uniform([-0.3, -0.5, 0.1], [0.7, 0.5, 0.9], size=(97, 3))
        shapes = np.asarray([np.diag(rng.uniform(0.008, 0.09, 3) ** 2) for _ in centers])
        uncertainty = np.asarray(
            [np.diag(rng.uniform(0.0005, 0.016, 3) ** 2) for _ in centers]
        )
        actual, iterations = NativeEllipsoidSupport().normals_sum(
            robot_center, robot_shape, centers, shapes, uncertainty
        )
        expected = np.asarray(
            [
                optimal_support_sum_separating_normal(
                    robot_center, robot_shape, center, shape, error
                )
                for center, shape, error in zip(centers, shapes, uncertainty)
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=2.0e-11)
        self.assertTrue(np.all(iterations <= 24))

    def test_pairwise_warm_batch_matches_one_robot_batches(self):
        rng = np.random.default_rng(419)
        count = 73
        robot_centers = rng.normal(size=(count, 3)) * 0.2
        robot_shapes = np.asarray(
            [np.diag(rng.uniform(0.01, 0.05, 3) ** 2) for _ in range(count)]
        )
        obstacle_centers = rng.normal(size=(count, 3)) * 0.3
        obstacle_shapes = np.asarray(
            [np.diag(rng.uniform(0.006, 0.08, 3) ** 2) for _ in range(count)]
        )
        uncertainty = np.asarray(
            [np.diag(rng.uniform(0.0005, 0.012, 3) ** 2) for _ in range(count)]
        )
        initial = rng.normal(size=(count, 3))
        initial /= np.linalg.norm(initial, axis=1)[:, None]
        kernel = NativeEllipsoidSupport()
        actual = kernel.normals_sum_pairs_warm(
            robot_centers,
            robot_shapes,
            obstacle_centers,
            obstacle_shapes,
            uncertainty,
            initial,
            max_iterations=64,
        )
        expected = [
            kernel.normals_sum_warm(
                robot_centers[index],
                robot_shapes[index],
                obstacle_centers[index : index + 1],
                obstacle_shapes[index : index + 1],
                uncertainty[index : index + 1],
                initial[index : index + 1],
                max_iterations=64,
            )
            for index in range(count)
        ]
        np.testing.assert_allclose(
            actual[0], np.concatenate([item[0] for item in expected]), atol=1.0e-14
        )
        np.testing.assert_array_equal(
            actual[1], np.concatenate([item[1] for item in expected])
        )
        np.testing.assert_allclose(
            actual[2], np.concatenate([item[2] for item in expected]), atol=1.0e-14
        )

    def test_newton_support_is_more_accurate_and_pairwise_deterministic(self):
        rng = np.random.default_rng(20260830)
        count = 128
        robot_centers = rng.normal(size=(count, 3)) * 0.2
        radii = rng.uniform(0.02, 0.06, count)
        robot_shapes = np.eye(3)[None, :, :] * radii[:, None, None] ** 2
        obstacle_centers = robot_centers + rng.normal(size=(count, 3)) * 0.18
        obstacle_shapes = []
        uncertainty_shapes = []
        for _ in range(count):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            axes = rng.uniform(0.004, 0.08, 3)
            obstacle_shapes.append(rotation @ np.diag(axes**2) @ rotation.T)
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            axes = rng.uniform(0.001, 0.018, 3)
            uncertainty_shapes.append(rotation @ np.diag(axes**2) @ rotation.T)
        obstacle_shapes = np.asarray(obstacle_shapes)
        uncertainty_shapes = np.asarray(uncertainty_shapes)
        initial = obstacle_centers - robot_centers
        initial /= np.linalg.norm(initial, axis=1)[:, None]
        kernel = NativeEllipsoidSupport()
        gradient = kernel.normals_sum_pairs_warm(
            robot_centers,
            robot_shapes,
            obstacle_centers,
            obstacle_shapes,
            uncertainty_shapes,
            initial,
            max_iterations=64,
        )
        newton = kernel.normals_sum_pairs_newton_warm(
            robot_centers,
            robot_shapes,
            obstacle_centers,
            obstacle_shapes,
            uncertainty_shapes,
            initial,
            max_iterations=16,
        )

        def objective(normals):
            delta = obstacle_centers - robot_centers
            return (
                np.einsum("ni,ni->n", normals, delta)
                - np.sqrt(np.einsum("ni,nij,nj->n", normals, robot_shapes, normals))
                - np.sqrt(
                    np.einsum("ni,nij,nj->n", normals, obstacle_shapes, normals)
                )
                - np.sqrt(
                    np.einsum(
                        "ni,nij,nj->n", normals, uncertainty_shapes, normals
                    )
                )
            )

        self.assertTrue(
            np.all(objective(newton[0]) >= objective(gradient[0]) - 1.0e-12)
        )
        np.testing.assert_allclose(
            np.linalg.norm(newton[0], axis=1), 1.0, atol=3.0e-15
        )
        self.assertLessEqual(float(np.max(newton[2])), 2.0e-8)
        expected = [
            kernel.normals_sum_newton_warm(
                robot_centers[index],
                robot_shapes[index],
                obstacle_centers[index : index + 1],
                obstacle_shapes[index : index + 1],
                uncertainty_shapes[index : index + 1],
                initial[index : index + 1],
                max_iterations=16,
            )
            for index in range(count)
        ]
        np.testing.assert_allclose(
            newton[0], np.concatenate([item[0] for item in expected]), atol=1.0e-14
        )
        np.testing.assert_array_equal(
            newton[1], np.concatenate([item[1] for item in expected])
        )
        np.testing.assert_allclose(
            newton[2], np.concatenate([item[2] for item in expected]), atol=1.0e-14
        )

    def test_zero_warm_rows_start_on_unit_center_line(self):
        kernel = NativeEllipsoidSupport()
        robot_centers = np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.4]])
        obstacle_centers = np.array([[0.7, 0.2, -0.1], [-0.3, 0.8, 0.9]])
        robot_shapes = np.repeat((0.04**2 * np.eye(3))[None, :, :], 2, axis=0)
        obstacle_shapes = np.repeat(
            np.diag(np.array([0.08, 0.03, 0.02]) ** 2)[None, :, :],
            2,
            axis=0,
        )
        uncertainty_shapes = np.repeat(
            np.diag(np.array([0.006, 0.004, 0.003]) ** 2)[None, :, :],
            2,
            axis=0,
        )
        normals, iterations, _ = kernel.normals_sum_pairs_newton_warm(
            robot_centers,
            robot_shapes,
            obstacle_centers,
            obstacle_shapes,
            uncertainty_shapes,
            np.zeros((2, 3)),
            max_iterations=0,
        )
        expected = obstacle_centers - robot_centers
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(normals, expected, atol=2.0e-15)
        np.testing.assert_array_equal(iterations, np.zeros(2, dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
