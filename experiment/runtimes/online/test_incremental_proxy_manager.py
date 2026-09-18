from __future__ import annotations

import unittest

import numpy as np

from incremental_proxy_manager import IncrementalMatchedProxyManager
from pointcloud_proxy import _split_by_normal_similarity
from pointcloud_proxy import (
    certified_translated_ellipsoid_outer,
    loewner_union_outer_shape,
    loewner_union_outer_shapes_grouped,
    loewner_union_outer_shapes_grouped_common_frame,
    minkowski_outer_shape,
)


class IncrementalMatchedProxyManagerTests(unittest.TestCase):
    def test_partitioned_directional_centervox_keeps_incompatible_views_separate(self):
        manager = IncrementalMatchedProxyManager(
            filter_size=0.02,
            cluster_size=0.10,
            maximum_uncertainty_union_inflation=1.05,
            uncertainty_fusion_mode="partitioned_separate_uncertainty",
            origin=np.zeros(3),
        )
        point = np.array([[0.105, 0.205, 0.405]])
        first = np.diag([0.006**2, 0.004**2, 0.0005**2])
        rotation = np.array(
            [[np.cos(0.30), 0.0, np.sin(0.30)],
             [0.0, 1.0, 0.0],
             [-np.sin(0.30), 0.0, np.cos(0.30)]]
        )
        second = rotation @ first @ rotation.T
        manager.update(point, np.zeros(1), first[None])
        manager.update(point, np.zeros(1), second[None])

        self.assertEqual(len(manager._spatial_cells), 1)
        spatial = next(iter(manager._spatial_cells.values()))
        self.assertEqual(len(spatial.samples), 2)
        self.assertEqual(len(manager.snapshot.filtered_points), 2)
        self.assertTrue(manager.coverage_audit().all_raw_sample_balls_certified)
        maximum_axes = np.sqrt(
            np.maximum(
                np.linalg.eigvalsh(manager.snapshot.filtered_uncertainty_shapes)[:, -1],
                0.0,
            )
        )
        self.assertLessEqual(float(np.max(maximum_axes)), 0.006 + 1.0e-9)

    def test_partitioned_directional_centervox_deletes_contained_view(self):
        manager = IncrementalMatchedProxyManager(
            filter_size=0.02,
            cluster_size=0.10,
            maximum_uncertainty_union_inflation=1.05,
            uncertainty_fusion_mode="partitioned_separate_uncertainty",
            origin=np.zeros(3),
        )
        point = np.array([[0.105, 0.205, 0.405]])
        outer = np.diag([0.006**2, 0.004**2, 0.001**2])
        inner = outer * 0.64
        manager.update(point, np.zeros(1), outer[None])
        manager.update(point, np.zeros(1), inner[None])

        spatial = next(iter(manager._spatial_cells.values()))
        self.assertEqual(len(spatial.samples), 1)
        retained = next(iter(spatial.samples.values()))
        self.assertEqual(retained.source_count, 2)
        self.assertEqual(retained.source_generations, frozenset({0, 1}))
        np.testing.assert_allclose(
            retained.uncertainty_shape, outer, rtol=3.0e-7, atol=1.0e-18
        )
        self.assertTrue(manager.coverage_audit().all_raw_sample_balls_certified)

    def test_radius_cap_sphere_candidate_family_is_irredundant(self):
        points = np.asarray(
            [[x, y, 0.3] for x in np.linspace(-0.03, 0.03, 7) for y in np.linspace(-0.03, 0.03, 7)]
        )
        shapes = np.repeat(np.eye(3)[None] * 4.0e-6, len(points), axis=0)
        manager = IncrementalMatchedProxyManager(
            filter_size=0.01,
            cluster_size=0.05,
            certificate_radius_limit=0.05,
            uncertainty_fusion_mode="separate_uncertainty",
            sphere_cover_mode="cap_irredundant",
            radius_limit_representation="sphere",
        )
        manager.update(points, np.zeros(len(points)), shapes)
        audit = manager.coverage_audit()
        self.assertTrue(audit.all_raw_sample_balls_certified)
        self.assertTrue(audit.sphere_cover_inclusion_minimal)
        self.assertLessEqual(float(np.max(manager.snapshot.sphere_radii)), 0.05 + 1e-12)

    def test_adaptive_ellipsoid_cover_is_safe_and_inclusion_minimal(self):
        grid = np.linspace(-0.04, 0.04, 9)
        points = np.asarray(
            [[x, y, 0.35] for x in grid for y in grid], dtype=float
        )
        shapes = np.repeat(
            np.diag([2.5e-5, 2.5e-5, 4.0e-6])[None], len(points), axis=0
        )
        manager = IncrementalMatchedProxyManager(
            filter_size=0.01,
            cluster_size=0.05,
            certificate_radius_limit=0.07,
            maximum_uncertainty_union_inflation=1.05,
            uncertainty_fusion_mode="separate_uncertainty",
            ellipsoid_cover_mode="adaptive_irredundant",
        )
        stats = manager.update(points, np.zeros(len(points)), shapes)
        audit = manager.coverage_audit()
        self.assertTrue(audit.all_raw_sample_balls_certified)
        self.assertTrue(audit.certificate_cover_inclusion_minimal)
        self.assertGreater(stats.certificate_cover_candidate_count, 0)
        self.assertEqual(
            stats.certificate_cover_selected_count, len(manager.snapshot.centers)
        )
        self.assertGreaterEqual(
            stats.certificate_cover_minimum_unique_witnesses, 1
        )

    def test_separate_uncertainty_accepts_explicit_identity_direct_inflation(self):
        manager = IncrementalMatchedProxyManager(
            filter_size=0.02,
            cluster_size=0.10,
            uncertainty_fusion_mode="separate_uncertainty",
            direct_thin_axis_inflation=1.0,
            origin=np.zeros(3),
        )
        self.assertEqual(manager.direct_thin_axis_inflation, 1.0)

        with self.assertRaisesRegex(ValueError, "must exceed one"):
            IncrementalMatchedProxyManager(
                filter_size=0.02,
                cluster_size=0.10,
                uncertainty_fusion_mode="fused_certified_centervox",
                direct_thin_axis_inflation=1.0,
                origin=np.zeros(3),
            )

    def test_certified_translated_ellipsoid_outer_contains_member_surfaces(self):
        rng = np.random.default_rng(20260830)
        centers = rng.normal(size=(31, 3)) * np.array([0.025, 0.018, 0.0004])
        shapes = []
        for _ in centers:
            generator = rng.normal(size=(3, 3)) * np.array(
                [[0.003], [0.002], [0.0003]]
            )
            shapes.append(generator @ generator.T + np.eye(3) * 1.0e-10)
        shapes = np.asarray(shapes)
        center, outer, _rotation, _half = certified_translated_ellipsoid_outer(
            centers, shapes
        )
        for member_center, member_shape in zip(centers, shapes):
            values, vectors = np.linalg.eigh(member_shape)
            root = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))
            for direction in rng.normal(size=(256, 3)):
                direction /= np.linalg.norm(direction)
                sample = member_center + root @ direction
                delta = sample - center
                value = float(delta @ np.linalg.solve(outer, delta))
                self.assertLessEqual(value, 1.0 + 1.0e-10)

    def test_fused_mode_preserves_planar_thin_axis_and_publishes_no_u(self):
        manager = IncrementalMatchedProxyManager(
            filter_size=0.02,
            cluster_size=0.10,
            normal_connection_distance=0.04,
            uncertainty_fusion_mode="fused_certified_ellipsoid",
            origin=np.zeros(3),
        )
        x, y = np.meshgrid(np.linspace(0.101, 0.139, 5), np.linspace(0.201, 0.239, 5))
        points = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.401)])
        shape = np.diag([0.003**2, 0.004**2, 0.0003**2])
        shapes = np.repeat(shape[None, :, :], len(points), axis=0)
        for frame_offset in (0.0, 0.0002, -0.0002):
            frame = points + np.array([0.001 * frame_offset, 0.0, frame_offset])
            manager.update(frame, np.full(len(frame), 0.004), shapes)
        proxies = manager.snapshot
        self.assertEqual(
            proxies.uncertainty_fusion_mode, "fused_certified_ellipsoid"
        )
        np.testing.assert_array_equal(
            proxies.proxy_uncertainty_shapes,
            np.zeros_like(proxies.proxy_uncertainty_shapes),
        )
        np.testing.assert_allclose(
            proxies.ellipsoid_outer_shapes,
            proxies.base_ellipsoid_shapes,
            atol=0.0,
            rtol=0.0,
        )
        # Tangential spread is centimetric, while the normal support remains
        # tied to the sub-millimetre depth uncertainty instead of inheriting it.
        normal_support = np.sqrt(
            np.maximum(proxies.base_ellipsoid_shapes[:, 2, 2], 0.0)
        )
        self.assertLess(float(np.max(normal_support)), 0.003)
        audit = manager.coverage_audit()
        self.assertTrue(audit.all_raw_sample_balls_certified)
        self.assertGreaterEqual(audit.minimum_centervox_cover_slack, -1.0e-12)
        self.assertGreaterEqual(audit.minimum_proxy_uncertainty_slack, -1.0e-12)

    def test_direct_fused_centervox_publishes_one_thin_proxy_per_cell(self):
        manager = IncrementalMatchedProxyManager(
            filter_size=0.02,
            cluster_size=0.10,
            normal_connection_distance=0.04,
            uncertainty_fusion_mode="fused_certified_centervox",
            origin=np.zeros(3),
        )
        points = np.array(
            [[0.011 + 0.021 * i, 0.011 + 0.021 * j, 0.401] for i in range(3) for j in range(3)]
        )
        shape = np.diag([0.003**2, 0.004**2, 0.0003**2])
        shapes = np.repeat(shape[None], len(points), axis=0)
        manager.update(points, np.full(len(points), 0.004), shapes)
        proxies = manager.snapshot
        self.assertEqual(len(proxies.centers), len(manager._cells))
        np.testing.assert_array_equal(
            proxies.filtered_cluster_indices, np.arange(len(proxies.centers))
        )
        np.testing.assert_array_equal(
            proxies.proxy_uncertainty_shapes,
            np.zeros_like(proxies.proxy_uncertainty_shapes),
        )
        self.assertTrue(manager.coverage_audit().all_raw_sample_balls_certified)

    def test_partitioned_direct_cover_caps_thin_axis_without_coverage_gaps(self):
        gamma = 1.10
        subdivisions = 2
        manager = IncrementalMatchedProxyManager(
            filter_size=0.02,
            cluster_size=0.10,
            normal_connection_distance=0.04,
            uncertainty_fusion_mode="fused_certified_centervox",
            direct_thin_axis_inflation=gamma,
            direct_tangent_subdivisions=subdivisions,
            origin=np.zeros(3),
        )
        points = np.array(
            [
                [0.002, 0.003, 0.4010],
                [0.018, 0.004, 0.4012],
                [0.003, 0.017, 0.4008],
                [0.017, 0.018, 0.4011],
            ]
        )
        measurement = np.diag([0.0015**2, 0.0018**2, 0.0004**2])
        shapes = np.repeat(measurement[None], len(points), axis=0)
        manager.update(points, np.full(len(points), 0.0018), shapes)
        proxies = manager.snapshot
        self.assertEqual(len(manager._cells), 1)
        self.assertEqual(len(proxies.centers), subdivisions**2)
        self.assertEqual(len(np.unique(proxies.proxy_ids)), subdivisions**2)
        self.assertTrue(manager.coverage_audit().all_raw_sample_balls_certified)

        cell = next(iter(manager._cells.values()))
        rotation = np.asarray(cell.fusion_rotation)
        lower = np.asarray(cell.fusion_lower)
        upper = np.asarray(cell.fusion_upper)
        narrow = int(cell.fusion_narrow_axis)
        normal = rotation[:, narrow]
        half = 0.5 * (upper - lower)
        normal_support = np.sqrt(
            np.einsum(
                "i,nij,j->n",
                normal,
                proxies.base_ellipsoid_shapes,
                normal,
            )
        )
        np.testing.assert_allclose(
            normal_support,
            np.full(subdivisions**2, gamma * max(half[narrow], 1.0e-7)),
            atol=1.0e-12,
            rtol=1.0e-12,
        )

        # Every point of a dense support-box grid must lie in at least one
        # published subproxy. This independently exercises all partition
        # boundaries, including shared edges and the original box corners.
        axes_samples = [
            np.linspace(lower[axis], upper[axis], 9) for axis in range(3)
        ]
        local = np.array(
            np.meshgrid(*axes_samples, indexing="ij")
        ).reshape(3, -1).T
        world = local @ rotation.T
        covered = np.zeros(len(world), dtype=bool)
        for center, shape in zip(
            proxies.centers, proxies.base_ellipsoid_shapes
        ):
            delta = world - center
            values = np.einsum(
                "ni,ij,nj->n", delta, np.linalg.inv(shape), delta
            )
            covered |= values <= 1.0 + 1.0e-10
        self.assertTrue(bool(np.all(covered)))

    def test_longest_tangent_binary_cover_uses_two_non_nested_proxies(self):
        manager = IncrementalMatchedProxyManager(
            filter_size=0.02,
            cluster_size=0.10,
            uncertainty_fusion_mode="fused_certified_centervox",
            direct_thin_axis_inflation=1.10,
            direct_tangent_subdivisions=1,
            direct_partition_mode="longest_tangent_binary",
            origin=np.zeros(3),
        )
        points = np.array(
            [
                [0.001, 0.002, 0.4008],
                [0.019, 0.003, 0.4012],
                [0.002, 0.018, 0.4009],
                [0.018, 0.019, 0.4011],
            ]
        )
        measurement = np.diag([0.0015**2, 0.0018**2, 0.0004**2])
        shapes = np.repeat(measurement[None], len(points), axis=0)
        manager.update(points, np.full(len(points), 0.0018), shapes)
        proxies = manager.snapshot
        self.assertEqual(len(manager._cells), 1)
        self.assertEqual(len(proxies.centers), 2)
        self.assertGreater(
            float(np.linalg.norm(proxies.centers[0] - proxies.centers[1])),
            1.0e-6,
        )
        self.assertTrue(manager.coverage_audit().all_raw_sample_balls_certified)

        cell = next(iter(manager._cells.values()))
        rotation = np.asarray(cell.fusion_rotation)
        lower = np.asarray(cell.fusion_lower)
        upper = np.asarray(cell.fusion_upper)
        local_axes = np.sqrt(
            np.maximum(
                np.einsum(
                    "ik,nij,jk->nk",
                    rotation,
                    proxies.base_ellipsoid_shapes,
                    rotation,
                ),
                0.0,
            )
        )
        narrow = int(cell.fusion_narrow_axis)
        expected_normal = 1.10 * max(
            0.5 * (upper[narrow] - lower[narrow]), 1.0e-7
        )
        np.testing.assert_allclose(
            local_axes[:, narrow], expected_normal, atol=1.0e-12
        )
        tangent = [axis for axis in range(3) if axis != narrow]
        np.testing.assert_allclose(
            local_axes[:, tangent[0]],
            local_axes[:, tangent[1]],
            atol=1.0e-12,
            rtol=1.0e-12,
        )

        # Independently verify the union certificate, including the split
        # plane and all original support-box corners.  The two shifted
        # ellipsoids must cover the box; neither may be a hidden, concentric
        # re-inflation of the other.
        axes_samples = [
            np.linspace(lower[axis], upper[axis], 9) for axis in range(3)
        ]
        local = np.array(
            np.meshgrid(*axes_samples, indexing="ij")
        ).reshape(3, -1).T
        world = local @ rotation.T
        covered = np.zeros(len(world), dtype=bool)
        for center, shape in zip(
            proxies.centers, proxies.base_ellipsoid_shapes
        ):
            delta = world - center
            values = np.einsum(
                "ni,ij,nj->n", delta, np.linalg.inv(shape), delta
            )
            covered |= values <= 1.0 + 1.0e-10
        self.assertTrue(bool(np.all(covered)))

    def test_normal_components_do_not_chain_across_a_corner(self):
        angles = np.deg2rad([0.0, 20.0, 40.0])
        normals = np.c_[np.cos(angles), np.sin(angles), np.zeros(3)]
        points = np.c_[np.arange(3) * 0.005, np.zeros(3), np.zeros(3)]
        groups = _split_by_normal_similarity(
            np.arange(3), points, normals, connection_distance=0.006,
            maximum_angle_degrees=25.0,
        )
        self.assertEqual(sorted(len(group) for group in groups), [1, 2])

    def test_cluster_partition_uses_fixed_world_origin(self):
        points = np.array(
            [[0.101, 0.001, 0.201], [0.109, 0.001, 0.201], [0.121, 0.001, 0.201]]
        )
        radii = np.full(len(points), 0.001)
        first = IncrementalMatchedProxyManager(
            filter_size=0.004, cluster_size=0.02, origin=np.zeros(3)
        )
        second = IncrementalMatchedProxyManager(
            filter_size=0.004, cluster_size=0.02, origin=np.zeros(3)
        )
        first.update(points, radii)
        # Discovering a smaller point in a different cell must not move the
        # existing 0.10/0.12 m cluster boundary.
        second.update(np.vstack([[-0.317, 0.001, 0.201], points]), np.r_[0.001, radii])
        a = first.snapshot.filtered_cluster_indices
        b = second.snapshot.filtered_cluster_indices[1:]
        self.assertEqual(bool(a[0] == a[1]), bool(b[0] == b[1]))
        self.assertEqual(bool(a[1] == a[2]), bool(b[1] == b[2]))

    def test_directional_outer_shapes_satisfy_support_and_loewner_bounds(self):
        q1 = np.diag([0.004**2, 0.009**2, 0.002**2])
        rotation = np.array(
            [[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]]
        )
        q2 = rotation @ np.diag([0.007**2, 0.003**2, 0.005**2]) @ rotation.T
        minkowski = minkowski_outer_shape(q1, q2)
        union = loewner_union_outer_shape(np.asarray([q1, q2]))
        rng = np.random.default_rng(7)
        for direction in rng.normal(size=(1000, 3)):
            direction /= np.linalg.norm(direction)
            lhs = np.sqrt(direction @ q1 @ direction) + np.sqrt(
                direction @ q2 @ direction
            )
            rhs = np.sqrt(direction @ minkowski @ direction)
            self.assertLessEqual(lhs, rhs + 1.0e-12)
        for shape in (q1, q2):
            generalized = np.linalg.eigvalsh(
                np.linalg.solve(union, shape)
            )
            self.assertLessEqual(float(np.max(generalized.real)), 1.0 + 1.0e-9)
        reverse = loewner_union_outer_shape(np.asarray([q2, q1]))
        np.testing.assert_allclose(union, reverse, atol=1.0e-14)

    def test_loewner_join_does_not_globally_scale_rotated_thin_axes(self):
        angles = np.linspace(-0.12, 0.12, 17)
        shapes = []
        for angle in angles:
            rotation = np.array(
                [
                    [np.cos(angle), 0.0, np.sin(angle)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(angle), 0.0, np.cos(angle)],
                ]
            )
            shapes.append(
                rotation
                @ np.diag([0.006**2, 0.004**2, 0.0002**2])
                @ rotation.T
            )
        union = loewner_union_outer_shape(np.asarray(shapes))
        for shape in shapes:
            self.assertGreaterEqual(
                float(np.min(np.linalg.eigvalsh(union - shape))), -1.0e-13
            )
        # A 0.24-rad fan needs finite thickness, but not a centimetre-scale
        # expansion of every semi-axis.
        self.assertLess(
            float(np.sqrt(np.max(np.linalg.eigvalsh(union)))), 0.009
        )

    def test_grouped_loewner_join_contains_every_member(self):
        rng = np.random.default_rng(211)
        groups = np.repeat(np.arange(19), rng.integers(1, 12, size=19))
        generators = rng.normal(size=(len(groups), 3, 3)) * 0.003
        shapes = np.einsum("nij,nkj->nik", generators, generators)
        joined = loewner_union_outer_shapes_grouped(shapes, groups, 19)
        self.assertEqual(joined.shape, (19, 3, 3))
        minimum = np.linalg.eigvalsh(joined[groups] - shapes)[:, 0]
        self.assertGreaterEqual(float(np.min(minimum)), -1.0e-12)
        permutation = rng.permutation(len(groups))
        permuted = loewner_union_outer_shapes_grouped(
            shapes[permutation], groups[permutation], 19
        )
        np.testing.assert_allclose(joined, permuted, atol=1.0e-14)

    def test_common_frame_grouped_join_is_certified_and_order_invariant(self):
        rng = np.random.default_rng(20260830)
        groups = np.repeat(np.arange(17), rng.integers(2, 9, size=17))
        generators = rng.normal(size=(len(groups), 3, 3)) * 0.003
        shapes = np.einsum("nij,nkj->nik", generators, generators)
        joined = loewner_union_outer_shapes_grouped_common_frame(
            shapes, groups, 17
        )
        minimum = np.linalg.eigvalsh(joined[groups] - shapes)[:, 0]
        self.assertGreaterEqual(float(np.min(minimum)), -1.0e-12)
        permutation = rng.permutation(len(groups))
        permuted = loewner_union_outer_shapes_grouped_common_frame(
            shapes[permutation], groups[permutation], 17
        )
        np.testing.assert_allclose(
            joined, permuted, atol=1.0e-15, rtol=1.0e-12
        )

    def test_directional_center_vox_proxy_contains_uncertain_samples(self):
        manager = IncrementalMatchedProxyManager(
            filter_size=0.015,
            cluster_size=0.08,
            normal_connection_distance=0.03,
        )
        points = np.array(
            [[0.20 + 0.012 * i, -0.03 + 0.012 * j, 0.5] for i in range(5) for j in range(6)]
        )
        shapes = np.repeat(
            np.diag([0.004**2, 0.006**2, 0.002**2])[None, :, :],
            len(points),
            axis=0,
        )
        radii = np.sqrt(np.linalg.eigvalsh(shapes)[:, -1])
        manager.update(points, radii, shapes)
        proxies = manager.snapshot
        self.assertIsNotNone(proxies.filtered_uncertainty_shapes)
        audit = manager.coverage_audit()
        self.assertTrue(audit.all_raw_sample_balls_certified)
        self.assertGreaterEqual(audit.minimum_centervox_cover_slack, -1.0e-9)
        self.assertGreaterEqual(audit.minimum_proxy_offset_slack, -1.0e-9)
        self.assertGreaterEqual(audit.minimum_sphere_certificate_slack, -1.0e-9)
        rng = np.random.default_rng(11)
        for point_index, (point, proxy_index) in enumerate(
            zip(proxies.filtered_points, proxies.filtered_cluster_indices)
        ):
            uncertainty = proxies.filtered_uncertainty_shapes[point_index]
            root_values, root_vectors = np.linalg.eigh(uncertainty)
            root = root_vectors @ np.diag(np.sqrt(np.maximum(root_values, 0.0)))
            for direction in rng.normal(size=(16, 3)):
                direction /= np.linalg.norm(direction)
                sample = point + root @ direction
                delta = sample - proxies.centers[proxy_index]
                for normal in rng.normal(size=(16, 3)):
                    normal /= np.linalg.norm(normal)
                    support = np.sqrt(
                        normal
                        @ proxies.ellipsoid_shapes[proxy_index]
                        @ normal
                    ) + np.sqrt(
                        normal
                        @ proxies.proxy_uncertainty_shapes[proxy_index]
                        @ normal
                    )
                    self.assertLessEqual(
                        float(normal @ delta), float(support) + 1.0e-9
                    )

    def test_adaptive_sphere_cover_is_safe_irredundant_and_shrunk(self):
        manager = IncrementalMatchedProxyManager(
            filter_size=0.006,
            cluster_size=0.030,
            normal_connection_distance=0.015,
            certificate_radius_limit=0.040,
            sphere_cover_mode="adaptive_irredundant",
            origin=np.zeros(3),
        )
        points = np.array(
            [
                [0.180 + 0.006 * i, -0.030 + 0.006 * j, 0.500]
                for i in range(12)
                for j in range(11)
            ]
        )
        shape = np.diag([0.0025**2, 0.0030**2, 0.0015**2])
        shapes = np.repeat(shape[None, :, :], len(points), axis=0)
        stats = manager.update(points, np.full(len(points), 0.003), shapes)
        proxies = manager.snapshot
        audit = manager.coverage_audit()

        self.assertEqual(proxies.sphere_cover_mode, "adaptive_irredundant")
        self.assertTrue(audit.all_raw_sample_balls_certified)
        self.assertTrue(audit.sphere_cover_inclusion_minimal)
        self.assertLessEqual(
            float(np.max(proxies.sphere_radii)), 0.040 + 1.0e-12
        )
        self.assertLess(
            float(np.min(proxies.sphere_radii)), 0.040 - 1.0e-6
        )
        self.assertLessEqual(
            stats.sphere_cover_selected_count,
            stats.sphere_cover_candidate_count,
        )
        self.assertLess(
            stats.sphere_cover_selected_count, stats.center_voxels
        )
        self.assertEqual(
            stats.sphere_cover_selected_count, len(proxies.centers)
        )
        self.assertGreaterEqual(
            audit.minimum_sphere_certificate_slack, -1.0e-9
        )

    def test_directional_center_relocation_is_certified_in_support_u(self):
        manager = IncrementalMatchedProxyManager(
            filter_size=0.02,
            cluster_size=0.08,
            normal_connection_distance=0.04,
            origin=np.zeros(3),
        )
        points = np.array(
            [[0.101, 0.101, 0.401], [0.118, 0.118, 0.401]]
        )
        shape = np.diag([0.003**2, 0.004**2, 0.002**2])
        shapes = np.repeat(shape[None, :, :], len(points), axis=0)
        radii = np.full(len(points), 0.004)
        manager.update(points[:1], radii[:1], shapes[:1])
        manager.update(points[1:], radii[1:], shapes[1:])
        proxies = manager.snapshot
        self.assertEqual(len(proxies.filtered_points), 1)
        # Relocating centers is certified directionally in U so tangential
        # displacement is not charged as an isotropic delta in every normal.
        self.assertGreater(
            float(np.trace(proxies.filtered_uncertainty_shapes[0])),
            float(np.trace(shape)),
        )
        self.assertAlmostEqual(float(proxies.proxy_offset_radii[0]), 0.0)
        audit = manager.coverage_audit()
        self.assertTrue(audit.all_raw_sample_balls_certified)
        self.assertGreaterEqual(audit.minimum_centervox_cover_slack, -1.0e-12)
        self.assertGreaterEqual(
            audit.minimum_measurement_uncertainty_slack, -1.0e-12
        )
        rng = np.random.default_rng(113)
        center = proxies.centers[0]
        for point in points:
            for normal in rng.normal(size=(128, 3)):
                normal /= np.linalg.norm(normal)
                lhs = float(normal @ (point - center)) + np.sqrt(
                    float(normal @ shape @ normal)
                )
                rhs = (
                    np.sqrt(
                        float(normal @ proxies.base_ellipsoid_shapes[0] @ normal)
                    )
                    + np.sqrt(
                        float(
                            normal
                            @ proxies.proxy_uncertainty_shapes[0]
                            @ normal
                        )
                    )
                    + float(proxies.proxy_offset_radii[0])
                )
                self.assertLessEqual(lhs, rhs + 1.0e-12)
    def test_repeated_frames_do_not_accumulate_center_relocation_drift(self):
        manager = IncrementalMatchedProxyManager(
            filter_size=0.02,
            cluster_size=0.08,
            normal_connection_distance=0.05,
        )
        points = np.array(
            [
                [0.011, 0.011, 0.411],
                [0.036, 0.011, 0.411],
                [0.011, 0.036, 0.411],
                [0.036, 0.036, 0.411],
            ]
        )
        radii = np.full(len(points), 0.003)
        manager.update(points, radii)
        first = manager.snapshot.proxy_offset_radii.copy()
        for _ in range(100):
            manager.update(points[::-1], radii)
        np.testing.assert_allclose(
            manager.snapshot.proxy_offset_radii, first, atol=1.0e-12
        )

    def test_contained_directional_reobservation_reuses_snapshot(self):
        manager = IncrementalMatchedProxyManager(
            filter_size=0.02,
            cluster_size=0.08,
            normal_connection_distance=0.05,
            ellipsoid_cover_mode="adaptive_irredundant",
            certificate_radius_limit=0.07,
            radius_limit_representation="ellipsoid",
        )
        points = np.array(
            [
                [0.011, 0.011, 0.411],
                [0.036, 0.011, 0.411],
                [0.011, 0.036, 0.411],
                [0.036, 0.036, 0.411],
            ]
        )
        shape = np.diag([0.003**2, 0.004**2, 0.002**2])
        shapes = np.repeat(shape[None], len(points), axis=0)
        radii = np.full(len(points), 0.004)
        manager.update(points, radii, shapes)
        first_hash = manager.snapshot_hash
        second = manager.update(points[::-1], radii, shapes[::-1])
        self.assertEqual(second.changed_center_voxels, 0)
        self.assertEqual(second.contained_cell_reuses, len(manager._cells))
        self.assertEqual(manager.snapshot_hash, first_hash)
        self.assertTrue(manager.coverage_audit().all_raw_sample_balls_certified)

    def test_alternating_directional_frames_do_not_inflate_delta_past_voxel(self):
        manager = IncrementalMatchedProxyManager(
            filter_size=0.02,
            cluster_size=0.08,
            normal_connection_distance=0.04,
            origin=np.zeros(3),
        )
        shape = np.diag([0.003**2, 0.004**2, 0.002**2])
        frames = (
            np.array([[0.101, 0.101, 0.401], [0.118, 0.118, 0.401]]),
            np.array([[0.102, 0.118, 0.401], [0.117, 0.102, 0.401]]),
        )
        for frame_index in range(80):
            points = frames[frame_index % 2]
            shapes = np.repeat(shape[None, :, :], len(points), axis=0)
            manager.update(points, np.full(len(points), 0.004), shapes)
        proxies = manager.snapshot
        self.assertLessEqual(
            float(np.max(proxies.proxy_offset_radii)),
            np.sqrt(3.0) * 0.02 + 1.0e-12,
        )
        self.assertTrue(manager.coverage_audit().all_raw_sample_balls_certified)

    def test_causal_updates_publish_one_matched_proxy_set(self):
        manager = IncrementalMatchedProxyManager(
            filter_size=0.01,
            cluster_size=0.08,
            maximum_aabb_overshoot=0.03,
            normal_connection_distance=0.03,
        )
        x, y = np.meshgrid(np.linspace(0.2, 0.28, 5), np.linspace(-0.04, 0.04, 5))
        first_points = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.5)])
        second_points = first_points + np.array([0.0, 0.0, 0.04])
        first = manager.update(first_points, np.full(len(first_points), 0.002))
        first_hash = manager.snapshot_hash
        second = manager.update(second_points, np.full(len(second_points), 0.002))
        proxies = manager.snapshot
        self.assertEqual(first.generation, 0)
        self.assertEqual(second.generation, 1)
        self.assertNotEqual(first_hash, manager.snapshot_hash)
        self.assertEqual(len(proxies.centers), len(proxies.sphere_radii))
        self.assertEqual(len(proxies.centers), len(proxies.ellipsoid_shapes))
        self.assertEqual(len(proxies.centers), len(proxies.proxy_offset_radii))
        self.assertIsNone(proxies.certificate_radius_limit)

    def test_fused_representative_crossing_proxy_bucket_refits_both_sides(self):
        manager = IncrementalMatchedProxyManager(
            filter_size=0.02,
            cluster_size=0.05,
            normal_connection_distance=0.04,
            uncertainty_fusion_mode="fused_certified_ellipsoid",
            origin=np.zeros(3),
        )
        shape = np.diag([0.002**2, 0.002**2, 0.0003**2])
        first = np.array([[0.049, 0.011, 0.401], [0.071, 0.011, 0.401]])
        manager.update(
            first, np.full(len(first), 0.002), np.repeat(shape[None], len(first), 0)
        )
        second = np.array([[0.059, 0.011, 0.401], [0.079, 0.011, 0.401]])
        manager.update(
            second,
            np.full(len(second), 0.002),
            np.repeat(shape[None], len(second), 0),
        )
        self.assertEqual(len(manager.snapshot.filtered_points), len(manager._cells))
        self.assertTrue(manager.coverage_audit().all_raw_sample_balls_certified)

    def test_uniform_final_certificate_radius_and_ellipsoid_longest_limit(self):
        radius_limit = 0.030
        manager = IncrementalMatchedProxyManager(
            filter_size=0.0075,
            cluster_size=0.10,
            maximum_aabb_overshoot=0.025,
            maximum_uncertainty_union_inflation=1.25,
            normal_connection_distance=0.020,
            certificate_radius_limit=radius_limit,
            origin=np.zeros(3),
        )
        x, y = np.meshgrid(
            np.linspace(0.201, 0.289, 12),
            np.linspace(0.101, 0.169, 9),
        )
        points = np.column_stack(
            [x.ravel(), y.ravel(), np.full(x.size, 0.501)]
        )
        uncertainty = np.repeat(
            np.diag([0.004**2, 0.006**2, 0.002**2])[None, :, :],
            len(points),
            axis=0,
        )
        manager.update(
            points,
            np.full(len(points), 0.006),
            uncertainty,
        )
        proxies = manager.snapshot
        effective_sphere = (
            proxies.sphere_radii + proxies.proxy_offset_radii
        )
        ellipsoid_longest = (
            np.sqrt(
                np.maximum(
                    np.linalg.eigvalsh(proxies.ellipsoid_outer_shapes)[:, -1],
                    0.0,
                )
            )
            + proxies.proxy_offset_radii
        )
        self.assertEqual(proxies.certificate_radius_limit, radius_limit)
        np.testing.assert_allclose(
            effective_sphere,
            np.full(len(effective_sphere), radius_limit),
            atol=1.0e-12,
        )
        self.assertLessEqual(float(np.max(ellipsoid_longest)), radius_limit + 1e-9)
        self.assertGreater(len(proxies.centers), 1)
        self.assertTrue(manager.coverage_audit().all_raw_sample_balls_certified)
        # Equal-radius proxies cannot strictly contain one another unless their
        # centers coincide; duplicate centers are forbidden by this fit.
        center_distances = np.linalg.norm(
            proxies.centers[:, None, :] - proxies.centers[None, :, :], axis=2
        )
        np.fill_diagonal(center_distances, np.inf)
        self.assertGreater(float(np.min(center_distances)), 1.0e-12)
        strict_containment = (
            center_distances
            + effective_sphere[None, :]
            < effective_sphere[:, None] - 1.0e-12
        )
        self.assertFalse(bool(np.any(strict_containment)))

    def test_uniform_radius_rejects_indivisible_uncertainty(self):
        manager = IncrementalMatchedProxyManager(
            filter_size=0.0075,
            cluster_size=0.10,
            certificate_radius_limit=0.030,
            origin=np.zeros(3),
        )
        point = np.array([[0.201, 0.101, 0.501]])
        uncertainty = np.diag([0.040**2, 0.002**2, 0.002**2])[None, :, :]
        with self.assertRaisesRegex(ValueError, "indivisible proxy requirement"):
            manager.update(point, np.array([0.040]), uncertainty)

    def test_uncertain_points_are_conservatively_covered(self):
        manager = IncrementalMatchedProxyManager(
            filter_size=0.01,
            cluster_size=0.10,
            maximum_aabb_overshoot=0.04,
            normal_connection_distance=0.04,
        )
        points = np.array(
            [[0.20 + 0.01 * i, -0.03 + 0.01 * j, 0.5] for i in range(6) for j in range(7)]
        )
        manager.update(points, np.full(len(points), 0.003))
        proxies = manager.snapshot
        for point, proxy_index in zip(
            proxies.filtered_points, proxies.filtered_cluster_indices
        ):
            center = proxies.centers[proxy_index]
            sphere_cover = (
                proxies.sphere_radii[proxy_index]
                + proxies.proxy_offset_radii[proxy_index]
            )
            self.assertLessEqual(float(np.linalg.norm(point - center)) + 0.003, sphere_cover + 1e-9)
            delta = point - center
            self.assertLessEqual(
                float(delta @ np.linalg.solve(proxies.ellipsoid_shapes[proxy_index], delta)),
                1.0 + 1e-8,
            )

    def test_scalar_centervox_chain_certifies_every_raw_sample_ball(self):
        rng = np.random.default_rng(37)
        points = rng.uniform(
            [0.1, -0.1, 0.3], [0.5, 0.1, 0.5], size=(300, 3)
        )
        points[:, 2] = 0.4
        radii = rng.uniform(0.001, 0.004, size=len(points))
        manager = IncrementalMatchedProxyManager(
            filter_size=0.012,
            cluster_size=0.08,
            maximum_aabb_overshoot=0.03,
        )
        manager.update(points, radii)
        audit = manager.coverage_audit()
        self.assertTrue(audit.all_raw_sample_balls_certified)
        self.assertGreater(audit.raw_sample_bins, 0)
        self.assertGreaterEqual(audit.minimum_centervox_cover_slack, -1.0e-9)
        self.assertGreaterEqual(audit.minimum_proxy_offset_slack, -1.0e-9)
        self.assertLessEqual(
            audit.maximum_base_ellipsoid_value, 1.0 + 1.0e-9
        )
        self.assertGreaterEqual(
            audit.minimum_sphere_certificate_slack, -1.0e-9
        )

    def test_proxy_ids_persist_for_overlapping_causal_generations(self):
        rng = np.random.default_rng(41)
        points = rng.uniform(
            [0.1, -0.08, 0.35], [0.45, 0.08, 0.45], size=(240, 3)
        )
        points[:, 2] = 0.4
        manager = IncrementalMatchedProxyManager(
            filter_size=0.01,
            cluster_size=0.07,
            maximum_aabb_overshoot=0.03,
        )
        manager.update(points[:180], np.full(180, 0.002))
        old_ids = set(map(int, manager.snapshot.proxy_ids))
        manager.update(points[180:], np.full(60, 0.002))
        new_ids = set(map(int, manager.snapshot.proxy_ids))
        self.assertTrue(old_ids.intersection(new_ids))
        self.assertEqual(len(new_ids), len(manager.snapshot.proxy_ids))


if __name__ == "__main__":
    unittest.main()
