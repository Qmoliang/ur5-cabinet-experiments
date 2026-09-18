"""Regression tests for the data-driven shelf/open-drawer experiment."""

import unittest

import mujoco
import numpy as np

from ellipsoid_model import build_robot_ellipsoid_certificate
from ellipsoid_qp_controller import EllipsoidLiuQPController
from certified_scene_compare_viewer import _outer_offset_shape
from liuqp_controller import LiuQPController
from model import build_model, build_robot_certificate, set_configuration
from multilevel_voxel_table import MultilevelVoxelTable
from pointcloud_proxy import (
    build_shelf_drawer_proxy_set,
    sample_box_surface_point_cloud,
)
from run_simulation import minimum_mujoco_world_contact
from shelf_drawer_scene import shelf_drawer_scene


class ShelfDrawerPointCloudTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = shelf_drawer_scene()
        cls.model = build_model(cls.scene)
        cls.data = mujoco.MjData(cls.model)
        set_configuration(cls.model, cls.data, np.asarray(cls.scene.q0))
        cls.proxies = build_shelf_drawer_proxy_set(cls.scene.boxes)

    def test_exact_start_is_collision_free(self) -> None:
        self.assertGreaterEqual(minimum_mujoco_world_contact(self.data, self.model), -1e-8)

    def test_matched_proxy_count_and_data_driven_anisotropy(self) -> None:
        p = self.proxies
        self.assertEqual(len(p.centers), len(p.sphere_radii))
        self.assertEqual(len(p.centers), len(p.ellipsoid_shapes))
        axes = np.sqrt(np.linalg.eigvalsh(p.ellipsoid_shapes))
        anisotropic = np.ptp(axes, axis=1) > 0.005
        self.assertGreater(np.mean(anisotropic), 0.5)
        self.assertLessEqual(p.maximum_ellipsoid_overshoot, 0.0201)

    def test_independent_dense_surface_has_a_certified_cover_radius(self) -> None:
        p = self.proxies
        validation = sample_box_surface_point_cloud(
            self.scene.boxes, p.validation_spacing
        )
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(p.filtered_points).query(validation, k=1)
        self.assertLessEqual(
            float(np.max(distances)), p.validation_sampled_max_distance + 1.0e-12
        )
        self.assertGreater(p.surface_cover_radius, 0.0)
        self.assertAlmostEqual(
            p.surface_cover_radius,
            p.validation_sampled_max_distance + p.validation_lipschitz_correction,
            places=12,
        )

    def test_viewer_outer_ellipsoid_contains_exact_offset_support(self) -> None:
        rng = np.random.default_rng(917)
        for shape in self.proxies.ellipsoid_shapes[:: max(1, len(self.proxies.centers) // 20)]:
            outer = _outer_offset_shape(shape, self.proxies.surface_cover_radius)
            normals = rng.normal(size=(100, 3))
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            exact_support = np.sqrt(
                np.einsum("ni,ij,nj->n", normals, shape, normals)
            ) + self.proxies.surface_cover_radius
            display_support = np.sqrt(
                np.einsum("ni,ij,nj->n", normals, outer, normals)
            )
            self.assertTrue(np.all(display_support + 1.0e-12 >= exact_support))

    def test_mvt_query_is_conservative_for_overlapping_proxy_aabbs(self) -> None:
        p = self.proxies
        table = MultilevelVoxelTable.from_ellipsoids(
            p.centers,
            p.ellipsoid_shapes,
            voxel_size=0.073,
            query_padding=0.16,
            offset_radii=p.surface_cover_radius,
        )
        obstacle_half = p.surface_cover_radius + np.sqrt(
            np.maximum(np.diagonal(p.ellipsoid_shapes, axis1=1, axis2=2), 0.0)
        )
        rng = np.random.default_rng(812)
        for _ in range(20):
            center = rng.uniform([-0.2, 0.0, 0.15], [0.85, 0.8, 1.0])
            half = rng.uniform(0.02, 0.08, size=3)
            candidates = set(table.query_aabb(center, half))
            padded_half = half + table.query_padding
            overlaps = np.all(
                np.abs(p.centers - center) <= obstacle_half + padded_half,
                axis=1,
            )
            self.assertTrue(set(np.flatnonzero(overlaps)).issubset(candidates))

    def test_first_sphere_and_full_ellipsoid_qps_are_solved(self) -> None:
        p = self.proxies
        sphere_robot = build_robot_certificate(self.model)
        sphere_table = MultilevelVoxelTable.from_spheres(
            p.centers,
            p.sphere_radii + p.surface_cover_radius,
            voxel_size=max(proxy.radius for proxy in sphere_robot) + 0.003,
            query_padding=0.16,
        )
        sphere = LiuQPController(
            self.model,
            self.data,
            self.scene,
            sphere_robot,
            p.centers,
            p.sphere_radii + p.surface_cover_radius,
            obstacle_index=sphere_table,
        )
        _, sphere_metrics = sphere.solve(np.asarray(self.scene.waypoints[-1]))
        self.assertTrue(sphere_metrics.status.startswith("solved"))
        self.assertLess(
            sphere_metrics.broadphase_candidate_pairs,
            sphere_metrics.raw_obstacle_spheres,
        )

        ellipsoid_robot = build_robot_ellipsoid_certificate(
            self.model, axial_overlap_factor=2.0
        )
        ellipsoid_table = MultilevelVoxelTable.from_ellipsoids(
            p.centers,
            p.ellipsoid_shapes,
            voxel_size=max(np.max(proxy.semi_axes) for proxy in ellipsoid_robot) + 0.003,
            query_padding=0.16,
            offset_radii=p.surface_cover_radius,
        )
        ellipsoid = EllipsoidLiuQPController(
            self.model,
            self.data,
            self.scene,
            ellipsoid_robot,
            p.centers,
            p.ellipsoid_shapes,
            obstacle_offsets=p.surface_cover_radius,
            obstacle_index=ellipsoid_table,
        )
        _, ellipsoid_metrics = ellipsoid.solve(np.asarray(self.scene.waypoints[-1]))
        self.assertTrue(ellipsoid_metrics.status.startswith("solved"))
        self.assertLess(
            ellipsoid_metrics.broadphase_candidate_pairs,
            ellipsoid_metrics.raw_obstacle_proxies,
        )


if __name__ == "__main__":
    unittest.main()
