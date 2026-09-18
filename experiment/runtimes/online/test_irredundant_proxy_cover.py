import unittest

import numpy as np

from irredundant_proxy_cover import (
    ellipsoid_candidate_coverage,
    inclusion_minimal_set_cover,
    reduce_ellipsoid_candidates,
)
from pointcloud_proxy import PointCloudProxySet


class IrredundantProxyCoverTests(unittest.TestCase):
    def test_set_cover_removes_deletable_candidates_and_keeps_witnesses(self):
        result = inclusion_minimal_set_cover(
            [np.array([0, 1, 2]), np.array([0]), np.array([1]), np.array([2])],
            3,
        )
        self.assertEqual(result.selected_candidates.tolist(), [0])
        self.assertEqual(result.unique_witness_counts.tolist(), [3])

    def test_ellipsoid_reduction_uses_sufficient_minkowski_inclusion(self):
        points = np.array([[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]])
        cell_u = np.repeat(np.diag([0.01, 0.01, 0.01])[None], 2, axis=0)
        core = np.array(
            [np.diag([1.0, 1.0, 1.0]), np.diag([0.04, 0.04, 0.04]), np.diag([0.04, 0.04, 0.04])]
        )
        centers = np.array([[0.0, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]])
        proxy_u = np.repeat(np.diag([0.02, 0.02, 0.02])[None], 3, axis=0)
        candidates = PointCloudProxySet(
            raw_points=points,
            filtered_points=points,
            centers=centers,
            sphere_radii=np.ones(3),
            ellipsoid_shapes=core,
            rotations=np.repeat(np.eye(3)[None], 3, axis=0),
            cluster_keys=np.arange(3)[:, None],
            filtered_cluster_indices=np.array([1, 2]),
            maximum_ellipsoid_overshoot=0.0,
            proxy_offset_radii=np.full(3, 0.02),
            filtered_uncertainty_shapes=cell_u,
            filtered_point_offsets=np.full(2, 0.01),
            proxy_uncertainty_shapes=proxy_u,
            base_sphere_radii=np.ones(3),
            base_ellipsoid_shapes=core,
            ellipsoid_outer_shapes=core,
        )
        coverage = ellipsoid_candidate_coverage(
            candidates,
            cell_points=points,
            cell_uncertainty_shapes=cell_u,
            cell_offsets=np.full(2, 0.01),
        )
        self.assertEqual(coverage[0].tolist(), [0, 1])
        reduced, audit = reduce_ellipsoid_candidates(candidates)
        self.assertEqual(len(reduced.centers), 1)
        self.assertEqual(audit.unique_witness_counts.tolist(), [2])


if __name__ == "__main__":
    unittest.main()
