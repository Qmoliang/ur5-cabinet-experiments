from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial import cKDTree

from model import BoxObstacle
from observability_evaluator import (
    ObservedPointSnapshot,
    _filter_risk_patch_indices,
    _first_seen_cycles,
    _nearest_surface_patch_indices,
)
from pointcloud_proxy import sample_box_surface_point_cloud


class ObservabilityEvaluatorTests(unittest.TestCase):
    def test_first_seen_respects_publication_time_and_cover_radius(self) -> None:
        truth = np.array([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]])
        snapshots = [
            ObservedPointSnapshot(
                publish_cycle=7,
                source_cycle=4,
                points=np.array([[0.101, 0.0, 0.0]]),
                cover_radii=np.array([0.002]),
            ),
            ObservedPointSnapshot(
                publish_cycle=2,
                source_cycle=1,
                points=np.array([[0.001, 0.0, 0.0]]),
                cover_radii=np.array([0.002]),
            ),
        ]
        first = _first_seen_cycles(truth, snapshots)
        np.testing.assert_array_equal(first, np.array([2, 7], dtype=np.int32))

    def test_uncovered_truth_point_stays_unseen(self) -> None:
        truth = np.array([[0.0, 0.0, 0.0]])
        snapshot = ObservedPointSnapshot(
            publish_cycle=1,
            source_cycle=0,
            points=np.array([[0.02, 0.0, 0.0]]),
            cover_radii=np.array([0.005]),
        )
        first = _first_seen_cycles(truth, [snapshot])
        self.assertEqual(first[0], np.iinfo(np.int32).max)

    def test_thin_box_opposite_face_is_not_a_risk_witness(self) -> None:
        box = BoxObstacle(
            "thin_plate",
            center=(0.0, 0.0, 0.0),
            half_size=(0.10, 0.10, 0.015),
        )
        points = sample_box_surface_point_cloud((box,), spacing=0.01)
        indices = _nearest_surface_patch_indices(
            np.array([0.0, 0.0, -0.10]),
            box,
            points,
            cKDTree(points),
            correction=0.0075,
        )
        selected = points[indices]
        self.assertTrue(np.allclose(selected[:, 2], -0.015))
        self.assertFalse(np.any(np.isclose(selected[:, 2], 0.015)))

    def test_sampling_correction_is_applied_only_once(self) -> None:
        center = np.zeros(3)
        points = np.array(
            [
                [0.157, 0.0, 0.0],  # clearance 0.107: within 0.1 + 0.008
                [0.163, 0.0, 0.0],  # clearance 0.113: only double-counting admits it
            ]
        )
        selected = _filter_risk_patch_indices(
            center,
            radius=0.05,
            box_points=points,
            indices=np.array([0, 1]),
            observation_distance=0.10,
            sampling_correction=0.008,
        )
        np.testing.assert_array_equal(selected, np.array([0]))


if __name__ == "__main__":
    unittest.main()
