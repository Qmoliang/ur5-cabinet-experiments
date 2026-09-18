import unittest
from types import SimpleNamespace

import numpy as np

from run_protocol_v3_async_online import (
    ContinuousProxyGuard,
    PerceptionPacket,
)


def _packet(*, sphere_radius: float, axes: tuple[float, float, float]):
    shape = np.diag(np.square(np.asarray(axes, dtype=float)))
    proxies = SimpleNamespace(
        proxy_ids=np.array([0], dtype=np.int64),
        centers=np.zeros((1, 3), dtype=float),
        sphere_radii=np.array([sphere_radius], dtype=float),
        base_sphere_radii=np.array([sphere_radius], dtype=float),
        base_ellipsoid_shapes=shape[None, :, :],
        ellipsoid_outer_shapes=shape[None, :, :],
        proxy_uncertainty_shapes=None,
        proxy_offset_radii=np.zeros(1, dtype=float),
    )
    return PerceptionPacket(
        generation=0,
        requested_frame=0,
        source_cycle=0,
        source_time_s=0.0,
        source_q=np.zeros(6),
        proxies=proxies,
        occupancy=None,
        mvt=None,
        frame_row={},
        completed_wall_time=0.0,
    )


class ContinuousProxyGuardTests(unittest.TestCase):
    def test_sphere_segment_collision_is_analytic(self):
        guard = ContinuousProxyGuard(
            _packet(sphere_radius=0.10, axes=(0.10, 0.10, 0.10)), "sphere"
        )
        crossing = guard.certify(
            np.array([[-0.20, 0.0, 0.0]]),
            np.array([[0.20, 0.0, 0.0]]),
            np.array([0.02]),
        )
        self.assertFalse(crossing.safe)
        self.assertEqual(crossing.colliding_pairs, 1)
        self.assertEqual(crossing.worsening_pairs, 1)
        clear = guard.certify(
            np.array([[-0.20, 0.13, 0.0]]),
            np.array([[0.20, 0.13, 0.0]]),
            np.array([0.02]),
        )
        self.assertTrue(clear.safe)

    def test_existing_overlap_allows_only_monotone_recovery(self):
        guard = ContinuousProxyGuard(
            _packet(sphere_radius=0.10, axes=(0.10, 0.10, 0.10)), "sphere"
        )
        recovery = guard.certify(
            np.array([[0.115, 0.0, 0.0]]),
            np.array([[0.125, 0.0, 0.0]]),
            np.array([0.02]),
        )
        self.assertTrue(recovery.safe)
        self.assertEqual(recovery.recovery_pairs, 1)
        worsening = guard.certify(
            np.array([[0.115, 0.0, 0.0]]),
            np.array([[0.110, 0.0, 0.0]]),
            np.array([0.02]),
        )
        self.assertFalse(worsening.safe)
        self.assertEqual(worsening.worsening_pairs, 1)

    def test_ellipsoid_short_sweep_lipschitz_bound(self):
        guard = ContinuousProxyGuard(
            _packet(sphere_radius=0.20, axes=(0.20, 0.05, 0.05)), "ellipsoid"
        )
        clear = guard.certify(
            np.array([[-0.001, 0.071, 0.0]]),
            np.array([[0.001, 0.071, 0.0]]),
            np.array([0.01]),
        )
        self.assertTrue(clear.safe)
        self.assertGreater(clear.minimum_clearance_m, 0.0)
        crossing = guard.certify(
            np.array([[-0.30, 0.0, 0.0]]),
            np.array([[0.30, 0.0, 0.0]]),
            np.array([0.01]),
        )
        self.assertFalse(crossing.safe)

    def test_zero_published_u_uses_single_ellipsoid_closest_point_path(self):
        packet = _packet(sphere_radius=0.20, axes=(0.20, 0.05, 0.05))
        packet.proxies.proxy_uncertainty_shapes = np.zeros((1, 3, 3))
        guard = ContinuousProxyGuard(packet, "ellipsoid")
        self.assertIsNone(guard.directional_uncertainty_shapes)
        self.assertIsNone(guard.uncertainty_eigenvalues)
        result = guard.certify(
            np.array([[-0.001, 0.071, 0.0]]),
            np.array([[0.001, 0.071, 0.0]]),
            np.array([0.01]),
        )
        self.assertTrue(result.safe)

    def test_long_tangential_sweep_may_conservatively_reject(self):
        guard = ContinuousProxyGuard(
            _packet(sphere_radius=0.20, axes=(0.20, 0.05, 0.05)), "ellipsoid"
        )
        result = guard.certify(
            np.array([[0.15, 0.055, 0.0]]),
            np.array([[0.19, 0.055, 0.0]]),
            np.array([0.01]),
        )
        # The Lipschitz lower bound is sufficient, not necessary.  The formal
        # controller halves an inconclusive 20 ms step without changing its
        # direction; it never treats this conservative result as a collision
        # observed from MuJoCo truth.
        self.assertFalse(result.safe)

    def test_directional_pair_batch_certifies_and_rejects(self):
        packet = _packet(sphere_radius=0.20, axes=(0.20, 0.05, 0.05))
        packet.proxies.proxy_uncertainty_shapes = np.diag(
            np.square([0.002, 0.003, 0.001])
        )[None, :, :]
        guard = ContinuousProxyGuard(packet, "ellipsoid")
        # Both swept AABBs overlap the proxy, so the native entry point must
        # solve more than one pair in a single call.  One row is slightly
        # offset to avoid testing two bit-identical inputs.
        starts = np.array([[-0.30, 0.01, 0.0], [-0.30, 0.0, 0.0]])
        ends = np.array([[0.30, 0.01, 0.0], [0.30, 0.0, 0.0]])
        radii = np.array([0.01, 0.01])
        result = guard.certify(starts, ends, radii)
        self.assertFalse(result.safe)
        self.assertEqual(result.candidate_pairs, 2)
        self.assertGreaterEqual(result.worsening_pairs, 1)


if __name__ == "__main__":
    unittest.main()
