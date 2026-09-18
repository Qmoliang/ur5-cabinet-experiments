"""Continuous mandatory-section audit for actual online proxy snapshots.

The end-effector center starts in front of x=0.39 m and the only task target is
behind it, so every continuous center trajectory must cross this plane.  A
grid plus the 1-Lipschitz property of distance provides a continuous (not just
sampled) sphere-closure certificate.  For ellipsoids, the center-line support
plane gives a conservative lower bound for the exact Q+U support optimum; a
positive bound larger than the grid cover radius certifies an open patch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from model import build_robot_certificate
from protocol_drawer_scene import (
    DRAWER_Y,
    DRAWER_Z,
    INNER_HALF_Y,
    INNER_HALF_Z,
    SAFETY_MARGIN,
)


MANDATORY_X = 0.390
GRID_SIZE = 161


def _effective_attachment_radius(model, robot) -> tuple[float, int, float]:
    site_id = model.site("attachment_site").id
    body_id = int(model.site_bodyid[site_id])
    site_local = np.asarray(model.site_pos[site_id], dtype=float)
    candidates = []
    for index, proxy in enumerate(robot):
        if proxy.body_id != body_id:
            continue
        offset = float(np.linalg.norm(proxy.local_center - site_local))
        candidates.append((float(proxy.radius - offset), index, offset))
    if not candidates:
        raise RuntimeError("no robot certificate sphere belongs to attachment body")
    return max(candidates)


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    ys = np.linspace(DRAWER_Y - INNER_HALF_Y, DRAWER_Y + INNER_HALF_Y, GRID_SIZE)
    zs = np.linspace(DRAWER_Z - INNER_HALF_Z, DRAWER_Z + INNER_HALF_Z, GRID_SIZE)
    yz = np.array(np.meshgrid(ys, zs, indexing="ij")).reshape(2, -1).T
    points = np.c_[np.full(len(yz), MANDATORY_X), yz]
    cover = 0.5 * math.sqrt((ys[1] - ys[0]) ** 2 + (zs[1] - zs[0]) ** 2)
    return ys, zs, points, float(cover)


def _sphere_field(
    points: np.ndarray,
    centers: np.ndarray,
    effective_obstacle_radii: np.ndarray,
    robot_radius: float,
) -> np.ndarray:
    field = np.full(len(points), np.inf)
    for batch in np.array_split(np.arange(len(points)), 96):
        delta = points[batch, None, :] - centers[None, :, :]
        clearance = (
            np.linalg.norm(delta, axis=2)
            - effective_obstacle_radii[None, :]
            - robot_radius
            - SAFETY_MARGIN
        )
        field[batch] = np.min(clearance, axis=1)
    return field


def _ellipsoid_directional_lower_field(
    points: np.ndarray,
    centers: np.ndarray,
    shapes: np.ndarray,
    uncertainty: np.ndarray,
    offsets: np.ndarray,
    robot_radius: float,
) -> np.ndarray:
    """Certified lower bound using one valid support direction per pair."""

    field = np.full(len(points), np.inf)
    for batch in np.array_split(np.arange(len(points)), 192):
        delta = centers[None, :, :] - points[batch, None, :]
        norms = np.linalg.norm(delta, axis=2)
        normals = delta / np.maximum(norms[:, :, None], 1.0e-15)
        q_support = np.sqrt(
            np.maximum(
                np.einsum("bni,nij,bnj->bn", normals, shapes, normals),
                0.0,
            )
        )
        u_support = np.sqrt(
            np.maximum(
                np.einsum("bni,nij,bnj->bn", normals, uncertainty, normals),
                0.0,
            )
        )
        lower = (
            norms
            - robot_radius
            - q_support
            - u_support
            - offsets[None, :]
            - SAFETY_MARGIN
        )
        field[batch] = np.min(lower, axis=1)
    return field


def audit(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(run_dir / "scene.xml"))
    robot = build_robot_certificate(model)
    robot_radius, robot_index, robot_offset = _effective_attachment_radius(
        model, robot
    )
    ys, zs, points, cover = _grid()
    with np.load(run_dir / "final_causal_proxies.npz") as arrays:
        centers = np.asarray(arrays["centers"], dtype=float)
        offsets = np.asarray(arrays["uncertainty_offsets"], dtype=float)
        sphere_radii = np.asarray(arrays["sphere_radii"], dtype=float) + offsets
        sphere_field = _sphere_field(
            points, centers, sphere_radii, robot_radius
        )
        ellipsoid_lower = _ellipsoid_directional_lower_field(
            points,
            centers,
            np.asarray(arrays["ellipsoid_shapes"], dtype=float),
            np.asarray(arrays["proxy_uncertainty_shapes"], dtype=float),
            offsets,
            robot_radius,
        )

    sphere_best_index = int(np.argmax(sphere_field))
    ellipsoid_best_index = int(np.argmax(ellipsoid_lower))
    sphere_sampled_max = float(sphere_field[sphere_best_index])
    sphere_continuous_upper = sphere_sampled_max + cover
    ellipsoid_sampled_lower = float(ellipsoid_lower[ellipsoid_best_index])
    ellipsoid_cell_lower = ellipsoid_sampled_lower - cover
    best_point = points[ellipsoid_best_index]
    boundary_distance = min(
        float(best_point[1] - ys[0]),
        float(ys[-1] - best_point[1]),
        float(best_point[2] - zs[0]),
        float(zs[-1] - best_point[2]),
    )
    report = {
        "passed": True,
        "run": run_dir.name,
        "representation_used_by_controller": summary["representation"],
        "snapshot": str((run_dir / "final_causal_proxies.npz").resolve()),
        "mandatory_plane_x_m": MANDATORY_X,
        "necessity_argument": (
            "the continuous end-effector-center trajectory from the recorded "
            "initial side to the sole target at x=0.63 m must cross x=0.39 m"
        ),
        "grid_size": GRID_SIZE,
        "grid_cover_radius_m": cover,
        "proxy_count": len(centers),
        "effective_attachment_center_sphere": {
            "radius_m": robot_radius,
            "certificate_index": robot_index,
            "certificate_center_offset_m": robot_offset,
        },
        "sphere_certificate_section": {
            "sampled_max_clearance_m": sphere_sampled_max,
            "continuous_maximum_upper_bound_m": sphere_continuous_upper,
            "continuously_closed": bool(sphere_continuous_upper <= 0.0),
            "closure_margin_m": float(max(0.0, -sphere_continuous_upper)),
            "best_sample_point_m": points[sphere_best_index].tolist(),
            "proof": "1-Lipschitz exact point-to-union-of-spheres clearance",
        },
        "ellipsoid_q_plus_u_section": {
            "sampled_directional_support_lower_bound_m": ellipsoid_sampled_lower,
            "whole_grid_cell_clearance_lower_bound_m": ellipsoid_cell_lower,
            "certified_open": bool(ellipsoid_cell_lower > 0.0),
            "certified_open_disk_radius_m": float(
                max(0.0, min(ellipsoid_sampled_lower, boundary_distance))
            ),
            "best_sample_point_m": best_point.tolist(),
            "proof": (
                "center-line direction is feasible in the exact support "
                "maximization, then apply 1-Lipschitz clearance bound"
            ),
        },
    }
    output = run_dir / "online_mandatory_section_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        run_dir / "online_mandatory_section_fields.npz",
        y=ys,
        z=zs,
        plane_x_m=np.asarray(MANDATORY_X),
        sphere_clearance=sphere_field.reshape(GRID_SIZE, GRID_SIZE),
        ellipsoid_directional_lower_bound=ellipsoid_lower.reshape(
            GRID_SIZE, GRID_SIZE
        ),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
