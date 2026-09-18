"""Audit v5 same-CenterVox sphere/ellipsoid pairs at the mandatory section.

This is a development gate, not a path planner.  The sphere result is a
continuous 1-Lipschitz branch-and-bound certificate over the entire physical
drawer-front section.  The ellipsoid result is a strict open-ball witness
computed with the same Q_R + Q_O + U support function and safeguarded native
Newton kernel used by online LiuQP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from certify_formal_paired_cross_section import (
    SAFETY_MARGIN_M,
    _mandatory_attachment_sphere,
    _opening,
    _sphere_section,
)
from native_ellipsoid_support import NativeEllipsoidSupport
from run_protocol_v3_async_online import _protocol_scene


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as arrays:
        return {name: np.asarray(arrays[name]) for name in arrays.files}


def _sphere_snapshot(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "centers": np.asarray(arrays["centers"], dtype=float),
        "sphere_radii": np.asarray(arrays["sphere_radii"], dtype=float),
        "uncertainty_offsets": np.asarray(
            arrays["proxy_offset_radii"], dtype=float
        ),
    }


def _exact_ellipsoid_open_point(
    arrays: dict[str, np.ndarray],
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    robot_radius: float,
) -> dict:
    centers = np.asarray(arrays["centers"], dtype=float)
    shapes = np.asarray(arrays["ellipsoid_shapes"], dtype=float)
    uncertainty = np.asarray(
        arrays["proxy_uncertainty_shapes"], dtype=float
    )
    offsets = np.asarray(arrays["proxy_offset_radii"], dtype=float)
    delta = centers - point[None, :]
    norms = np.linalg.norm(delta, axis=1)
    initial = np.zeros_like(delta)
    valid = norms > 1.0e-12
    initial[valid] = delta[valid] / norms[valid, None]
    initial[~valid, 0] = 1.0

    kernel = NativeEllipsoidSupport()
    robot_shape = np.eye(3, dtype=float) * float(robot_radius) ** 2
    normals, iterations, residuals = kernel.normals_sum_newton_warm(
        point,
        robot_shape,
        centers,
        shapes,
        uncertainty,
        initial,
        max_iterations=16,
    )
    retry = residuals > 1.0e-7
    retry_count = int(np.count_nonzero(retry))
    if retry_count:
        retry_normals, retry_iterations, retry_residuals = (
            kernel.normals_sum_newton_warm(
                point,
                robot_shape,
                centers[retry],
                shapes[retry],
                uncertainty[retry],
                np.zeros((retry_count, 3), dtype=float),
                max_iterations=64,
            )
        )
        normals[retry] = retry_normals
        iterations[retry] += retry_iterations
        residuals[retry] = retry_residuals
    maximum_residual = float(np.max(residuals)) if len(residuals) else 0.0
    if maximum_residual > 1.0e-7:
        raise RuntimeError(
            "ellipsoid support normal failed the online 1e-7 KKT gate: "
            f"{maximum_residual:.9g}"
        )

    obstacle_extent = np.sqrt(
        np.maximum(np.einsum("ni,nij,nj->n", normals, shapes, normals), 0.0)
    )
    uncertainty_extent = np.sqrt(
        np.maximum(
            np.einsum("ni,nij,nj->n", normals, uncertainty, normals), 0.0
        )
    )
    clearances = (
        np.einsum("ni,ni->n", normals, centers - point[None, :])
        - float(robot_radius)
        - obstacle_extent
        - uncertainty_extent
        - offsets
        - SAFETY_MARGIN_M
    )
    limiting = int(np.argmin(clearances))
    clearance = float(clearances[limiting])
    boundary_distance = float(
        np.min(np.r_[point[1:3] - lower, upper - point[1:3]])
    )
    open_radius = min(clearance, boundary_distance)
    normal = normals[limiting]
    q_extent = max(float(obstacle_extent[limiting]), 1.0e-12)
    u_extent = max(float(uncertainty_extent[limiting]), 1.0e-12)
    surface = (
        centers[limiting]
        - shapes[limiting] @ normal / q_extent
        - uncertainty[limiting] @ normal / u_extent
        - offsets[limiting] * normal
    )
    return {
        "test_point_m": point.tolist(),
        "proxy_count": int(len(centers)),
        "exact_support_clearance_m": clearance,
        "opening_boundary_distance_m": boundary_distance,
        "strict_open_ball_radius_m": float(max(0.0, open_radius)),
        "has_strict_open_ball": bool(open_radius > 0.0),
        "limiting_proxy_index": limiting,
        "limiting_surface_point_m": surface.tolist(),
        "limiting_normal": normal.tolist(),
        "safeguarded_newton_iterations_total": int(np.sum(iterations)),
        "retry_count": retry_count,
        "maximum_kkt_residual": maximum_residual,
        "formula": (
            "n^T(o-p)-sqrt(n^T Q_R n)-sqrt(n^T Q_O n)-"
            "sqrt(n^T U n)-delta-d_safe"
        ),
    }


def audit(pair_dir: Path, scene_version: str, grid_size: int) -> dict:
    pair_dir = pair_dir.resolve()
    sphere_path = pair_dir / "sphere_proxies.npz"
    ellipsoid_path = pair_dir / "ellipsoid_proxies.npz"
    metadata_path = pair_dir / "result.json"
    if not sphere_path.is_file() or not ellipsoid_path.is_file():
        raise FileNotFoundError("pair directory must contain both proxy NPZ files")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = {row["representation"]: row for row in metadata["rows"]}
    if rows["sphere"]["rebuilt_centervox_sha256"] != rows["ellipsoid"][
        "rebuilt_centervox_sha256"
    ]:
        raise RuntimeError("sphere and ellipsoid were not rebuilt from the same CenterVox")

    scene = _protocol_scene(scene_version)
    mandatory = _mandatory_attachment_sphere(scene)
    lower, upper = _opening(scene)
    sphere = _load(sphere_path)
    ellipsoid = _load(ellipsoid_path)
    sphere_section = _sphere_section(
        _sphere_snapshot(sphere),
        mandatory["front_x_m"],
        lower,
        upper,
        mandatory["radius_m"],
        grid_size,
    )
    center_point = np.array(
        [
            mandatory["front_x_m"],
            0.5 * (lower[0] + upper[0]),
            0.5 * (lower[1] + upper[1]),
        ],
        dtype=float,
    )
    ellipsoid_open = _exact_ellipsoid_open_point(
        ellipsoid,
        center_point,
        lower,
        upper,
        mandatory["radius_m"],
    )
    result = {
        "certificate": "v5_development_same_centervox_cross_section_v1",
        "formal_evidence": False,
        "pair_directory": str(pair_dir),
        "scene_version": scene_version,
        "same_centervox_sha256": rows["sphere"]["rebuilt_centervox_sha256"],
        "sphere_proxy_sha256": _sha256(sphere_path),
        "ellipsoid_proxy_sha256": _sha256(ellipsoid_path),
        "safety_margin_m": SAFETY_MARGIN_M,
        "mandatory_robot_sphere": mandatory,
        "physical_opening_lower_yz_m": lower.tolist(),
        "physical_opening_upper_yz_m": upper.tolist(),
        "sphere_section": sphere_section,
        "ellipsoid_center_opening": ellipsoid_open,
        "desired_geometric_contrast_passed": bool(
            sphere_section["strictly_closed"]
            and ellipsoid_open["has_strict_open_ball"]
        ),
    }
    output = pair_dir / "continuous_cross_section_v5"
    output.mkdir(exist_ok=True)
    (output / "certificate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair_dir", type=Path)
    parser.add_argument("--scene-version", default="camera_quarter")
    parser.add_argument("--grid-size", type=int, default=101)
    args = parser.parse_args()
    if args.grid_size < 3:
        raise ValueError("grid size must be at least 3")
    print(
        json.dumps(
            audit(args.pair_dir, args.scene_version, args.grid_size),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
