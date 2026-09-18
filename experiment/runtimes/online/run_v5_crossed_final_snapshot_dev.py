"""Development-only same-CenterVox crossed representation replay.

The script reads one frozen final CenterVox snapshot and independently builds
an irredundant sphere certificate and an irredundant ellipsoid certificate.
It never feeds results back to either frozen trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from incremental_proxy_manager import IncrementalMatchedProxyManager


ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "formal_results" / "final_two_camera"
OUTPUT = FINAL / "formal_v5_0_dev" / "crossed_final_snapshot_smoke"
SOURCE_ROOTS = {
    "sphere": FINAL
    / (
        "formal_strict_goal_v4_3_surface_core_sphere_"
        "camera_quarter_cv7.5mm_cr70mm_u1p05_stride1_vmax180mmps_"
        "affinity_perception_heavy_resolution_bounded_formal3_"
        "adaptive_irredundant_sphere_cover"
    ),
    "ellipsoid": FINAL
    / (
        "formal_strict_goal_v4_3_surface_core_ellipsoid_"
        "camera_quarter_cv7.5mm_cr70mm_u1p05_stride1_vmax180mmps_"
        "affinity_perception_heavy_formal_redesign_no_coreinflate_r70_paired"
    ),
}


def one_run(root: Path) -> Path:
    matches = [path for path in root.iterdir() if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one run below {root}, found {len(matches)}")
    return matches[0]


def hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def save_proxy_set(path: Path, proxies) -> None:
    np.savez_compressed(
        path,
        centers=proxies.centers,
        sphere_radii=proxies.sphere_radii,
        ellipsoid_shapes=proxies.ellipsoid_shapes,
        ellipsoid_outer_shapes=proxies.ellipsoid_outer_shapes,
        proxy_uncertainty_shapes=proxies.proxy_uncertainty_shapes,
        proxy_offset_radii=proxies.proxy_offset_radii,
        filtered_points=proxies.filtered_points,
        filtered_uncertainty_shapes=proxies.filtered_uncertainty_shapes,
        filtered_point_offsets=proxies.filtered_point_offsets,
        filtered_cluster_indices=proxies.filtered_cluster_indices,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("sphere", "ellipsoid"), required=True)
    parser.add_argument(
        "--source-run",
        type=Path,
        help=(
            "Explicit frozen run directory (or its one-run parent). "
            "Required for final v5 paired replays; legacy roots are fallback only."
        ),
    )
    parser.add_argument(
        "--representation", choices=("sphere", "ellipsoid", "both"), default="both"
    )
    parser.add_argument("--radius-limit-mm", type=float, default=70.0)
    parser.add_argument(
        "--sphere-policy",
        choices=("resolution_bounded", "cap"),
        default="resolution_bounded",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.radius_limit_mm <= 0.0:
        parser.error("--radius-limit-mm must be positive")

    if args.source_run is None:
        run_dir = one_run(SOURCE_ROOTS[args.source])
    elif (args.source_run / "final_causal_proxies.npz").is_file():
        run_dir = args.source_run
    else:
        run_dir = one_run(args.source_run)
    archive = np.load(run_dir / "final_causal_proxies.npz")
    points = np.asarray(archive["filtered_points"], dtype=float)
    uncertainty = np.asarray(
        archive["filtered_uncertainty_shapes"], dtype=float
    )
    offsets = np.asarray(
        archive["filtered_point_offsets"], dtype=float
    ).reshape(-1)
    source_hash = hash_arrays(points, uncertainty, offsets)
    representations = (
        ("sphere", "ellipsoid")
        if args.representation == "both"
        else (args.representation,)
    )
    radius_limit = args.radius_limit_mm / 1000.0
    output = args.output / (
        f"source-{args.source}-R{args.radius_limit_mm:g}mm-"
        f"spherePolicy-{args.sphere_policy}"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite crossed replay: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    rebuilt_hashes: list[str] = []
    for representation in representations:
        manager = IncrementalMatchedProxyManager(
            filter_size=0.0075,
            cluster_size=0.10,
            maximum_aabb_overshoot=0.025,
            maximum_uncertainty_union_inflation=1.05,
            certificate_radius_limit=radius_limit,
            uncertainty_fusion_mode="separate_uncertainty",
            sphere_cover_mode=(
                (
                    "adaptive_irredundant"
                    if args.sphere_policy == "resolution_bounded"
                    else "cap_irredundant"
                )
                if representation == "sphere"
                else "matched"
            ),
            ellipsoid_cover_mode=(
                "adaptive_irredundant"
                if representation == "ellipsoid"
                else "matched"
            ),
            radius_limit_representation=(
                "sphere" if representation == "sphere" else "ellipsoid"
            ),
        )
        started = time.perf_counter()
        stats = manager.update(points, offsets, uncertainty)
        update_ms = (time.perf_counter() - started) * 1000.0
        audit = manager.coverage_audit()
        proxies = manager.snapshot
        if representation == "sphere":
            effective_scales = np.asarray(proxies.sphere_radii, dtype=float) + np.asarray(
                proxies.proxy_offset_radii, dtype=float
            )
        else:
            effective_scales = np.sqrt(
                np.maximum(
                    np.linalg.eigvalsh(proxies.ellipsoid_outer_shapes)[:, -1],
                    0.0,
                )
            ) + np.asarray(proxies.proxy_offset_radii, dtype=float)
        rebuilt_hash = hash_arrays(
            proxies.filtered_points,
            proxies.filtered_uncertainty_shapes,
            proxies.filtered_point_offsets,
        )
        rebuilt_hashes.append(rebuilt_hash)
        point_replay_error = float(
            np.max(np.abs(proxies.filtered_points - points))
        )
        uncertainty_replay_error = float(
            np.max(
                np.abs(proxies.filtered_uncertainty_shapes - uncertainty)
            )
        )
        offset_replay_error = float(
            np.max(np.abs(proxies.filtered_point_offsets - offsets))
        )
        if (
            point_replay_error > 1.0e-15
            or uncertainty_replay_error > 1.0e-10
            or offset_replay_error > 1.0e-15
        ):
            raise AssertionError("common CenterVox replay exceeded numeric tolerance")
        if not audit.all_raw_sample_balls_certified:
            raise AssertionError(f"{representation} coverage audit failed")
        row = {
            "development_only": True,
            "source_trajectory": args.source,
            "representation": representation,
            "radius_limit_mm": args.radius_limit_mm,
            "sphere_policy": (
                args.sphere_policy if representation == "sphere" else None
            ),
            "source_run": str(run_dir),
            "source_centervox_sha256": source_hash,
            "rebuilt_centervox_sha256": rebuilt_hash,
            "point_replay_max_error": point_replay_error,
            "uncertainty_replay_max_error": uncertainty_replay_error,
            "offset_replay_max_error": offset_replay_error,
            "center_voxels": len(points),
            "candidate_count": (
                stats.sphere_cover_candidate_count
                if representation == "sphere"
                else stats.certificate_cover_candidate_count
            ),
            "selected_count": len(proxies.centers),
            "effective_scale_min_mm": float(np.min(effective_scales) * 1000.0),
            "effective_scale_median_mm": float(
                np.median(effective_scales) * 1000.0
            ),
            "effective_scale_max_mm": float(np.max(effective_scales) * 1000.0),
            "reverse_deleted": (
                stats.sphere_cover_reverse_deleted
                if representation == "sphere"
                else stats.certificate_cover_reverse_deleted
            ),
            "minimum_unique_witnesses": (
                1
                if representation == "sphere"
                and audit.sphere_cover_inclusion_minimal
                else stats.certificate_cover_minimum_unique_witnesses
            ),
            "coverage_passed": audit.all_raw_sample_balls_certified,
            "inclusion_minimal": (
                audit.sphere_cover_inclusion_minimal
                if representation == "sphere"
                else audit.certificate_cover_inclusion_minimal
            ),
            "update_ms": update_ms,
        }
        rows.append(row)
        save_proxy_set(output / f"{representation}_proxies.npz", proxies)
    if len(set(rebuilt_hashes)) != 1:
        raise AssertionError("crossed representations did not consume identical CenterVox")
    result = {
        "protocol": "v5.0-development-crossed-final-snapshot",
        "formal_evidence": False,
        "source_centervox_sha256": source_hash,
        "rows": rows,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
