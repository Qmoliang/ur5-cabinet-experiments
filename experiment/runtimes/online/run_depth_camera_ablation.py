"""Sensor-driven sphere/ellipsoid LiuQP ablation with an active UR5 wrist scan."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from depth_camera_perception import (
    UR5MountedDepthCamera,
    acquire_active_wrist_scan,
    fuse_depth_observations,
)
from model import build_model, set_configuration
from pointcloud_proxy import PointCloudProxySet, fit_matched_voxel_proxies
from run_geometry_ablation import run_scene
from shelf_drawer_scene import shelf_drawer_certified_scene


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results_depth_camera"


def load_sensor_proxy_set(path: Path | None = None) -> PointCloudProxySet:
    """Load a frozen wrist-depth proxy set for repeatable controller audits."""

    source = RESULTS_ROOT / "wrist_depth_proxy_set.npz" if path is None else Path(path)
    payload = np.load(source)
    centers = np.asarray(payload["centers"], dtype=float)
    count = len(centers)
    rotations = (
        np.asarray(payload["rotations"], dtype=float)
        if "rotations" in payload.files
        else np.repeat(np.eye(3)[None, :, :], count, axis=0)
    )
    filtered_cluster_indices = (
        np.asarray(payload["filtered_cluster_indices"], dtype=np.int64)
        if "filtered_cluster_indices" in payload.files
        else np.zeros(len(payload["filtered_points"]), dtype=np.int64)
    )
    return PointCloudProxySet(
        raw_points=np.asarray(payload["raw_points"], dtype=float),
        filtered_points=np.asarray(payload["filtered_points"], dtype=float),
        centers=centers,
        sphere_radii=np.asarray(payload["sphere_radii"], dtype=float),
        ellipsoid_shapes=np.asarray(payload["ellipsoid_shapes"], dtype=float),
        rotations=rotations,
        cluster_keys=np.arange(count, dtype=np.int64)[:, None],
        filtered_cluster_indices=filtered_cluster_indices,
        maximum_ellipsoid_overshoot=0.025,
        surface_cover_radius=float(np.max(payload["proxy_offset_radii"])),
        proxy_offset_radii=np.asarray(payload["proxy_offset_radii"], dtype=float),
    )


def drawer_scan_configurations(q0: np.ndarray) -> list[np.ndarray]:
    """Serpentine pan/pitch sweep that remains in front of the drawer."""

    offsets = (
        (-0.24, -0.32),
        (-0.24, 0.00),
        (-0.24, 0.32),
        (0.00, 0.32),
        (0.00, 0.00),
        (0.00, -0.32),
        (0.24, -0.32),
        (0.24, 0.00),
        (0.24, 0.32),
    )
    configurations = []
    for wrist_1, wrist_2 in offsets:
        q = np.asarray(q0, dtype=float).copy()
        q[3] += wrist_1
        q[4] += wrist_2
        configurations.append(q)
    return configurations


def build_sensor_proxy_set(
    *,
    width: int,
    height: int,
    pixel_stride: int,
    filter_size: float,
    cluster_size: float,
    maximum_overshoot: float,
    calibrated_depth_bound: float,
    depth_noise_std: float,
):
    scene = shelf_drawer_certified_scene()
    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(scene.q0))
    # Fixed UR5 reachable-workspace crop.  It is independent of obstacle names
    # and removes the distant floor without using MuJoCo segmentation labels.
    lower = np.array([-0.20, -0.20, 0.05])
    upper = np.array([1.10, 1.10, 1.20])

    started = time.perf_counter()
    with UR5MountedDepthCamera(
        model,
        width=width,
        height=height,
        pixel_stride=pixel_stride,
        maximum_range=1.60,
        depth_noise_std=depth_noise_std,
        seed=0,
    ) as camera:
        scan = acquire_active_wrist_scan(
            model,
            data,
            camera,
            drawer_scan_configurations(np.asarray(scene.q0)),
            dt=0.02,
            maximum_joint_speed=0.45,
            return_to_start=True,
        )
    fused = fuse_depth_observations(
        scan.observations,
        voxel_size=filter_size,
        calibrated_depth_bound=calibrated_depth_bound,
        workspace_lower=lower,
        workspace_upper=upper,
        accepted_geom_ids=None,
    )
    if len(fused.filtered_points) < 100:
        raise RuntimeError("active wrist scan produced too few environment points")
    proxies = fit_matched_voxel_proxies(
        fused.filtered_points,
        filter_size=filter_size,
        cluster_size=cluster_size,
        maximum_aabb_overshoot=maximum_overshoot,
        already_filtered=True,
        filtered_point_offsets=fused.filtered_cover_radii,
        normal_connection_distance=max(0.010, 1.75 * filter_size),
    )
    proxies = replace(
        proxies,
        raw_points=fused.raw_points,
        surface_cover_radius=fused.cover_radius,
    )
    elapsed = time.perf_counter() - started
    acquisition = {
        "camera_name": "ur5_depth_wrist",
        "camera_mount": "wrist_3_link",
        "camera_resolution": [int(width), int(height)],
        "pixel_stride": int(pixel_stride),
        "scan_pose_count": len(scan.observations),
        "scan_duration_s": scan.duration,
        "scan_trajectory_steps": len(scan.q_trajectory),
        "scan_penetrating_steps": scan.penetrating_steps,
        "raw_depth_point_count": len(fused.raw_points),
        "centervox_point_count": len(fused.filtered_points),
        "proxy_count": len(proxies.centers),
        "mean_proxy_offset_m": float(np.mean(proxies.proxy_offset_radii)),
        "maximum_proxy_offset_m": float(np.max(proxies.proxy_offset_radii)),
        "maximum_observed_surface_cover_m": fused.cover_radius,
        "maximum_ellipsoid_overshoot_m": proxies.maximum_ellipsoid_overshoot,
        "pipeline_wall_time_s": float(elapsed),
        "uses_ground_truth_geometry_for_controller": False,
        "segmentation_role": (
            "simulation-only audit; controller cloud uses workspace crop and XYZ only"
        ),
    }
    return scene, proxies, scan, fused, acquisition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--pixel-stride", type=int, default=1)
    parser.add_argument("--filter-size", type=float, default=0.0025)
    parser.add_argument("--cluster-size", type=float, default=0.100)
    parser.add_argument("--maximum-overshoot", type=float, default=0.025)
    parser.add_argument("--depth-bound", type=float, default=0.003)
    parser.add_argument("--depth-noise-std", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument(
        "--certificate",
        choices=("both", "sphere", "ellipsoid"),
        default="both",
    )
    args = parser.parse_args()

    scene, proxies, scan, fused, acquisition = build_sensor_proxy_set(
        width=args.width,
        height=args.height,
        pixel_stride=args.pixel_stride,
        filter_size=args.filter_size,
        cluster_size=args.cluster_size,
        maximum_overshoot=args.maximum_overshoot,
        calibrated_depth_bound=args.depth_bound,
        depth_noise_std=args.depth_noise_std,
    )
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RESULTS_ROOT / "wrist_depth_proxy_set.npz",
        raw_points=fused.raw_points,
        filtered_points=fused.filtered_points,
        filtered_cover_radii=fused.filtered_cover_radii,
        centers=proxies.centers,
        sphere_radii=proxies.sphere_radii,
        ellipsoid_shapes=proxies.ellipsoid_shapes,
        rotations=proxies.rotations,
        filtered_cluster_indices=proxies.filtered_cluster_indices,
        proxy_offset_radii=proxies.proxy_offset_radii,
        scan_q_trajectory=scan.q_trajectory,
        camera_positions=fused.camera_positions,
    )
    (RESULTS_ROOT / "wrist_depth_acquisition.json").write_text(
        json.dumps(acquisition, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(acquisition, indent=2))

    certificates = (
        ("sphere", "ellipsoid")
        if args.certificate == "both"
        else (args.certificate,)
    )
    summaries = {}
    for certificate in certificates:
        summaries[certificate] = run_scene(
            scene,
            certificate,
            guidance="direct",
            robot_geometry="sphere-limit",
            pointcloud_cluster_size=args.cluster_size,
            pointcloud_maximum_overshoot=args.maximum_overshoot,
            pointcloud_filter_size=args.filter_size,
            apply_surface_cover=True,
            duration_override=args.duration,
            pointcloud_proxies_override=proxies,
            pointcloud_output_tag="wrist_depth",
        )
    summary_path = RESULTS_ROOT / "wrist_depth_ablation_summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            existing.update(summaries)
            summaries = existing
    summary_path.write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
