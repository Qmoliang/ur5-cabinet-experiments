"""Perturbation robustness audit for the structurally closed sphere baseline."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np

import run_geometry_ablation as ablation
from model import (
    attachment_position,
    build_model,
    build_robot_certificate,
    set_configuration,
)
from run_depth_camera_ablation import load_sensor_proxy_set
from run_simulation import conservative_environment_clearance
from shelf_drawer_scene import shelf_drawer_certified_scene


ROOT = Path(__file__).resolve().parent / "results_depth_camera" / "strict_long_tool_sphere_perturbations"
TRIALS = 12
Q_PERTURBATION_RAD = 0.005
TARGET_PERTURBATION_M = 0.001
OBJECTIVE_DITHER_STD = 0.10


def main() -> None:
    base = replace(
        shelf_drawer_certified_scene(),
        name="shelf_drawer_strict_long_tool",
        duration=20.0,
    )
    nominal_model = build_model(base)
    nominal_data = mujoco.MjData(nominal_model)
    set_configuration(nominal_model, nominal_data, np.asarray(base.q0))
    start = attachment_position(nominal_model, nominal_data)
    nominal_goal = np.array([0.730, start[1], start[2]])
    proxies = load_sensor_proxy_set(
        Path(__file__).resolve().parent
        / "results_depth_camera"
        / "wrist_depth_proxy_set_cluster140.npz"
    )
    if proxies.proxy_offset_radii is None:
        raise RuntimeError("strict sensor proxies need per-proxy offsets")
    effective_radii = proxies.sphere_radii + proxies.proxy_offset_radii
    rng = np.random.default_rng(20260829)
    rows = []
    attempts = 0
    while len(rows) < TRIALS and attempts < 1000:
        attempts += 1
        q_delta = rng.uniform(-Q_PERTURBATION_RAD, Q_PERTURBATION_RAD, 6)
        target_delta = rng.uniform(
            -TARGET_PERTURBATION_M, TARGET_PERTURBATION_M, 3
        )
        q0 = np.asarray(base.q0) + q_delta
        goal = nominal_goal + target_delta
        scene = replace(
            base,
            q0=tuple(float(value) for value in q0),
            waypoints=(tuple(start), tuple(float(value) for value in goal)),
        )
        model = build_model(scene)
        data = mujoco.MjData(model)
        set_configuration(model, data, q0)
        robot = build_robot_certificate(model)
        start_clearance = conservative_environment_clearance(
            model,
            data,
            robot,
            proxies.centers,
            effective_radii,
            0.006,
        )
        if data.ncon or start_clearance < 0.0:
            continue
        trial = len(rows)
        ablation.RESULTS_ROOT = ROOT / f"trial_{trial:02d}"
        summary = ablation.run_scene(
            scene,
            "sphere",
            "direct",
            "sphere-limit",
            objective_dither_std=OBJECTIVE_DITHER_STD,
            objective_dither_seed=trial,
            pointcloud_cluster_size=0.140,
            pointcloud_maximum_overshoot=0.025,
            pointcloud_filter_size=0.0025,
            apply_surface_cover=True,
            duration_override=20.0,
            pointcloud_proxies_override=proxies,
            pointcloud_output_tag="wrist_depth_frozen_c140_strict_perturbed",
            goal_tolerance=0.005,
        )
        rows.append(
            {
                "trial": trial,
                "q_delta_rad": q_delta.tolist(),
                "target_delta_m": target_delta.tolist(),
                "start_certificate_clearance_m": float(start_clearance),
                "success": bool(summary["success"]),
                "stalled": bool(summary["stalled"]),
                "final_error_m": float(summary["final_error_m"]),
                "recent_goal_progress_m": float(summary["recent_goal_progress_m"]),
                "qdot_sign_flip_rate": float(summary["qdot_sign_flip_rate"]),
                "solved_qp_fraction": float(summary["solved_qp_fraction"]),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False))
    if len(rows) != TRIALS:
        raise RuntimeError(f"only found {len(rows)} valid starts in {attempts} attempts")
    certificate = json.loads(
        (
            Path(__file__).resolve().parent
            / "results_depth_camera"
            / "wrist_depth_strict_long_tool_certificate.json"
        ).read_text(encoding="utf-8")
    )
    aggregate = {
        "scene": base.name,
        "proxy_source": "same frozen wrist-depth cluster140 proxy set as strict ablation",
        "trials": TRIALS,
        "q_perturbation_bound_rad": Q_PERTURBATION_RAD,
        "target_perturbation_bound_m": TARGET_PERTURBATION_M,
        "objective_dither_std": OBJECTIVE_DITHER_STD,
        "duration_per_trial_s": 20.0,
        "successes": int(sum(int(row["success"]) for row in rows)),
        "minimum_final_error_m": float(min(row["final_error_m"] for row in rows)),
        "maximum_solved_qp_fraction": float(max(row["solved_qp_fraction"] for row in rows)),
        "sphere_cross_section_barrier": certificate["sphere_cross_section_barrier"],
        "runs": rows,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
