"""Perturbation audit for the sphere-closed 150 mm shelf-drawer scene."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np

import run_geometry_ablation as ablation
from model import build_model, build_robot_certificate, set_configuration
from pointcloud_proxy import build_shelf_drawer_proxy_set
from run_simulation import conservative_environment_clearance
from shelf_drawer_scene import shelf_drawer_narrow_scene


ROOT = (
    Path(__file__).resolve().parent
    / "results_geometry_ablation"
    / "shelf_drawer_narrow_sphere_perturbation_robustness"
)
TRIALS = 12
Q_PERTURBATION_RAD = 0.005
OBJECTIVE_DITHER_STD = 0.10


def main() -> None:
    base = shelf_drawer_narrow_scene()
    rng = np.random.default_rng(20260828)
    accepted = []
    attempts = 0
    while len(accepted) < TRIALS and attempts < 1000:
        attempts += 1
        q_delta = rng.uniform(-Q_PERTURBATION_RAD, Q_PERTURBATION_RAD, 6)
        q0 = np.asarray(base.q0) + q_delta
        scene = replace(base, q0=tuple(float(value) for value in q0), duration=20.0)
        model = build_model(scene)
        data = mujoco.MjData(model)
        set_configuration(model, data, q0)
        robot = build_robot_certificate(model)
        proxies = build_shelf_drawer_proxy_set(
            scene.boxes,
            cluster_size=0.100,
            maximum_aabb_overshoot=0.025,
        )
        start_clearance = conservative_environment_clearance(
            model,
            data,
            robot,
            proxies.centers,
            proxies.sphere_radii,
            0.006,
        )
        if data.ncon or start_clearance < 0.0:
            continue

        trial = len(accepted)
        ablation.RESULTS_ROOT = ROOT / f"trial_{trial:02d}"
        summary = ablation.run_scene(
            scene,
            "sphere",
            "direct",
            "sphere-limit",
            objective_dither_std=OBJECTIVE_DITHER_STD,
            objective_dither_seed=trial,
            pointcloud_cluster_size=0.100,
            pointcloud_maximum_overshoot=0.025,
        )
        accepted.append(
            {
                "trial": trial,
                "q_delta_rad": q_delta.tolist(),
                "start_certificate_clearance_m": float(start_clearance),
                "success": bool(summary["success"]),
                "stalled": bool(summary["stalled"]),
                "final_error_m": float(summary["final_error_m"]),
                "recent_goal_progress_m": float(summary["recent_goal_progress_m"]),
                "qdot_sign_flip_rate": float(summary["qdot_sign_flip_rate"]),
                "solved_qp_fraction": float(summary["solved_qp_fraction"]),
            }
        )

    if len(accepted) != TRIALS:
        raise RuntimeError(f"Only found {len(accepted)} valid starts after {attempts} attempts")

    feasibility_path = (
        Path(__file__).resolve().parent
        / "results_geometry_ablation"
        / "shelf_drawer_narrow_pca_graph_mahal_cluster100mm_over025mm_feasibility.json"
    )
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    aggregate = {
        "scene": base.name,
        "trials": TRIALS,
        "q_perturbation_bound_rad": Q_PERTURBATION_RAD,
        "target_perturbation_m": 0.0,
        "objective_dither_std": OBJECTIVE_DITHER_STD,
        "duration_per_trial_s": 20.0,
        "successes": int(sum(int(row["success"]) for row in accepted)),
        "minimum_final_error_m": float(min(row["final_error_m"] for row in accepted)),
        "maximum_recent_goal_progress_m": float(
            max(row["recent_goal_progress_m"] for row in accepted)
        ),
        "minimum_solved_qp_fraction": float(
            min(row["solved_qp_fraction"] for row in accepted)
        ),
        "sphere_cross_section_barrier": feasibility["sphere_cross_section_barrier"],
        "runs": accepted,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
