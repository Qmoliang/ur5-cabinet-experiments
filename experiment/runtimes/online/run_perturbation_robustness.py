"""Robustness audit for the sphere-closed LiuQP slot experiment."""

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
    certificate_world_state,
    set_configuration,
)
from robust_slot_scene import (
    proxy_opening_certificate,
    robust_slot_scene,
    sample_matched_slot_spheres,
)
from run_simulation import conservative_environment_clearance


ROOT = Path(__file__).resolve().parent / "results_geometry_ablation" / "robustness_sphere_dither"
TRIALS = 12
Q_PERTURBATION_RAD = 0.005
TARGET_PERTURBATION_M = 0.005
OBJECTIVE_DITHER_STD = 0.10


def main() -> None:
    base = robust_slot_scene()
    rng = np.random.default_rng(20260828)
    accepted: list[dict] = []
    attempts = 0
    while len(accepted) < TRIALS and attempts < 1000:
        attempts += 1
        q_delta = rng.uniform(-Q_PERTURBATION_RAD, Q_PERTURBATION_RAD, 6)
        target_delta = rng.uniform(-TARGET_PERTURBATION_M, TARGET_PERTURBATION_M, 3)
        q0 = np.asarray(base.q0) + q_delta
        target = np.asarray(base.waypoints[-1]) + target_delta
        scene = replace(
            base,
            q0=tuple(float(value) for value in q0),
            waypoints=(base.waypoints[0], tuple(float(value) for value in target)),
            duration=8.0,
        )
        model = build_model(scene)
        data = mujoco.MjData(model)
        set_configuration(model, data, q0)
        robot = build_robot_certificate(model)
        centers, radii, _ = sample_matched_slot_spheres(scene.boxes)
        start_clearance = conservative_environment_clearance(
            model, data, robot, centers, radii, 0.006
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
        )
        accepted.append(
            {
                "trial": trial,
                "q_delta_rad": q_delta.tolist(),
                "target_delta_m": target_delta.tolist(),
                "start_certificate_clearance_m": float(start_clearance),
                "start_exact_contacts": int(data.ncon),
                "success": bool(summary["success"]),
                "final_error_m": float(summary["final_error_m"]),
                "solved_qp_fraction": float(summary["solved_qp_fraction"]),
            }
        )

    if len(accepted) != TRIALS:
        raise RuntimeError(f"Only found {len(accepted)} valid starts after {attempts} attempts")

    model = build_model(base)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(base.q0))
    robot = build_robot_certificate(model)
    positions, _, radii = certificate_world_state(model, data, robot)
    ee = attachment_position(model, data)
    critical = int(np.argmin(np.linalg.norm(positions - ee, axis=1)))
    barrier = proxy_opening_certificate(float(radii[critical]))
    barrier.update(
        {
            "critical_robot_proxy_index": critical,
            "critical_robot_body": robot[critical].body_name,
            "critical_proxy_equals_attachment_error_m": float(
                np.linalg.norm(positions[critical] - ee)
            ),
            "wall_spans_workspace_y_and_z": True,
            "robust_transverse_perturbation_radius_m": float(
                -0.5 * barrier["sphere_modeled_opening_m"]
            ),
            "logical_conclusion": (
                "At this fixed 204-obstacle-sphere certificate level, the "
                "critical end-effector sphere has no feasible transverse "
                "crossing point. Objective or initial-state perturbations do "
                "not change that certificate; refining the sphere tree does."
            ),
        }
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "barrier_certificate.json").write_text(
        json.dumps(barrier, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    aggregate = {
        "trials": TRIALS,
        "q_perturbation_bound_rad": Q_PERTURBATION_RAD,
        "target_perturbation_bound_m": TARGET_PERTURBATION_M,
        "objective_dither_std": OBJECTIVE_DITHER_STD,
        "successes": int(sum(int(row["success"]) for row in accepted)),
        "minimum_final_error_m": float(min(row["final_error_m"] for row in accepted)),
        "minimum_solved_qp_fraction": float(
            min(row["solved_qp_fraction"] for row in accepted)
        ),
        "runs": accepted,
    }
    (ROOT / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
