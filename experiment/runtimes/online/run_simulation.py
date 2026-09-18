"""Run the UR5e LiuQP reproduction in shelf and cage scenes.

Examples
--------
Run both paper-inspired/extension scenes headlessly::

    python run_simulation.py --scene all

Open an interactive MuJoCo viewer::

    python run_simulation.py --scene shelf --viewer

Use only the final Cartesian goal (local-planner ablation)::

    python run_simulation.py --scene cage --guidance direct

``prescribed`` guidance uses the sequence of Cartesian goals allowed by the
paper's Eq. (17).  LiuQP remains the online optimizer at every step; the
waypoints only supply P_tilde, as in the paper's trajectory-following tests.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from liuqp_controller import LiuQPController
from model import (
    DT,
    SCENES,
    SceneDefinition,
    attachment_position,
    build_model,
    build_robot_certificate,
    build_xml,
    certificate_world_state,
    sample_box_sphere_tree,
    set_configuration,
)


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"


def conservative_environment_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_spheres,
    obstacle_centers: np.ndarray,
    obstacle_radii: np.ndarray,
    safety_margin: float,
) -> float:
    positions, _, robot_radii = certificate_world_state(model, data, robot_spheres)
    delta = positions[:, None, :] - obstacle_centers[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    clearance = distance - robot_radii[:, None] - obstacle_radii[None, :] - safety_margin
    return float(np.min(clearance))


def minimum_mujoco_world_contact(data: mujoco.MjData, model: mujoco.MjModel) -> float:
    """Minimum exact primitive contact distance for robot-vs-world pairs."""

    distances: list[float] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if (body1 == 0) != (body2 == 0):
            distances.append(float(contact.dist))
    return float(min(distances, default=np.inf))


def minimum_mujoco_self_contact(data: mujoco.MjData, model: mujoco.MjModel) -> float:
    distances: list[float] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if body1 > 0 and body2 > 0 and body1 != body2:
            distances.append(float(contact.dist))
    return float(min(distances, default=np.inf))


def integrate_with_safety_backtracking(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qdot: np.ndarray,
    robot_spheres,
    obstacle_centers: np.ndarray,
    obstacle_radii: np.ndarray,
    safety_margin: float,
) -> tuple[float, float]:
    """Euler-integrate Algorithm 1 and guard the local linearization.

    The paper specifies one-step Euler integration after every sequential QP.
    Because Eq. (23) linearizes nonlinear forward kinematics, this reproduction
    adds a standard backtracking guard: reduce only the step length until the
    conservative sphere certificate is non-penetrating.  It adds no new motion
    objective and is reported separately from the paper equations.
    """

    q_before = data.qpos.copy()
    current_clearance = conservative_environment_clearance(
        model, data, robot_spheres, obstacle_centers, obstacle_radii, safety_margin
    )
    accepted_scale = 0.0
    accepted_clearance = current_clearance
    for scale in (
        1.0,
        0.5,
        0.25,
        0.125,
        0.0625,
        0.03125,
        0.015625,
        0.0078125,
        0.00390625,
        0.001953125,
        0.0009765625,
        0.0,
    ):
        q_trial = q_before.copy()
        if scale:
            mujoco.mj_integratePos(model, q_trial, qdot, DT * scale)
        set_configuration(model, data, q_trial)
        clearance = conservative_environment_clearance(
            model, data, robot_spheres, obstacle_centers, obstacle_radii, safety_margin
        )
        # Never introduce penetration.  If an initial reconstructed scene has
        # a tiny conservative overlap, accept only non-worsening motion.
        threshold = min(-2.0e-5, current_clearance - 2.0e-5)
        if clearance >= threshold:
            accepted_scale = scale
            accepted_clearance = clearance
            break
    return float(accepted_scale), float(accepted_clearance)


def render_frame(renderer: mujoco.Renderer, data: mujoco.MjData, output: Path) -> None:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.30, 0.0, 0.58])
    camera.distance = 2.15
    camera.azimuth = -35.0
    camera.elevation = -20.0
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
    try:
        import imageio.v3 as iio

        iio.imwrite(output, pixels)
    except ImportError:
        from PIL import Image

        Image.fromarray(pixels).save(output)


def make_plot(rows: list[dict[str, float | int | str]], output: Path, scene_name: str) -> None:
    t = np.asarray([float(row["time_s"]) for row in rows])
    error = np.asarray([float(row["final_target_error_m"]) for row in rows])
    clearance = np.asarray([float(row["certificate_clearance_m"]) for row in rows])
    active = np.asarray([float(row["active_obstacle_rows"]) for row in rows])
    raw = np.asarray([float(row["raw_obstacle_spheres"]) for row in rows])

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), sharex=True)
    axes[0].plot(t, 1000.0 * error, color="#1f77b4", linewidth=1.7)
    axes[0].set_ylabel("goal error [mm]")
    axes[0].grid(alpha=0.25)
    axes[1].plot(t, 1000.0 * clearance, color="#d62728", linewidth=1.5)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("sphere clearance [mm]")
    axes[1].grid(alpha=0.25)
    axes[2].plot(t, raw, color="#a9a9a9", label="unpruned pairs")
    axes[2].plot(t, active, color="#2ca02c", label="active Eq. (23) rows")
    axes[2].set_ylabel("pair / row count")
    axes[2].set_xlabel("time [s]")
    axes[2].set_yscale("log")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.25)
    fig.suptitle(f"UR5e LiuQP - {scene_name}")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def run_scene(
    scene: SceneDefinition,
    guidance: str,
    *,
    viewer: bool = False,
    render: bool = True,
) -> dict[str, float | int | str | bool | None]:
    output_dir = RESULTS_ROOT / f"{scene.name}_{guidance}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scene.xml").write_text(build_xml(scene), encoding="utf-8")

    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(scene.q0))
    robot_spheres = build_robot_certificate(model)
    obstacle_centers, obstacle_radii, _ = sample_box_sphere_tree(scene.boxes)
    controller = LiuQPController(
        model,
        data,
        scene,
        robot_spheres,
        obstacle_centers,
        obstacle_radii,
    )

    targets = [np.asarray(scene.waypoints[-1])]
    if guidance == "prescribed":
        targets = [np.asarray(point) for point in scene.waypoints]
    target_index = 0
    final_target = np.asarray(scene.waypoints[-1])
    steps = int(round(scene.duration / DT))
    arrival_tolerance = 0.035
    rows: list[dict[str, float | int | str]] = []

    passive = None
    if viewer:
        from mujoco import viewer as mujoco_viewer

        passive = mujoco_viewer.launch_passive(model, data)

    renderer = None
    render_error = ""
    if render:
        try:
            renderer = mujoco.Renderer(model, height=720, width=960)
        except Exception as exc:  # platform-dependent OpenGL setup
            render_error = f"{type(exc).__name__}: {exc}"

    checkpoint_steps = {0, steps // 3, 2 * steps // 3, steps - 1}
    started_wall = time.perf_counter()
    for step in range(steps):
        target = targets[target_index]
        ee_before = attachment_position(model, data)
        if (
            np.linalg.norm(target - ee_before) <= arrival_tolerance
            and target_index + 1 < len(targets)
        ):
            target_index += 1
            target = targets[target_index]

        qdot, metrics = controller.solve(target)
        scale, clearance = integrate_with_safety_backtracking(
            model,
            data,
            qdot,
            robot_spheres,
            obstacle_centers,
            obstacle_radii,
            controller.safety_margin,
        )
        ee = attachment_position(model, data)
        world_contact = minimum_mujoco_world_contact(data, model)
        self_contact = minimum_mujoco_self_contact(data, model)
        row = metrics.as_dict()
        row.update(
            {
                "step": step,
                "time_s": step * DT,
                "target_index": target_index,
                "target_x": float(target[0]),
                "target_y": float(target[1]),
                "target_z": float(target[2]),
                "ee_x": float(ee[0]),
                "ee_y": float(ee[1]),
                "ee_z": float(ee[2]),
                "final_target_error_m": float(np.linalg.norm(final_target - ee)),
                "accepted_step_scale": scale,
                "certificate_clearance_m": clearance,
                "mujoco_world_contact_dist_m": world_contact,
                "mujoco_self_contact_dist_m": self_contact,
            }
        )
        for joint_index, joint_position in enumerate(data.qpos[:6], start=1):
            row[f"q{joint_index}_rad"] = float(joint_position)
        rows.append(row)

        if renderer is not None and step in checkpoint_steps:
            try:
                render_frame(renderer, data, output_dir / f"checkpoint_{step:04d}.png")
            except Exception as exc:
                render_error = f"{type(exc).__name__}: {exc}"
                renderer.close()
                renderer = None
        if passive is not None:
            passive.sync()
            time.sleep(DT)
            if not passive.is_running():
                break

        if target_index == len(targets) - 1 and np.linalg.norm(final_target - ee) < 0.018:
            # Hold a short tail so the final state appears in logs/rendering.
            if step > 10 and all(
                float(previous["final_target_error_m"]) < 0.018 for previous in rows[-10:]
            ):
                break

    elapsed = time.perf_counter() - started_wall
    if passive is not None:
        passive.close()
    if renderer is not None:
        try:
            render_frame(renderer, data, output_dir / "checkpoint_final.png")
        finally:
            renderer.close()

    csv_path = output_dir / "trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_plot(rows, output_dir / "metrics.png", scene.name)

    final_error = float(rows[-1]["final_target_error_m"])
    certificate_min = float(min(float(row["certificate_clearance_m"]) for row in rows))
    finite_world = [
        float(row["mujoco_world_contact_dist_m"])
        for row in rows
        if np.isfinite(float(row["mujoco_world_contact_dist_m"]))
    ]
    finite_self = [
        float(row["mujoco_self_contact_dist_m"])
        for row in rows
        if np.isfinite(float(row["mujoco_self_contact_dist_m"]))
    ]
    solved_fraction = float(
        np.mean([str(row["status"]).startswith("solved") for row in rows])
    )
    active = np.asarray([int(row["active_obstacle_rows"]) for row in rows])
    raw = np.asarray([int(row["raw_obstacle_spheres"]) for row in rows])
    summary: dict[str, float | int | str | bool | None] = {
        "scene": scene.name,
        "scene_description": scene.description,
        "guidance": guidance,
        "success": bool(final_error < 0.03 and certificate_min >= -2.0e-5),
        "steps": len(rows),
        "simulated_time_s": float(rows[-1]["time_s"]),
        "wall_time_s": float(elapsed),
        "final_error_m": final_error,
        "minimum_certificate_clearance_m": certificate_min,
        "minimum_mujoco_world_contact_distance_m": (
            float(min(finite_world)) if finite_world else None
        ),
        "minimum_mujoco_self_contact_distance_m": (
            float(min(finite_self)) if finite_self else None
        ),
        "solved_qp_fraction": solved_fraction,
        "mean_qp_solve_ms": float(np.mean([float(row["solve_ms"]) for row in rows])),
        "p95_qp_solve_ms": float(np.percentile([float(row["solve_ms"]) for row in rows], 95)),
        "robot_certificate_spheres": len(robot_spheres),
        "obstacle_spheres": len(obstacle_centers),
        "mean_unpruned_pairs": float(np.mean(raw)),
        "mean_active_hyperplanes": float(np.mean(active)),
        "mean_reduction_fraction": float(1.0 - np.mean(active / raw)),
        "max_near_penalty_terms": int(max(int(row["near_penalty_terms"]) for row in rows)),
        "max_contact_repulsion_rows": int(max(int(row["contact_repulsion_rows"]) for row in rows)),
        "minimum_accepted_step_scale": float(min(float(row["accepted_step_scale"]) for row in rows)),
        "render_error": render_error,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("all", *SCENES), default="all")
    parser.add_argument("--guidance", choices=("prescribed", "direct"), default="prescribed")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    selected = list(SCENES) if args.scene == "all" else [args.scene]
    summaries = []
    for name in selected:
        summary = run_scene(
            SCENES[name],
            args.guidance,
            viewer=args.viewer,
            render=not args.no_render,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    aggregate_name = (
        f"all_{args.guidance}_summaries.json"
        if args.scene == "all"
        else f"{args.scene}_{args.guidance}_summary.json"
    )
    (RESULTS_ROOT / aggregate_name).write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
