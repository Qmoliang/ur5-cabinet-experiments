"""Screen physical secondary shoulder-camera mounts before online experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from audit_camera_yaw_grid import _camera_cycles, _visible_indices
from model import DT, build_model, set_configuration
from observability_evaluator import _first_risk_cycles, _sample_truth_surfaces
from protocol_drawer_scene import (
    DRAWER_Y,
    DRAWER_Z,
    protocol_drawer_v6_scene,
    protocol_drawer_v6s4_scene,
)


TRUTH_SPACING = 0.012
OBSERVATION_DISTANCE = 0.10
GUARD_TIME = 0.10
LENS_PROTRUSION_FROM_HOUSING_CENTER = 0.012575


def _camera_quaternion_for_target(
    body_rotation: np.ndarray,
    body_position: np.ndarray,
    local_position: np.ndarray,
    world_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_position = body_position + body_rotation @ local_position
    optical_world = world_target - world_position
    optical_world /= np.linalg.norm(optical_world)
    optical_local = body_rotation.T @ optical_world
    up_local = body_rotation.T @ np.array([0.0, 0.0, 1.0])
    camera_y = up_local - optical_local * float(up_local @ optical_local)
    camera_y /= np.linalg.norm(camera_y)
    camera_z = -optical_local
    camera_x = np.cross(camera_y, camera_z)
    rotation = np.column_stack((camera_x, camera_y, camera_z))
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    return quaternion, optical_local


def _has_penetration(
    model: mujoco.MjModel, data: mujoco.MjData, q_trajectory: np.ndarray
) -> tuple[bool, int | None, float]:
    minimum = 0.0
    for cycle, q in enumerate(q_trajectory):
        set_configuration(model, data, q)
        penetrating = [
            float(data.contact[index].dist)
            for index in range(data.ncon)
            if float(data.contact[index].dist) < -1.0e-8
        ]
        if penetrating:
            minimum = min(minimum, min(penetrating))
            return True, cycle, minimum
    return False, None, minimum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    base_scene = protocol_drawer_v6_scene()
    candidate_scene = protocol_drawer_v6s4_scene()
    q_history = np.load(args.run_dir / "q_history.npy")
    q_trajectory = np.vstack((np.asarray(base_scene.q0, dtype=float), q_history))

    truth, box_ranges = _sample_truth_surfaces(base_scene.boxes, TRUTH_SPACING)
    correction = math.sqrt(2.0) * TRUTH_SPACING / 2.0
    first_risk = _first_risk_cycles(
        base_scene,
        q_trajectory,
        truth,
        box_ranges,
        OBSERVATION_DISTANCE,
        correction,
    )
    sentinel = np.iinfo(np.int32).max
    risk_indices = np.flatnonzero(first_risk != sentinel)
    guard_cycles = int(math.ceil(GUARD_TIME / DT))
    deadlines = np.maximum(0, first_risk[risk_indices] - guard_cycles)
    camera_cycles = _camera_cycles(len(q_trajectory))

    model = build_model(candidate_scene)
    data = mujoco.MjData(model)
    shoulder_body = model.body("shoulder_link").id
    set_configuration(model, data, np.asarray(candidate_scene.q0, dtype=float))
    body_rotation = data.xmat[shoulder_body].reshape(3, 3).copy()
    body_position = data.xpos[shoulder_body].copy()
    camera_id = model.camera("ur5_depth_shoulder_right").id
    housing_id = model.geom("protocol_d405_shoulder_housing_right").id
    camera_ids = (
        model.camera("ur5_depth_wrist").id,
        model.camera("ur5_depth_wrist_right").id,
        model.camera("ur5_depth_shoulder").id,
        camera_id,
    )
    option = mujoco.MjvOption()
    option.geomgroup[1] = 1
    geomgroup = np.asarray(option.geomgroup, dtype=np.uint8)
    target = np.array([0.55, DRAWER_Y, DRAWER_Z], dtype=float)

    x_values = (-0.16, -0.12, -0.08, -0.04, 0.04, 0.08, 0.12, 0.16)
    y_values = (-0.16, -0.12, 0.12, 0.16)
    z_values = (0.08, 0.12, 0.16)
    rows: list[dict] = []
    for x in x_values:
        for y in y_values:
            for z in z_values:
                local_position = np.array([x, y, z], dtype=float)
                quaternion, optical_local = _camera_quaternion_for_target(
                    body_rotation,
                    body_position,
                    local_position,
                    target,
                )
                model.cam_pos[camera_id] = local_position
                model.cam_quat[camera_id] = quaternion
                model.geom_pos[housing_id] = (
                    local_position
                    - LENS_PROTRUSION_FROM_HOUSING_CENTER * optical_local
                )
                model.geom_quat[housing_id] = quaternion
                collides, first_contact, minimum_contact = _has_penetration(
                    model, data, q_trajectory
                )
                row = {
                    "local_position_m": local_position.tolist(),
                    "housing_center_m": model.geom_pos[housing_id].tolist(),
                    "collision_free_on_source_trajectory": not collides,
                    "first_contact_cycle": first_contact,
                    "minimum_contact_distance_m": minimum_contact,
                    "risk_samples": int(len(risk_indices)),
                    "visible_before_deadline": 0,
                    "missed_before_deadline": int(len(risk_indices)),
                }
                if collides:
                    rows.append(row)
                    continue

                first_seen = np.full(len(risk_indices), sentinel, dtype=np.int32)
                for cycle in camera_cycles:
                    eligible = np.flatnonzero(
                        (first_seen == sentinel) & (cycle <= deadlines)
                    )
                    if not len(eligible):
                        break
                    set_configuration(model, data, q_trajectory[cycle])
                    for current_camera in camera_ids:
                        remaining = eligible[first_seen[eligible] == sentinel]
                        if not len(remaining):
                            break
                        visible_local = _visible_indices(
                            model,
                            data,
                            current_camera,
                            truth[risk_indices[remaining]],
                            geomgroup,
                        )
                        first_seen[remaining[visible_local]] = cycle
                missed = first_seen == sentinel
                row["visible_before_deadline"] = int(np.count_nonzero(~missed))
                row["missed_before_deadline"] = int(np.count_nonzero(missed))
                missed_by_box: dict[str, int] = {}
                for item in box_ranges:
                    in_box = (
                        (risk_indices >= item.start) & (risk_indices < item.stop)
                    )
                    missed_by_box[item.box.name] = int(
                        np.count_nonzero(missed & in_box)
                    )
                row["missed_by_box"] = missed_by_box
                rows.append(row)

    rows.sort(
        key=lambda row: (
            not row["collision_free_on_source_trajectory"],
            row["missed_before_deadline"],
        )
    )
    report = {
        "audit_role": "physical_secondary_shoulder_mount_visibility_screen",
        "controller_success_used_for_ranking": False,
        "truth_feedback_to_control": False,
        "source_run": str(args.run_dir.resolve()),
        "mount_body": "shoulder_link",
        "housing_half_size_m": [0.021075, 0.021075, 0.011575],
        "candidate_count": len(rows),
        "ranking": rows,
        "elapsed_s": time.perf_counter() - started,
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
