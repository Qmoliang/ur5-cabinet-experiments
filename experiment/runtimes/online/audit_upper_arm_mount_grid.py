"""Physical upper-arm RGB-D mount screen on a saved contact-free trajectory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from audit_camera_yaw_grid import _camera_cycles, _visible_indices
from audit_secondary_shoulder_mount_grid import (
    LENS_PROTRUSION_FROM_HOUSING_CENTER,
    _camera_quaternion_for_target,
    _has_penetration,
)
from model import DT, build_model, set_configuration
from observability_evaluator import _first_risk_cycles, _sample_truth_surfaces
from protocol_drawer_scene import (
    DRAWER_Y,
    DRAWER_Z,
    protocol_drawer_camera_mount_grid_scene,
    protocol_drawer_v6_scene,
)


TRUTH_SPACING = 0.012
OBSERVATION_DISTANCE = 0.10
GUARD_TIME = 0.10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    base_scene = protocol_drawer_v6_scene()
    candidate_scene = protocol_drawer_camera_mount_grid_scene()
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
    body_id = model.body("upper_arm_link").id
    set_configuration(model, data, np.asarray(candidate_scene.q0, dtype=float))
    body_rotation = data.xmat[body_id].reshape(3, 3).copy()
    body_position = data.xpos[body_id].copy()
    camera_id = model.camera("ur5_depth_upper_arm_candidate").id
    housing_id = model.geom("protocol_d405_upper_arm_candidate_housing").id
    camera_ids = (
        model.camera("ur5_depth_wrist").id,
        model.camera("ur5_depth_wrist_right").id,
        model.camera("ur5_depth_shoulder").id,
        camera_id,
    )
    option = mujoco.MjvOption()
    option.geomgroup[1] = 1
    geomgroup = np.asarray(option.geomgroup, dtype=np.uint8)
    target_y_values = (DRAWER_Y - 0.12, DRAWER_Y, DRAWER_Y + 0.12)

    radius = 0.078
    radial_directions = (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (math.sqrt(0.5), math.sqrt(0.5)),
        (math.sqrt(0.5), -math.sqrt(0.5)),
        (-math.sqrt(0.5), math.sqrt(0.5)),
        (-math.sqrt(0.5), -math.sqrt(0.5)),
    )
    z_values = (0.10, 0.20, 0.30)
    rows: list[dict] = []
    for radial_x, radial_y in radial_directions:
        for z in z_values:
            for target_y in target_y_values:
                target = np.array([0.55, target_y, DRAWER_Z], dtype=float)
                local_position = np.array(
                    [radius * radial_x, radius * radial_y, z], dtype=float
                )
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
                "aim_point_m": target.tolist(),
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
        "audit_role": "physical_upper_arm_mount_visibility_screen",
        "controller_success_used_for_ranking": False,
        "truth_feedback_to_control": False,
        "source_run": str(args.run_dir.resolve()),
        "mount_body": "upper_arm_link",
        "upper_arm_capsule_radius_m": 0.05,
        "camera_center_radial_offset_m": radius,
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
