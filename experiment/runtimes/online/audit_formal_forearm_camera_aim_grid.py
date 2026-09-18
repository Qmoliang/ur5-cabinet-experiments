"""Read-only visibility audit for the frozen wrist/forearm camera layout.

Candidate aims are ranked only by collision-free mounting and whether the two
physical cameras see every surface witness before it becomes collision-relevant.
Controller success is deliberately not part of the ranking.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from audit_camera_yaw_grid import _camera_cycles, _visible_indices, _yaw_quaternion
from audit_secondary_shoulder_mount_grid import (
    LENS_PROTRUSION_FROM_HOUSING_CENTER,
    _camera_quaternion_for_target,
    _has_penetration,
)
from model import DT, build_model, set_configuration
from observability_evaluator import _first_risk_cycles, _sample_truth_surfaces
from protocol_drawer_scene import DRAWER_Y, DRAWER_Z, formal_drawer_two_camera_scene


TRUTH_SPACING = 0.012
OBSERVATION_DISTANCE = 0.10
GUARD_TIME = 0.10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    scene = formal_drawer_two_camera_scene()
    q_history = np.load(args.run_dir / "q_history.npy")
    q_trajectory = np.vstack((np.asarray(scene.q0, dtype=float), q_history))
    truth, box_ranges = _sample_truth_surfaces(scene.boxes, TRUTH_SPACING)
    correction = math.sqrt(2.0) * TRUTH_SPACING / 2.0
    first_risk = _first_risk_cycles(
        scene,
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

    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(scene.q0, dtype=float))
    body_id = model.body("forearm_link").id
    body_rotation = data.xmat[body_id].reshape(3, 3).copy()
    body_position = data.xpos[body_id].copy()
    wrist_id = model.camera("ur5_depth_wrist").id
    forearm_id = model.camera("ur5_depth_forearm").id
    wrist_housing_id = model.geom("formal_d405_wrist_housing").id
    housing_id = model.geom("formal_d405_forearm_housing").id
    option = mujoco.MjvOption()
    option.geomgroup[1] = 1
    geomgroup = np.asarray(option.geomgroup, dtype=np.uint8)

    rows: list[dict] = []
    wrist_position = model.cam_pos[wrist_id].copy()
    for wrist_yaw_deg in (0.0, 10.0, 20.0, 30.0):
        wrist_quaternion = _yaw_quaternion(wrist_yaw_deg)
        yaw = math.radians(wrist_yaw_deg)
        wrist_optical = np.array([math.sin(yaw), math.cos(yaw), 0.0])
        model.cam_quat[wrist_id] = wrist_quaternion
        model.geom_quat[wrist_housing_id] = wrist_quaternion
        model.geom_pos[wrist_housing_id] = (
            wrist_position
            - LENS_PROTRUSION_FROM_HOUSING_CENTER * wrist_optical
        )
        for mount_x in (0.0,):
            for mount_z in (0.26, 0.28, 0.30, 0.31, 0.32):
                local_position = np.array(
                    [mount_x, 0.075, mount_z], dtype=float
                )
                for target_y_offset in (-0.20, -0.18, -0.16, -0.14, -0.12):
                    for target_z_offset in (-0.08, -0.06, -0.04):
                        target = np.array(
                            [
                                0.35,
                                DRAWER_Y + target_y_offset,
                                DRAWER_Z + target_z_offset,
                            ],
                            dtype=float,
                        )
                        quaternion, optical_local = _camera_quaternion_for_target(
                            body_rotation,
                            body_position,
                            local_position,
                            target,
                        )
                        model.cam_pos[forearm_id] = local_position
                        model.cam_quat[forearm_id] = quaternion
                        model.geom_pos[housing_id] = (
                            local_position
                            - LENS_PROTRUSION_FROM_HOUSING_CENTER * optical_local
                        )
                        model.geom_quat[housing_id] = quaternion
                        collides, first_contact, minimum_contact = _has_penetration(
                            model, data, q_trajectory
                        )
                        first_seen = np.full(
                            len(risk_indices), sentinel, dtype=np.int32
                        )
                        if not collides:
                            for cycle in camera_cycles:
                                eligible = np.flatnonzero(
                                    (first_seen == sentinel) & (cycle <= deadlines)
                                )
                                if not len(eligible):
                                    break
                                set_configuration(
                                    model, data, q_trajectory[cycle]
                                )
                                for camera_id in (wrist_id, forearm_id):
                                    remaining = eligible[
                                        first_seen[eligible] == sentinel
                                    ]
                                    if not len(remaining):
                                        break
                                    visible_local = _visible_indices(
                                        model,
                                        data,
                                        camera_id,
                                        truth[risk_indices[remaining]],
                                        geomgroup,
                                        image_aspect=320.0 / 180.0,
                                    )
                                    first_seen[remaining[visible_local]] = cycle
                        missed = first_seen == sentinel
                        ever_seen_after_deadline = np.zeros(
                            len(risk_indices), dtype=bool
                        )
                        remaining_missed = np.flatnonzero(missed)
                        if not collides and len(remaining_missed):
                            for cycle in camera_cycles:
                                remaining = remaining_missed[
                                    ~ever_seen_after_deadline[remaining_missed]
                                ]
                                if not len(remaining):
                                    break
                                set_configuration(
                                    model, data, q_trajectory[cycle]
                                )
                                for camera_id in (wrist_id, forearm_id):
                                    still_remaining = remaining[
                                        ~ever_seen_after_deadline[remaining]
                                    ]
                                    if not len(still_remaining):
                                        break
                                    visible_local = _visible_indices(
                                        model,
                                        data,
                                        camera_id,
                                        truth[risk_indices[still_remaining]],
                                        geomgroup,
                                        image_aspect=320.0 / 180.0,
                                    )
                                    ever_seen_after_deadline[
                                        still_remaining[visible_local]
                                    ] = True
                        missed_by_box: dict[str, int] = {}
                        for item in box_ranges:
                            in_box = (
                                (risk_indices >= item.start)
                                & (risk_indices < item.stop)
                            )
                            missed_by_box[item.box.name] = int(
                                np.count_nonzero(missed & in_box)
                            )
                        rows.append(
                            {
                                "wrist_yaw_deg": wrist_yaw_deg,
                                "wrist_quaternion_wxyz": (
                                    wrist_quaternion.tolist()
                                ),
                                "wrist_housing_center_m": model.geom_pos[
                                    wrist_housing_id
                                ].tolist(),
                                "local_position_m": local_position.tolist(),
                                "target_m": target.tolist(),
                                "target_y_offset_m": target_y_offset,
                                "target_z_offset_m": target_z_offset,
                                "quaternion_wxyz": quaternion.tolist(),
                                "housing_center_m": model.geom_pos[
                                    housing_id
                                ].tolist(),
                                "collision_free_on_source_trajectory": (
                                    not collides
                                ),
                                "first_contact_cycle": first_contact,
                                "minimum_contact_distance_m": minimum_contact,
                                "risk_samples": int(len(risk_indices)),
                                "visible_before_deadline": int(
                                    np.count_nonzero(~missed)
                                ),
                                "missed_before_deadline": int(
                                    np.count_nonzero(missed)
                                ),
                                "missed_but_eventually_visible": int(
                                    np.count_nonzero(
                                        missed & ever_seen_after_deadline
                                    )
                                ),
                                "never_visible_during_trajectory": int(
                                    np.count_nonzero(
                                        missed & ~ever_seen_after_deadline
                                    )
                                ),
                                "missed_point_bounds_m": (
                                    {
                                        "minimum": np.min(
                                            truth[risk_indices[missed]], axis=0
                                        ).tolist(),
                                        "maximum": np.max(
                                            truth[risk_indices[missed]], axis=0
                                        ).tolist(),
                                    }
                                    if np.any(missed)
                                    else None
                                ),
                                "missed_first_risk_cycle_range": (
                                    [
                                        int(
                                            np.min(
                                                first_risk[
                                                    risk_indices[missed]
                                                ]
                                            )
                                        ),
                                        int(
                                            np.max(
                                                first_risk[
                                                    risk_indices[missed]
                                                ]
                                            )
                                        ),
                                    ]
                                    if np.any(missed)
                                    else None
                                ),
                                "missed_deadline_cycle_range": (
                                    [
                                        int(np.min(deadlines[missed])),
                                        int(np.max(deadlines[missed])),
                                    ]
                                    if np.any(missed)
                                    else None
                                ),
                                "missed_by_box": missed_by_box,
                            }
                        )

    rows.sort(
        key=lambda row: (
            not row["collision_free_on_source_trajectory"],
            row["missed_before_deadline"],
        )
    )
    report = {
        "audit_role": "formal_two_camera_visibility_screen",
        "controller_success_used_for_ranking": False,
        "truth_feedback_to_control": False,
        "source_run": str(args.run_dir.resolve()),
        "mount_body": "forearm_link",
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
