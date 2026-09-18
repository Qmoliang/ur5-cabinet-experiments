"""Visibility-only D405 mount audit on frozen sphere and ellipsoid runs.

Candidates are ranked only by collision-free physical housings and causal
visibility of every truth-surface witness before risk.  Controller success,
proxy geometry, and QP output are not inputs to the ranking.
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
D405_VERTICAL_FOV_DEG = 58.0


def _wrist_quaternion(yaw_degrees: float, pitch_degrees: float):
    """Aim wrist-local +Y with yaw about +Z and an upward pitch."""

    yaw = math.radians(float(yaw_degrees))
    pitch = math.radians(float(pitch_degrees))
    optical = np.array(
        [
            math.sin(yaw) * math.cos(pitch),
            math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
        ]
    )
    reference_up = np.array([0.0, 0.0, 1.0])
    camera_y = reference_up - optical * float(reference_up @ optical)
    camera_y /= np.linalg.norm(camera_y)
    camera_z = -optical
    camera_x = np.cross(camera_y, camera_z)
    rotation = np.column_stack((camera_x, camera_y, camera_z))
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    return quaternion, optical


def _trajectory_context(run_dir: Path, scene):
    history = np.load(run_dir / "q_history.npy")
    trajectory = np.vstack((np.asarray(scene.q0, dtype=float), history))
    truth, ranges = _sample_truth_surfaces(scene.boxes, TRUTH_SPACING)
    correction = math.sqrt(2.0) * TRUTH_SPACING / 2.0
    first_risk = _first_risk_cycles(
        scene,
        trajectory,
        truth,
        ranges,
        OBSERVATION_DISTANCE,
        correction,
    )
    sentinel = np.iinfo(np.int32).max
    risk_indices = np.flatnonzero(first_risk != sentinel)
    deadlines = np.maximum(
        0,
        first_risk[risk_indices]
        - int(math.ceil(GUARD_TIME / DT)),
    )
    return {
        "trajectory": trajectory,
        "truth": truth,
        "ranges": ranges,
        "risk_indices": risk_indices,
        "deadlines": deadlines,
        "camera_cycles": _camera_cycles(len(trajectory)),
    }


def _visibility_row(model, data, context, camera_ids, geomgroup):
    sentinel = np.iinfo(np.int32).max
    risk_indices = context["risk_indices"]
    first_seen = np.full(len(risk_indices), sentinel, dtype=np.int32)
    for cycle in context["camera_cycles"]:
        eligible = np.flatnonzero(
            (first_seen == sentinel) & (cycle <= context["deadlines"])
        )
        if not len(eligible):
            break
        set_configuration(model, data, context["trajectory"][cycle])
        for camera_id in camera_ids:
            remaining = eligible[first_seen[eligible] == sentinel]
            if not len(remaining):
                break
            visible = _visible_indices(
                model,
                data,
                camera_id,
                context["truth"][risk_indices[remaining]],
                geomgroup,
                image_aspect=320.0 / 180.0,
            )
            first_seen[remaining[visible]] = cycle
    missed = first_seen == sentinel
    missed_by_box = {}
    for item in context["ranges"]:
        in_box = (risk_indices >= item.start) & (risk_indices < item.stop)
        missed_by_box[item.box.name] = int(np.count_nonzero(missed & in_box))
    return {
        "risk_samples": int(len(risk_indices)),
        "visible_before_deadline": int(np.count_nonzero(~missed)),
        "missed_before_deadline": int(np.count_nonzero(missed)),
        "missed_by_box": missed_by_box,
        "missed_point_bounds_m": (
            None
            if not np.any(missed)
            else {
                "minimum": np.min(
                    context["truth"][risk_indices[missed]], axis=0
                ).tolist(),
                "maximum": np.max(
                    context["truth"][risk_indices[missed]], axis=0
                ).tolist(),
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ellipsoid_run_dir", type=Path)
    parser.add_argument("sphere_run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    scene = formal_drawer_two_camera_scene()
    contexts = {
        "ellipsoid": _trajectory_context(args.ellipsoid_run_dir, scene),
        "sphere": _trajectory_context(args.sphere_run_dir, scene),
    }
    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(scene.q0, dtype=float))
    wrist_id = model.camera("ur5_depth_wrist").id
    forearm_id = model.camera("ur5_depth_forearm").id
    wrist_housing_id = model.geom("formal_d405_wrist_housing").id
    forearm_housing_id = model.geom("formal_d405_forearm_housing").id
    forearm_body_id = model.body("forearm_link").id
    body_rotation = data.xmat[forearm_body_id].reshape(3, 3).copy()
    body_position = data.xpos[forearm_body_id].copy()
    wrist_position = model.cam_pos[wrist_id].copy()
    model.cam_fovy[wrist_id] = D405_VERTICAL_FOV_DEG
    model.cam_fovy[forearm_id] = D405_VERTICAL_FOV_DEG
    option = mujoco.MjvOption()
    option.geomgroup[1] = 1
    geomgroup = np.asarray(option.geomgroup, dtype=np.uint8)

    rows = []
    # The first 72-candidate screen showed that the only remaining deadline
    # misses were on the drawer ceiling.  Expand the wrist elevation without
    # changing either frozen trajectory or the two physical mounting sites.
    # Ranking still uses collision and visibility only, never QP success.
    for wrist_yaw_deg in (-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0):
      for wrist_pitch_deg in (-30.0, -15.0, 0.0, 15.0, 30.0):
        wrist_quat, wrist_optical = _wrist_quaternion(
            wrist_yaw_deg, wrist_pitch_deg
        )
        model.cam_quat[wrist_id] = wrist_quat
        model.geom_quat[wrist_housing_id] = wrist_quat
        model.geom_pos[wrist_housing_id] = (
            wrist_position
            - LENS_PROTRUSION_FROM_HOUSING_CENTER * wrist_optical
        )
        for mount_z in (0.31,):
            local_position = np.array([0.0, 0.075, mount_z])
            for target_y_offset in (-0.20,):
                for target_z_offset in (-0.04,):
                    target = np.array(
                        [
                            0.35,
                            DRAWER_Y + target_y_offset,
                            DRAWER_Z + target_z_offset,
                        ]
                    )
                    forearm_quat, optical_local = _camera_quaternion_for_target(
                        body_rotation,
                        body_position,
                        local_position,
                        target,
                    )
                    model.cam_pos[forearm_id] = local_position
                    model.cam_quat[forearm_id] = forearm_quat
                    model.geom_pos[forearm_housing_id] = (
                        local_position
                        - LENS_PROTRUSION_FROM_HOUSING_CENTER * optical_local
                    )
                    model.geom_quat[forearm_housing_id] = forearm_quat
                    collision = {}
                    collision_free = True
                    for name, context in contexts.items():
                        hit, cycle, distance = _has_penetration(
                            model, data, context["trajectory"]
                        )
                        collision[name] = {
                            "collision_free": not hit,
                            "first_contact_cycle": cycle,
                            "minimum_contact_distance_m": distance,
                        }
                        collision_free &= not hit
                    visibility = {}
                    if collision_free:
                        for name, context in contexts.items():
                            visibility[name] = _visibility_row(
                                model,
                                data,
                                context,
                                (wrist_id, forearm_id),
                                geomgroup,
                            )
                    else:
                        for name, context in contexts.items():
                            visibility[name] = {
                                "risk_samples": int(len(context["risk_indices"])),
                                "visible_before_deadline": 0,
                                "missed_before_deadline": int(
                                    len(context["risk_indices"])
                                ),
                            }
                    missed = [
                        visibility[name]["missed_before_deadline"]
                        for name in ("ellipsoid", "sphere")
                    ]
                    rows.append(
                        {
                            "wrist_yaw_deg": wrist_yaw_deg,
                            "wrist_pitch_deg": wrist_pitch_deg,
                            "wrist_quaternion_wxyz": wrist_quat.tolist(),
                            "wrist_housing_center_m": model.geom_pos[
                                wrist_housing_id
                            ].tolist(),
                            "forearm_local_position_m": local_position.tolist(),
                            "forearm_target_m": target.tolist(),
                            "forearm_target_y_offset_m": target_y_offset,
                            "forearm_target_z_offset_m": target_z_offset,
                            "forearm_quaternion_wxyz": forearm_quat.tolist(),
                            "forearm_housing_center_m": model.geom_pos[
                                forearm_housing_id
                            ].tolist(),
                            "d405_vertical_fov_deg": D405_VERTICAL_FOV_DEG,
                            "collision_free_on_both_trajectories": collision_free,
                            "collision": collision,
                            "visibility": visibility,
                            "maximum_missed_before_deadline": max(missed),
                            "total_missed_before_deadline": sum(missed),
                        }
                    )

    rows.sort(
        key=lambda row: (
            not row["collision_free_on_both_trajectories"],
            row["maximum_missed_before_deadline"],
            row["total_missed_before_deadline"],
        )
    )
    report = {
        "audit_role": "formal_two_trajectory_d405_visibility_screen",
        "controller_success_used_for_ranking": False,
        "truth_feedback_to_control": False,
        "vertical_fov_deg": D405_VERTICAL_FOV_DEG,
        "source_runs": {
            "ellipsoid": str(args.ellipsoid_run_dir.resolve()),
            "sphere": str(args.sphere_run_dir.resolve()),
        },
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
