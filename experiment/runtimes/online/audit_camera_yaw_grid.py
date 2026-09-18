"""Visibility-only wrist-camera yaw screening on a saved collision-free run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from model import DT, build_model, set_configuration
from observability_evaluator import _first_risk_cycles, _sample_truth_surfaces
from protocol_drawer_scene import CAMERA_HZ, protocol_drawer_v6_scene


CAMERA_MINIMUM_RANGE = 0.070
CAMERA_MAXIMUM_RANGE = 1.60
OBSERVATION_DISTANCE = 0.10
GUARD_TIME = 0.10
TRUTH_SPACING = 0.012


def _yaw_quaternion(yaw_degrees: float) -> np.ndarray:
    """Camera optical axis yawed from wrist-local +Y about local +Z."""

    yaw = math.radians(float(yaw_degrees))
    optical = np.array([math.sin(yaw), math.cos(yaw), 0.0])
    camera_y = np.array([0.0, 0.0, 1.0])
    camera_z = -optical
    camera_x = np.cross(camera_y, camera_z)
    rotation = np.column_stack((camera_x, camera_y, camera_z))
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    return quaternion


def _camera_cycles(count: int) -> list[int]:
    cycles: list[int] = []
    previous_tick = -1
    for cycle in range(count):
        tick = int(np.floor(cycle * DT * CAMERA_HZ + 1.0e-12))
        if tick != previous_tick:
            cycles.append(cycle)
            previous_tick = tick
    return cycles


def _visible_indices(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    points: np.ndarray,
    geomgroup: np.ndarray,
    image_aspect: float = 240.0 / 180.0,
) -> np.ndarray:
    if not len(points):
        return np.empty(0, dtype=np.int64)
    position = data.cam_xpos[camera_id]
    rotation = data.cam_xmat[camera_id].reshape(3, 3)
    displacement = points - position
    ranges = np.linalg.norm(displacement, axis=1)
    local = displacement @ rotation
    depth = -local[:, 2]
    fovy = math.radians(float(model.cam_fovy[camera_id]))
    half_y = 0.5 * fovy
    half_x = math.atan(math.tan(half_y) * float(image_aspect))
    horizontal = np.arctan2(local[:, 0], depth)
    vertical = np.arctan2(local[:, 1], depth)
    in_frustum = (
        (depth > 0.0)
        & (np.abs(horizontal) <= half_x)
        & (np.abs(vertical) <= half_y)
        & (ranges >= CAMERA_MINIMUM_RANGE)
        & (ranges <= CAMERA_MAXIMUM_RANGE)
    )
    selected = np.flatnonzero(in_frustum)
    if not len(selected):
        return selected
    directions = np.ascontiguousarray(
        displacement[selected] / ranges[selected, None], dtype=np.float64
    )
    hit_ranges = np.full(len(selected), -1.0, dtype=np.float64)
    geom_ids = np.full(len(selected), -1, dtype=np.int32)
    mujoco.mj_multiRay(
        model,
        data,
        np.ascontiguousarray(position, dtype=np.float64),
        directions.reshape(-1),
        geomgroup,
        True,
        -1,
        geom_ids,
        hit_ranges,
        None,
        len(selected),
        CAMERA_MAXIMUM_RANGE,
    )
    first_hit_matches = (hit_ranges >= 0.0) & (
        np.abs(hit_ranges - ranges[selected]) <= 1.0e-4
    )
    return selected[first_hit_matches]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--angles", default="-40,-20,0,20,40")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    angles = tuple(float(value) for value in args.angles.split(","))
    scene = protocol_drawer_v6_scene()
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
    left_id = model.camera("ur5_depth_wrist").id
    right_id = model.camera("ur5_depth_wrist_right").id
    shoulder_id = model.camera("ur5_depth_shoulder").id
    option = mujoco.MjvOption()
    option.geomgroup[1] = 1
    geomgroup = np.asarray(option.geomgroup, dtype=np.uint8)

    rows: list[dict] = []
    for left_yaw in angles:
        for right_yaw in angles:
            model.cam_quat[left_id] = _yaw_quaternion(left_yaw)
            model.cam_quat[right_id] = _yaw_quaternion(right_yaw)
            first_seen = np.full(len(risk_indices), sentinel, dtype=np.int32)
            for cycle in camera_cycles:
                eligible = np.flatnonzero(
                    (first_seen == sentinel) & (cycle <= deadlines)
                )
                if not len(eligible):
                    break
                set_configuration(model, data, q_trajectory[cycle])
                for camera_id in (left_id, right_id, shoulder_id):
                    remaining = eligible[first_seen[eligible] == sentinel]
                    if not len(remaining):
                        break
                    visible_local = _visible_indices(
                        model,
                        data,
                        camera_id,
                        truth[risk_indices[remaining]],
                        geomgroup,
                    )
                    first_seen[remaining[visible_local]] = cycle
            missed = first_seen == sentinel
            missed_by_box: dict[str, int] = {}
            for item in box_ranges:
                in_box = (
                    (risk_indices >= item.start) & (risk_indices < item.stop)
                )
                missed_by_box[item.box.name] = int(
                    np.count_nonzero(missed & in_box)
                )
            rows.append(
                {
                    "left_yaw_deg": left_yaw,
                    "right_yaw_deg": right_yaw,
                    "risk_samples": int(len(risk_indices)),
                    "visible_before_deadline": int(np.count_nonzero(~missed)),
                    "missed_before_deadline": int(np.count_nonzero(missed)),
                    "missed_by_box": missed_by_box,
                }
            )
    rows.sort(
        key=lambda row: (
            row["missed_before_deadline"],
            abs(row["left_yaw_deg"] - 20.0)
            + abs(row["right_yaw_deg"] + 20.0),
        )
    )
    report = {
        "audit_role": "visibility_only_camera_candidate_screen",
        "controller_success_used_for_ranking": False,
        "truth_feedback_to_control": False,
        "source_run": str(args.run_dir.resolve()),
        "source_run_required_collision_free": True,
        "observation_distance_m": OBSERVATION_DISTANCE,
        "guard_time_s": GUARD_TIME,
        "camera_minimum_range_m": CAMERA_MINIMUM_RANGE,
        "candidate_angles_deg": list(angles),
        "camera_sample_cycles": len(camera_cycles),
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
