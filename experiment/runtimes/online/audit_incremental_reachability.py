"""Independent exact-MuJoCo reachability audit for incremental drawer v2.

The generated joint path is evaluation evidence only.  The online LiuQP runner
never imports this module or reads its outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from incremental_drawer_scene import incremental_drawer_v3_protocol_scene
from model import (
    JOINT_NAMES,
    attachment_position,
    build_model,
    set_configuration,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results_incremental_v2" / "reachability_audit_v3"


def collision_free(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> bool:
    set_configuration(model, data, q)
    return all(data.contact[index].dist >= -1.0e-8 for index in range(data.ncon))


def edge_free(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    first: np.ndarray,
    second: np.ndarray,
    maximum_joint_step: float = 0.010,
) -> bool:
    count = max(
        2,
        int(math.ceil(float(np.max(np.abs(second - first))) / maximum_joint_step))
        + 1,
    )
    return all(collision_free(model, data, q) for q in np.linspace(first, second, count))


def solve_position_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    seed: np.ndarray,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    seed = np.clip(np.asarray(seed), lower + 1.0e-5, upper - 1.0e-5)

    def residual(q: np.ndarray) -> np.ndarray:
        set_configuration(model, data, q)
        position = attachment_position(model, data) - target
        if target_rotation is None:
            return position
        current = data.site_xmat[
            model.site("attachment_site").id
        ].reshape(3, 3)
        error_matrix = target_rotation @ current.T
        orientation = 0.5 * np.array(
            [
                error_matrix[2, 1] - error_matrix[1, 2],
                error_matrix[0, 2] - error_matrix[2, 0],
                error_matrix[1, 0] - error_matrix[0, 1],
            ]
        )
        return np.r_[position, orientation]

    result = least_squares(
        residual,
        seed,
        bounds=(lower + 1.0e-5, upper - 1.0e-5),
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
        max_nfev=240,
    )
    return result.x.copy(), float(np.linalg.norm(residual(result.x)))


def unique_append(values: list[np.ndarray], candidate: np.ndarray, tolerance: float = 0.08) -> None:
    if all(float(np.linalg.norm(candidate - value)) > tolerance for value in values):
        values.append(candidate.copy())


def audit(seed: int = 4317) -> dict:
    scene = incremental_drawer_v3_protocol_scene()
    model = build_model(scene)
    data = mujoco.MjData(model)
    q0 = np.asarray(scene.q0, dtype=float)
    set_configuration(model, data, q0)
    target_rotation = data.site_xmat[
        model.site("attachment_site").id
    ].reshape(3, 3).copy()
    start = attachment_position(model, data)
    target = np.asarray(scene.waypoints[-1], dtype=float)
    lower = np.array(
        [model.jnt_range[model.joint(name).id, 0] for name in JOINT_NAMES]
    )
    upper = np.array(
        [model.jnt_range[model.joint(name).id, 1] for name in JOINT_NAMES]
    )
    if not collision_free(model, data, q0):
        raise RuntimeError("candidate initial configuration is in collision")

    # The task-space samples define the property being audited (raise outside,
    # then insert). They are not available to either online controller.
    pre_entry = np.array([0.285, target[1], target[2]])
    first = np.linspace(start, pre_entry, 19)
    second = np.linspace(pre_entry, target, 31)[1:]
    task_samples = np.vstack([first, second])
    rng = np.random.default_rng(seed)

    def position_options(layer_index: int) -> list[np.ndarray]:
        point = task_samples[layer_index]
        if layer_index in (0, len(task_samples) - 1) or point[0] < 0.27:
            return [point]
        offsets = (
            (0.0, 0.0),
            (-0.018, 0.0),
            (0.018, 0.0),
            (0.0, -0.018),
            (0.0, 0.018),
            (-0.018, -0.018),
            (-0.018, 0.018),
            (0.018, -0.018),
            (0.018, 0.018),
        )
        return [
            point + np.array([0.0, dy, dz])
            for dy, dz in offsets
        ]

    forward_layers: list[list[np.ndarray]] = [[q0.copy()]]
    forward_parents: list[list[int]] = [[-1]]
    attempted_ik = 0
    for layer_index, point in enumerate(task_samples[1:], start=1):
        previous = forward_layers[-1]
        seeded_targets: list[tuple[np.ndarray, np.ndarray]] = [
            (value.copy(), option)
            for value in previous
            for option in position_options(layer_index)
        ]
        # Multiple deterministic seeds expose redundant IK branches early,
        # before the wrist crosses the drawer front.
        random_count = 5 if point[0] < 0.36 else 9
        seeded_targets.extend(
            (
                rng.uniform(lower + 0.01, upper - 0.01),
                option,
            )
            for option in position_options(layer_index)
            for _ in range(random_count)
        )
        candidates: list[np.ndarray] = []
        candidate_parents: list[int] = []
        for ik_seed, ik_target in seeded_targets:
            attempted_ik += 1
            q, error = solve_position_ik(
                model,
                data,
                ik_seed,
                ik_target,
                lower,
                upper,
                target_rotation,
            )
            if error > 0.0015 or not collision_free(model, data, q):
                continue
            for parent_index, parent in enumerate(previous):
                if edge_free(model, data, parent, q):
                    if all(
                        float(np.linalg.norm(q - old)) > 0.08
                        for old in candidates
                    ):
                        candidates.append(q)
                        candidate_parents.append(parent_index)
                    break
            if len(candidates) >= 30:
                break
        if not candidates:
            break
        forward_layers.append(candidates)
        forward_parents.append(candidate_parents)

    # A one-way layer expansion can discard the IK branch that becomes useful
    # only after insertion. Expand independently from many collision-free goal
    # IK solutions and require an exact collision-free bridge between the two
    # trees. This is an audit planner, never controller guidance.
    goal_candidates: list[np.ndarray] = []
    goal_seeds = list(forward_layers[-1])
    goal_seeds.extend(
        rng.uniform(lower + 0.01, upper - 0.01) for _ in range(420)
    )
    for ik_seed in goal_seeds:
        attempted_ik += 1
        q, error = solve_position_ik(
            model,
            data,
            ik_seed,
            task_samples[-1],
            lower,
            upper,
            target_rotation,
        )
        if error <= 0.0015 and collision_free(model, data, q):
            unique_append(goal_candidates, q)
        if len(goal_candidates) >= 36:
            break
    if not goal_candidates:
        return {
            "passed": False,
            "failure": "no_collision_free_goal_ik",
            "attempted_ik": attempted_ik,
            "path_role": "offline_audit_never_controller_input",
        }

    last_layer = len(task_samples) - 1
    backward_layers: dict[int, list[np.ndarray]] = {
        last_layer: goal_candidates
    }
    backward_next: dict[int, list[int]] = {
        last_layer: [-1] * len(goal_candidates)
    }
    meet: tuple[int, int, int] | None = None
    for layer_index in range(last_layer - 1, -1, -1):
        following = backward_layers[layer_index + 1]
        seeded_targets = [
            (value.copy(), option)
            for value in following
            for option in position_options(layer_index)
        ]
        random_count = (
            6 if task_samples[layer_index][0] < 0.36 else 10
        )
        seeded_targets.extend(
            (
                rng.uniform(lower + 0.01, upper - 0.01),
                option,
            )
            for option in position_options(layer_index)
            for _ in range(random_count)
        )
        candidates = []
        candidate_next = []
        for ik_seed, ik_target in seeded_targets:
            attempted_ik += 1
            q, error = solve_position_ik(
                model,
                data,
                ik_seed,
                ik_target,
                lower,
                upper,
                target_rotation,
            )
            if error > 0.0015 or not collision_free(model, data, q):
                continue
            for next_index, next_q in enumerate(following):
                if edge_free(model, data, q, next_q):
                    if all(
                        float(np.linalg.norm(q - old)) > 0.08
                        for old in candidates
                    ):
                        candidates.append(q)
                        candidate_next.append(next_index)
                    break
            if len(candidates) >= 30:
                break
        if not candidates:
            break
        backward_layers[layer_index] = candidates
        backward_next[layer_index] = candidate_next
        if layer_index < len(forward_layers):
            for forward_index, forward_q in enumerate(
                forward_layers[layer_index]
            ):
                for backward_index, backward_q in enumerate(candidates):
                    if edge_free(model, data, forward_q, backward_q):
                        meet = (layer_index, forward_index, backward_index)
                        break
                if meet is not None:
                    break
        if meet is not None:
            break

    if meet is None:
        return {
            "passed": False,
            "failure": "bidirectional_exact_audit_trees_did_not_connect",
            "attempted_ik": attempted_ik,
            "forward_reachable_layers": len(forward_layers),
            "backward_reachable_min_layer": min(backward_layers),
            "path_role": "offline_audit_never_controller_input",
        }

    meet_layer, forward_node, backward_node = meet
    forward_path = []
    node = forward_node
    for layer_index in range(meet_layer, -1, -1):
        forward_path.append(forward_layers[layer_index][node])
        node = forward_parents[layer_index][node]
    forward_path.reverse()
    backward_path = []
    node = backward_node
    for layer_index in range(meet_layer, last_layer + 1):
        backward_path.append(backward_layers[layer_index][node])
        node = backward_next[layer_index][node]
    coarse_path = forward_path + backward_path

    dense: list[np.ndarray] = [coarse_path[0]]
    minimum_exact_distance = math.inf
    for first_q, second_q in zip(coarse_path[:-1], coarse_path[1:]):
        count = max(
            2,
            int(math.ceil(float(np.max(np.abs(second_q - first_q))) / 0.01))
            + 1,
        )
        for q in np.linspace(first_q, second_q, count)[1:]:
            set_configuration(model, data, q)
            if data.ncon:
                minimum_exact_distance = min(
                    minimum_exact_distance,
                    min(data.contact[index].dist for index in range(data.ncon)),
                )
            if not collision_free(model, data, q):
                raise AssertionError("dense verification found a collision")
            dense.append(q.copy())
    dense_path = np.asarray(dense)
    set_configuration(model, data, dense_path[-1])
    final_error = float(np.linalg.norm(attachment_position(model, data) - target))
    payload = np.ascontiguousarray(dense_path).tobytes()
    path_hash = hashlib.sha256(payload).hexdigest()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "exact_audit_path_not_controller_input.npz",
        q=dense_path,
        task_samples=task_samples,
    )
    result = {
        "passed": final_error <= 0.018,
        "scene": scene.name,
        "path_role": "offline_audit_never_controller_input",
        "fixed_attachment_orientation": True,
        "attempted_ik": attempted_ik,
        "task_layers": len(task_samples),
        "coarse_configurations": len(coarse_path),
        "dense_verified_configurations": len(dense_path),
        "maximum_dense_joint_step_rad": float(
            np.max(np.abs(np.diff(dense_path, axis=0)))
        ),
        "minimum_reported_contact_distance_m": (
            None if math.isinf(minimum_exact_distance) else minimum_exact_distance
        ),
        "final_position_error_m": final_error,
        "path_sha256": path_hash,
    }
    (OUTPUT / "audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
