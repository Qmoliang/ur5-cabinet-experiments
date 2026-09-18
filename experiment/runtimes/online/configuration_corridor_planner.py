"""Automatic full-robot configuration corridor over an IRIS task path."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares

from ellipsoid_model import (
    build_robot_ellipsoid_certificate,
    ellipsoid_world_state,
    spheres_as_isotropic_ellipsoids,
)
from ellipsoid_qp_controller import EllipsoidLiuQPController
from incremental_drawer_scene import incremental_drawer_v12_certified_candidate_scene
from model import (
    JOINT_NAMES,
    attachment_position,
    build_model,
    build_robot_certificate,
    set_configuration,
)
from native_mvt import NativeMVT


@dataclass(frozen=True)
class ConfigurationCorridor:
    q_targets: np.ndarray
    task_targets: np.ndarray
    task_rotations: np.ndarray
    minimum_proxy_clearance: float
    attempted_goal_ik: int
    tested_goal_branches: int
    elapsed_ms: float
    planning_method: str


class _ProxyChecker:
    def __init__(
        self,
        centers,
        shapes,
        offsets,
        safety_margin: float,
        robot_certificate: str = "ellipsoid",
        uncertainty_shapes: np.ndarray | None = None,
        pruning_shapes: np.ndarray | None = None,
    ):
        self.scene = incremental_drawer_v12_certified_candidate_scene()
        self.model = build_model(self.scene)
        self.data = mujoco.MjData(self.model)
        if robot_certificate == "ellipsoid":
            self.robot = build_robot_ellipsoid_certificate(self.model)
        elif robot_certificate == "sphere":
            self.robot = spheres_as_isotropic_ellipsoids(
                build_robot_certificate(self.model)
            )
        else:
            raise ValueError("robot_certificate must be sphere or ellipsoid")
        maximum_radius = max(float(np.max(item.semi_axes)) for item in self.robot)
        self.index = NativeMVT.from_ellipsoids(
            centers,
            shapes if pruning_shapes is None else pruning_shapes,
            offsets,
            voxel_size=maximum_radius + 0.003,
            query_padding=0.120 + safety_margin,
            simd=True,
        )
        self.controller = EllipsoidLiuQPController(
            self.model,
            self.data,
            self.scene,
            self.robot,
            centers,
            shapes,
            obstacle_offsets=offsets,
            obstacle_uncertainty_shapes=uncertainty_shapes,
            obstacle_pruning_shapes=pruning_shapes,
            safety_margin=safety_margin,
            obstacle_index=self.index,
        )
        self._clearance_cache: dict[bytes, float] = {}

    def clearance(self, q: np.ndarray) -> float:
        key = np.round(np.asarray(q, dtype=float), decimals=10).tobytes()
        cached = self._clearance_cache.get(key)
        if cached is not None:
            return cached
        set_configuration(self.model, self.data, q)
        positions, _, _, shapes = ellipsoid_world_state(
            self.model, self.data, self.robot
        )
        best = np.inf
        for robot_index, (position, shape) in enumerate(zip(positions, shapes)):
            planes = self.controller.prune_redundant_obstacle_ellipsoids(
                position, shape, robot_index=robot_index
            )
            if planes:
                best = min(best, min(item.clearance for item in planes))
        value = float(best)
        self._clearance_cache[key] = value
        return value

    def edge_clearance(
        self, first: np.ndarray, second: np.ndarray, maximum_joint_step: float
    ) -> float:
        count = max(
            2,
            int(math.ceil(float(np.max(np.abs(second - first))) / maximum_joint_step)) + 1,
        )
        return min(self.clearance(q) for q in np.linspace(first, second, count))

    def edge_clearance_in_task_tube(
        self,
        first: np.ndarray,
        second: np.ndarray,
        task_first: np.ndarray,
        task_second: np.ndarray,
        maximum_joint_step: float,
        tube_radius: float = 0.140,
    ) -> float:
        """Proxy clearance with an FK certificate around an IRIS path edge."""

        count = max(
            2,
            int(math.ceil(float(np.max(np.abs(second - first))) / maximum_joint_step)) + 1,
        )
        best = np.inf
        for fraction, q in zip(
            np.linspace(0.0, 1.0, count), np.linspace(first, second, count)
        ):
            value = self.clearance(q)
            if value < 0.0:
                return value
            # clearance() leaves checker.data at q.
            expected = (1.0 - fraction) * task_first + fraction * task_second
            deviation = float(
                np.linalg.norm(attachment_position(self.model, self.data) - expected)
            )
            best = min(best, value, tube_radius - deviation)
            if deviation > tube_radius:
                return tube_radius - deviation
        return float(best)

    def close(self) -> None:
        self.index.close()


def _solve_ik(model, data, seed, target, target_rotation, lower, upper):
    seed = np.clip(seed, lower + 1.0e-5, upper - 1.0e-5)
    site_id = model.site("attachment_site").id

    def residual(q):
        set_configuration(model, data, q)
        position = attachment_position(model, data) - target
        if target_rotation is None:
            return position
        current = data.site_xmat[site_id].reshape(3, 3)
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
        xtol=1.0e-9,
        ftol=1.0e-9,
        gtol=1.0e-9,
        max_nfev=180,
    )
    return result.x.copy(), float(np.linalg.norm(residual(result.x)))


def _unwrap_near(q: np.ndarray, reference: np.ndarray, lower, upper) -> np.ndarray:
    """Choose the collision-equivalent 2*pi representative nearest reference."""

    result = np.asarray(q, dtype=float).copy()
    for index in range(len(result)):
        choices = result[index] + 2.0 * np.pi * np.arange(-2, 3)
        choices = choices[(choices >= lower[index]) & (choices <= upper[index])]
        if len(choices):
            result[index] = choices[int(np.argmin(np.abs(choices - reference[index])))]
    return result


def _rrt_connect_configuration_path(
    checker: _ProxyChecker,
    q_start: np.ndarray,
    goals: list[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    *,
    maximum_iterations: int = 2400,
    extension_step: float = 0.14,
    search_edge_step: float = 0.055,
    final_edge_step: float = 0.025,
) -> tuple[np.ndarray, float] | None:
    """Deterministic bidirectional C-space fallback using proxy checks only."""

    def trace(nodes, parents, index):
        path = []
        while index >= 0:
            path.append(nodes[index])
            index = parents[index]
        return list(reversed(path))

    def steer(first, second):
        delta = second - first
        scale = min(1.0, extension_step / max(float(np.max(np.abs(delta))), 1.0e-12))
        return np.clip(first + scale * delta, lower + 1.0e-5, upper - 1.0e-5)

    for raw_goal in goals:
        goal = _unwrap_near(raw_goal, q_start, lower, upper)
        if checker.clearance(goal) < 0.0:
            continue
        a_nodes, a_parents = [q_start.copy()], [-1]
        b_nodes, b_parents = [goal.copy()], [-1]
        a_is_start = True

        def extend(nodes, parents, target):
            values = np.asarray(nodes)
            nearest = int(np.argmin(np.linalg.norm(values - target, axis=1)))
            candidate = steer(values[nearest], target)
            if checker.edge_clearance(values[nearest], candidate, search_edge_step) < 0.0:
                return None
            nodes.append(candidate)
            parents.append(nearest)
            return len(nodes) - 1

        for iteration in range(maximum_iterations):
            if iteration % 10 == 0:
                sample = b_nodes[0]
            elif iteration % 5 == 0:
                sample = rng.uniform(lower + 0.01, upper - 0.01)
            else:
                fraction = rng.uniform()
                line = (1.0 - fraction) * q_start + fraction * goal
                sigma = 0.10 + 0.45 * math.sin(math.pi * fraction)
                sample = np.clip(
                    line + rng.normal(0.0, sigma, size=len(q_start)),
                    lower + 0.01,
                    upper - 0.01,
                )
            a_index = extend(a_nodes, a_parents, sample)
            if a_index is not None:
                meet = a_nodes[a_index]
                b_index = None
                for _ in range(32):
                    candidate = extend(b_nodes, b_parents, meet)
                    if candidate is None:
                        break
                    b_index = candidate
                    if float(np.max(np.abs(b_nodes[b_index] - meet))) <= 1.0e-9:
                        a_path = trace(a_nodes, a_parents, a_index)
                        b_path = trace(b_nodes, b_parents, b_index)
                        coarse = (
                            a_path + list(reversed(b_path[:-1]))
                            if a_is_start
                            else b_path + list(reversed(a_path[:-1]))
                        )
                        minimum = np.inf
                        valid = True
                        for first, second in zip(coarse[:-1], coarse[1:]):
                            value = checker.edge_clearance(
                                first, second, final_edge_step
                            )
                            minimum = min(minimum, value)
                            if value < 0.0:
                                valid = False
                                break
                        if valid:
                            return np.asarray(coarse), float(minimum)
                        break
            a_nodes, b_nodes = b_nodes, a_nodes
            a_parents, b_parents = b_parents, a_parents
            a_is_start = not a_is_start
    return None


def _bidirectional_task_layer_path(
    checker: _ProxyChecker,
    model,
    data,
    task_targets: np.ndarray,
    q_start: np.ndarray,
    goal_candidates: list[np.ndarray],
    target_rotation: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    *,
    search_edge_step: float = 0.045,
    final_edge_step: float = 0.025,
    diagnostics: dict | None = None,
) -> tuple[np.ndarray, float] | None:
    """Bidirectional multi-branch IK tree on autonomous IRIS path layers."""

    targets = np.asarray(task_targets, dtype=float)
    # Keep every autonomous IRIS control layer.  The resulting ~24 mm task
    # spacing prevents a valid IK branch from being lost by coarse layer
    # interpolation, while each accepted joint edge is still rechecked.
    layers_xyz = targets

    def options(index: int) -> list[np.ndarray]:
        center = layers_xyz[index]
        if index in (0, len(layers_xyz) - 1):
            return [center]
        tangent = layers_xyz[index + 1] - layers_xyz[index - 1]
        tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
        axis = np.eye(3)[int(np.argmin(np.abs(tangent)))]
        first = np.cross(tangent, axis)
        first /= max(float(np.linalg.norm(first)), 1.0e-12)
        second = np.cross(tangent, first)
        return [center] + [
            center + radius * (math.cos(angle) * first + math.sin(angle) * second)
            for radius in (0.040, 0.080, 0.120)
            for angle in np.linspace(0.0, 2.0 * math.pi, 4, endpoint=False)
        ]

    def append_unique(values, parents, q, parent):
        if all(float(np.linalg.norm(q - old)) > 0.08 for old in values):
            values.append(q)
            parents.append(parent)

    forward_layers: list[list[np.ndarray]] = [[q_start.copy()]]
    forward_parents: list[list[int]] = [[-1]]
    for layer in range(1, len(layers_xyz)):
        previous = forward_layers[-1]
        candidates, parents = [], []
        seeded = [(q.copy(), point) for q in previous for point in options(layer)]
        seeded.extend(
            (rng.uniform(lower + 0.01, upper - 0.01), point)
            for point in options(layer)
            for _ in range(4)
        )
        for seed, point in seeded:
            layer_rotation = (
                target_rotation if layer == len(layers_xyz) - 1 else None
            )
            q, error = _solve_ik(
                model, data, seed, point, layer_rotation, lower, upper
            )
            if error > 0.0015 or checker.clearance(q) < 0.0:
                continue
            order = np.argsort([np.linalg.norm(q - old) for old in previous])
            for parent in order:
                if checker.edge_clearance_in_task_tube(
                    previous[int(parent)], q,
                    layers_xyz[layer - 1], layers_xyz[layer], search_edge_step
                ) >= 0.0:
                    append_unique(candidates, parents, q, int(parent))
                    break
            if len(candidates) >= 30:
                break
        if not candidates:
            break
        forward_layers.append(candidates)
        forward_parents.append(parents)
    if diagnostics is not None:
        diagnostics["selected_task_layers"] = len(layers_xyz)
        diagnostics["forward_layer_counts"] = [len(item) for item in forward_layers]

    last = len(layers_xyz) - 1
    safe_goals = []
    for goal in goal_candidates:
        q = _unwrap_near(goal, q_start, lower, upper)
        if checker.clearance(q) >= 0.0:
            safe_goals.append(q)
    if not safe_goals:
        if diagnostics is not None:
            diagnostics["failure"] = "no_safe_goal_after_unwrap"
        return None
    backward_layers: dict[int, list[np.ndarray]] = {last: safe_goals}
    backward_next: dict[int, list[int]] = {last: [-1] * len(safe_goals)}
    meet = None
    for layer in range(last - 1, -1, -1):
        following = backward_layers[layer + 1]
        candidates, next_nodes = [], []
        if layer == 0:
            for next_index, next_q in enumerate(following):
                if checker.edge_clearance_in_task_tube(
                    q_start,
                    next_q,
                    layers_xyz[0],
                    layers_xyz[1],
                    search_edge_step,
                ) >= 0.0:
                    candidates = [q_start.copy()]
                    next_nodes = [next_index]
                    break
        seeded = [(q.copy(), point) for q in following for point in options(layer)]
        seeded.extend(
            (rng.uniform(lower + 0.01, upper - 0.01), point)
            for point in options(layer)
            for _ in range(5)
        )
        for seed, point in seeded:
            if layer == 0 and candidates:
                break
            layer_rotation = target_rotation if layer == 0 else None
            q, error = _solve_ik(
                model, data, seed, point, layer_rotation, lower, upper
            )
            if error > 0.0015 or checker.clearance(q) < 0.0:
                continue
            order = np.argsort([np.linalg.norm(q - old) for old in following])
            for next_index in order:
                if checker.edge_clearance_in_task_tube(
                    q, following[int(next_index)],
                    layers_xyz[layer], layers_xyz[layer + 1], search_edge_step
                ) >= 0.0:
                    append_unique(candidates, next_nodes, q, int(next_index))
                    break
            if len(candidates) >= 30:
                break
        if not candidates:
            break
        backward_layers[layer] = candidates
        backward_next[layer] = next_nodes
        if layer < len(forward_layers):
            for fi, fq in enumerate(forward_layers[layer]):
                for bi, bq in enumerate(candidates):
                    if checker.edge_clearance_in_task_tube(
                        fq, bq, layers_xyz[layer], layers_xyz[layer],
                        search_edge_step
                    ) >= 0.0:
                        meet = (layer, fi, bi)
                        break
                if meet is not None:
                    break
        if meet is not None:
            break
    if meet is None:
        if diagnostics is not None:
            diagnostics["backward_layer_counts"] = {
                int(key): len(value) for key, value in backward_layers.items()
            }
            diagnostics["failure"] = "bidirectional_layers_did_not_meet"
        return None

    layer, forward_node, backward_node = meet
    first_half = []
    node = forward_node
    for index in range(layer, -1, -1):
        first_half.append(forward_layers[index][node])
        node = forward_parents[index][node]
    first_half.reverse()
    second_half = []
    node = backward_node
    for index in range(layer, last + 1):
        second_half.append(backward_layers[index][node])
        node = backward_next[index][node]
    coarse = np.asarray(first_half + second_half)
    coarse_task = np.asarray(
        list(layers_xyz[: layer + 1]) + list(layers_xyz[layer:])
    )
    minimum = np.inf
    for first, second, task_first, task_second in zip(
        coarse[:-1], coarse[1:], coarse_task[:-1], coarse_task[1:]
    ):
        value = checker.edge_clearance_in_task_tube(
            first, second, task_first, task_second, final_edge_step
        )
        minimum = min(minimum, value)
        if value < 0.0:
            if diagnostics is not None:
                diagnostics["failure"] = "final_25mrad_edge_recheck_failed"
            return None
    # Reject C-space branch changes whose FK interpolation turns a short IRIS
    # corridor into a large workspace excursion.  This is a representation-
    # independent quality gate, not an obstacle or path waypoint.
    sampled_task = []
    for first, second in zip(coarse[:-1], coarse[1:]):
        count = max(
            2,
            int(math.ceil(float(np.max(np.abs(second - first))) / final_edge_step)) + 1,
        )
        for q in np.linspace(first, second, count)[:-1]:
            set_configuration(model, data, q)
            sampled_task.append(attachment_position(model, data))
    set_configuration(model, data, coarse[-1])
    sampled_task.append(attachment_position(model, data))
    sampled_task = np.asarray(sampled_task)
    fk_length = float(np.linalg.norm(np.diff(sampled_task, axis=0), axis=1).sum())
    iris_length = float(np.linalg.norm(np.diff(targets, axis=0), axis=1).sum())
    if fk_length > 2.0 * iris_length:
        if diagnostics is not None:
            diagnostics["failure"] = "fk_path_length_exceeds_twice_iris_length"
            diagnostics["fk_path_length_m"] = fk_length
            diagnostics["iris_path_length_m"] = iris_length
        return None
    if diagnostics is not None:
        diagnostics["failure"] = None
    return coarse, float(minimum)


def plan_configuration_corridor(
    task_targets: np.ndarray,
    q_start: np.ndarray,
    obstacle_centers: np.ndarray,
    obstacle_shapes: np.ndarray,
    obstacle_offsets: np.ndarray,
    *,
    safety_margin: float = 0.006,
    random_seed: int = 8617,
    random_goal_seeds: int = 280,
    maximum_goal_candidates: int = 36,
    maximum_joint_step: float = 0.025,
    robot_certificate: str = "ellipsoid",
    obstacle_uncertainty_shapes: np.ndarray | None = None,
    obstacle_pruning_shapes: np.ndarray | None = None,
) -> ConfigurationCorridor:
    """Find a proxy-safe IK branch; no task coordinates are created here."""

    started = time.perf_counter()
    task_targets = np.asarray(task_targets, dtype=float).reshape(-1, 3)
    q_start = np.asarray(q_start, dtype=float).reshape(6)
    checker = _ProxyChecker(
        obstacle_centers,
        obstacle_shapes,
        obstacle_offsets,
        safety_margin,
        robot_certificate=robot_certificate,
        uncertainty_shapes=obstacle_uncertainty_shapes,
        pruning_shapes=obstacle_pruning_shapes,
    )
    model, data = checker.model, checker.data
    set_configuration(model, data, q_start)
    # Use one fixed, representation-independent retrieval attitude.  This is
    # deliberately stricter than the position-only task and prevents the IK
    # search from exploiting arbitrary wrist rotations differently between
    # sphere and ellipsoid trials.
    target_rotation = data.site_xmat[
        model.site("attachment_site").id
    ].reshape(3, 3).copy()
    lower = np.array(
        [model.jnt_range[model.joint(name).id, 0] for name in JOINT_NAMES]
    )
    upper = np.array(
        [model.jnt_range[model.joint(name).id, 1] for name in JOINT_NAMES]
    )
    rng = np.random.default_rng(random_seed)

    def finalize(
        coarse: np.ndarray, clearance: float, planning_method: str
    ) -> ConfigurationCorridor:
        dense = [np.asarray(coarse[0]).copy()]
        for first, second in zip(coarse[:-1], coarse[1:]):
            count = max(
                2,
                int(
                    math.ceil(
                        float(np.max(np.abs(second - first)))
                        / maximum_joint_step
                    )
                )
                + 1,
            )
            dense.extend(np.linspace(first, second, count)[1:])
        dense_q = np.asarray(dense)
        dense_task = []
        dense_rotations = []
        for q in dense_q:
            set_configuration(model, data, q)
            dense_task.append(attachment_position(model, data))
            dense_rotations.append(
                data.site_xmat[
                    model.site("attachment_site").id
                ].reshape(3, 3).copy()
            )
        return ConfigurationCorridor(
            q_targets=dense_q,
            task_targets=np.asarray(dense_task),
            task_rotations=np.asarray(dense_rotations),
            minimum_proxy_clearance=float(clearance),
            attempted_goal_ik=attempted,
            tested_goal_branches=len(goal_candidates),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            planning_method=planning_method,
        )
    seeds = [q_start.copy()]
    seeds.extend(
        rng.uniform(lower + 0.01, upper - 0.01)
        for _ in range(random_goal_seeds)
    )
    goal_candidates: list[np.ndarray] = []
    attempted = 0
    try:
        for seed in seeds:
            attempted += 1
            q, error = _solve_ik(
                model,
                data,
                seed,
                task_targets[-1],
                target_rotation,
                lower,
                upper,
            )
            if error > 0.0015 or checker.clearance(q) < 0.0:
                continue
            if all(np.linalg.norm(q - old) > 0.08 for old in goal_candidates):
                goal_candidates.append(q)
            if len(goal_candidates) >= maximum_goal_candidates:
                break
        layer_diagnostics: dict = {}
        layered = _bidirectional_task_layer_path(
            checker,
            model,
            data,
            task_targets,
            q_start,
            goal_candidates,
            target_rotation,
            lower,
            upper,
            rng,
            diagnostics=layer_diagnostics,
        )
        if layered is not None:
            coarse, clearance = layered
            return finalize(
                coarse, clearance, "bidirectional_iris_task_layers"
            )
        # Fast deterministic continuation is the common case.  Only invoke
        # the wider layered graph below when every direct IK branch breaks.
        for goal in goal_candidates:
            reverse_path = [goal]
            branch_clearance = checker.clearance(goal)
            valid = True
            for target in task_targets[-2:0:-1]:
                attempted += 1
                q, error = _solve_ik(
                    model,
                    data,
                    reverse_path[-1],
                    target,
                    target_rotation,
                    lower,
                    upper,
                )
                if error > 0.0015:
                    valid = False
                    break
                edge_value = checker.edge_clearance(
                    reverse_path[-1], q, maximum_joint_step
                )
                if edge_value < 0.0:
                    valid = False
                    break
                branch_clearance = min(branch_clearance, edge_value)
                reverse_path.append(q)
            if not valid:
                continue
            edge_value = checker.edge_clearance(
                reverse_path[-1], q_start, maximum_joint_step
            )
            if edge_value < 0.0:
                continue
            branch_clearance = min(branch_clearance, edge_value)
            return finalize(
                np.asarray([q_start] + list(reversed(reverse_path))),
                branch_clearance,
                "reverse_iris_task_layers",
            )
        last_layer = len(task_targets) - 1
        layers: dict[int, list[np.ndarray]] = {last_layer: goal_candidates}
        next_nodes: dict[int, list[int]] = {
            last_layer: [-1] * len(goal_candidates)
        }
        branch_clearances: dict[int, list[float]] = {
            last_layer: [checker.clearance(q) for q in goal_candidates]
        }
        layer_cap = 12
        random_layer_seeds = 8
        for layer in range(last_layer - 1, 0, -1):
            following = layers[layer + 1]
            layer_seeds = [value.copy() for value in following]
            layer_seeds.extend(
                rng.uniform(lower + 0.01, upper - 0.01)
                for _ in range(random_layer_seeds)
            )
            candidates: list[np.ndarray] = []
            candidate_next: list[int] = []
            candidate_clearances: list[float] = []
            for seed in layer_seeds:
                attempted += 1
                q, error = _solve_ik(
                    model,
                    data,
                    seed,
                    task_targets[layer],
                    target_rotation,
                    lower,
                    upper,
                )
                if error > 0.0015:
                    continue
                q_clearance = checker.clearance(q)
                if q_clearance < 0.0:
                    continue
                if any(np.linalg.norm(q - old) <= 0.08 for old in candidates):
                    continue
                order = np.argsort(
                    [np.linalg.norm(q - next_q) for next_q in following]
                )
                best_next = None
                best_value = -np.inf
                for next_index in order:
                    edge_value = checker.edge_clearance(
                        q, following[int(next_index)], maximum_joint_step
                    )
                    if edge_value < 0.0:
                        continue
                    value = min(
                        q_clearance,
                        edge_value,
                        branch_clearances[layer + 1][int(next_index)],
                    )
                    if value > best_value:
                        best_value = value
                        best_next = int(next_index)
                if best_next is None:
                    continue
                candidates.append(q)
                candidate_next.append(best_next)
                candidate_clearances.append(float(best_value))
                if len(candidates) >= layer_cap:
                    break
            if not candidates:
                break
            layers[layer] = candidates
            next_nodes[layer] = candidate_next
            branch_clearances[layer] = candidate_clearances

        best_path = None
        best_clearance = -np.inf
        if 1 in layers:
            for node, first_q in enumerate(layers[1]):
                edge_value = checker.edge_clearance(
                    q_start, first_q, maximum_joint_step
                )
                value = min(edge_value, branch_clearances[1][node])
                if value < 0.0 or value <= best_clearance:
                    continue
                path = [q_start.copy()]
                current = node
                for layer in range(1, last_layer + 1):
                    path.append(layers[layer][current])
                    current = next_nodes[layer][current]
                best_clearance = float(value)
                best_path = np.asarray(path)
        if best_path is None:
            raise RuntimeError(
                "no full-robot proxy-safe IK branch follows the autonomous task path; "
                f"layer_diagnostics={layer_diagnostics}"
            )
        return finalize(
            best_path, best_clearance, "reverse_layered_iris_task_graph"
        )
    finally:
        checker.close()
