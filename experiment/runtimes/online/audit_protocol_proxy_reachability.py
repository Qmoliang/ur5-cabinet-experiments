"""Independent goal-IK audit for the causal planning snapshot.

The script never writes a path consumed by the controller.  It diagnoses
whether a target configuration exists under the same proxy certificate and
also reports MuJoCo exact contacts for each converged IK solution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from configuration_corridor_planner import _ProxyChecker, _solve_ik
from ellipsoid_model import ellipsoid_world_state
from incremental_drawer_scene import incremental_drawer_v12_certified_candidate_scene
from model import JOINT_NAMES, attachment_position, build_model, set_configuration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--seeds", type=int, default=280)
    parser.add_argument("--certificate", choices=("sphere", "ellipsoid"), default="ellipsoid")
    parser.add_argument("--fixed-initial-orientation", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    arrays = np.load(args.snapshot)
    scene = incremental_drawer_v12_certified_candidate_scene()
    model = build_model(scene)
    data = mujoco.MjData(model)
    q0 = np.asarray(scene.q0)
    set_configuration(model, data, q0)
    target_rotation = (
        data.site_xmat[model.site("attachment_site").id].reshape(3, 3).copy()
        if args.fixed_initial_orientation
        else None
    )
    lower = np.array([model.jnt_range[model.joint(name).id, 0] for name in JOINT_NAMES])
    upper = np.array([model.jnt_range[model.joint(name).id, 1] for name in JOINT_NAMES])
    shapes = arrays["ellipsoid_shapes"] if args.certificate == "ellipsoid" else np.asarray(
        [np.eye(3) * radius**2 for radius in arrays["sphere_radii"]]
    )
    uncertainty = arrays["proxy_uncertainty_shapes"] if args.certificate == "ellipsoid" else np.zeros_like(shapes)
    pruning = arrays["ellipsoid_outer_shapes"] if args.certificate == "ellipsoid" else shapes
    checker = _ProxyChecker(
        arrays["centers"], shapes, arrays["proxy_offsets"], 0.001,
        robot_certificate=args.certificate,
        uncertainty_shapes=uncertainty,
        pruning_shapes=pruning,
    )
    rng = np.random.default_rng(8617)
    seeds = [q0.copy()]
    seeds.extend(rng.uniform(lower + 0.01, upper - 0.01) for _ in range(args.seeds))
    records = []

    def proxy_detail(q):
        set_configuration(checker.model, checker.data, q)
        positions, _, _, robot_shapes = ellipsoid_world_state(
            checker.model, checker.data, checker.robot
        )
        best = None
        for robot_index, (position, robot_shape) in enumerate(zip(positions, robot_shapes)):
            for plane in checker.controller.prune_redundant_obstacle_ellipsoids(
                position, robot_shape, robot_index=robot_index
            ):
                if best is None or plane.clearance < best["clearance_m"]:
                    proxy = checker.robot[robot_index]
                    best = {
                        "clearance_m": float(plane.clearance),
                        "robot_proxy_index": robot_index,
                        "robot_body": proxy.body_name,
                        "robot_source_geom": int(proxy.source_geom_id),
                        "robot_semi_axes_m": proxy.semi_axes.tolist(),
                        "obstacle_proxy_index": int(plane.obstacle_index),
                        "normal": plane.normal_to_obstacle.tolist(),
                    }
        return best
    try:
        for seed in seeds:
            q, error = _solve_ik(
                model, data, seed, np.asarray(scene.waypoints[-1]), target_rotation, lower, upper
            )
            if error > 0.0015:
                continue
            clearance = checker.clearance(q)
            set_configuration(model, data, q)
            contacts = int(data.ncon)
            records.append({
                "ik_error_m": error,
                "proxy_clearance_m": clearance,
                "exact_contact_count": contacts,
                "q": q.tolist(),
                "achieved": attachment_position(model, data).tolist(),
                "limiting_proxy": proxy_detail(q),
                "exact_contacts": [
                    {
                        "geom1": model.geom(int(data.contact[index].geom1)).name,
                        "geom2": model.geom(int(data.contact[index].geom2)).name,
                        "distance_m": float(data.contact[index].dist),
                    }
                    for index in range(data.ncon)
                ],
            })
    finally:
        checker.close()
    records.sort(key=lambda row: row["proxy_clearance_m"], reverse=True)
    report = {
        "scene": scene.name,
        "certificate": args.certificate,
        "orientation_constraint": (
            "fixed_initial" if args.fixed_initial_orientation else "position_task_free_orientation"
        ),
        "snapshot": str(args.snapshot.resolve()),
        "attempted": len(seeds),
        "converged": len(records),
        "proxy_safe": sum(row["proxy_clearance_m"] >= 0.0 for row in records),
        "exact_contact_free": sum(row["exact_contact_count"] == 0 for row in records),
        "best": records[:10],
    }
    output = args.output or args.snapshot.with_name(f"goal_ik_audit_{args.certificate}.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
