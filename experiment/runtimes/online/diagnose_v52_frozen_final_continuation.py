"""Continue final-goal-only LiuQP on a completed run's frozen final proxies.

This is a read-only development diagnostic.  It never changes or appends to
the source run and its output is not formal trajectory evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from model import build_robot_certificate, set_configuration
from run_protocol_v3_async_online import DT, _protocol_scene
from run_protocol_v3_online_ablation import _build_mvt_only, _new_controller


def resolve_run(path: Path) -> Path:
    if (path / "summary.json").exists():
        return path
    children = [item for item in path.iterdir() if (item / "summary.json").exists()]
    if len(children) != 1:
        raise ValueError(f"expected exactly one run below {path}, found {len(children)}")
    return children[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--cycles", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cycles <= 0:
        parser.error("--cycles must be positive")

    run = resolve_run(args.run)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary["representation"] != "ellipsoid":
        raise ValueError("this exact support-sum diagnostic requires an ellipsoid run")
    archive = np.load(run / "final_causal_proxies.npz")
    proxies = SimpleNamespace(
        proxy_ids=np.asarray(archive["proxy_ids"], dtype=np.int64),
        centers=np.asarray(archive["centers"], dtype=float),
        sphere_radii=np.asarray(archive["sphere_radii"], dtype=float),
        base_sphere_radii=np.asarray(archive["base_sphere_radii"], dtype=float),
        base_ellipsoid_shapes=np.asarray(archive["ellipsoid_shapes"], dtype=float),
        ellipsoid_outer_shapes=np.asarray(
            archive["ellipsoid_outer_shapes"], dtype=float
        ),
        proxy_uncertainty_shapes=np.asarray(
            archive["proxy_uncertainty_shapes"], dtype=float
        ),
        proxy_offset_radii=np.asarray(archive["uncertainty_offsets"], dtype=float),
    )
    archive.close()

    model = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
    data = mujoco.MjData(model)
    q = np.asarray(np.load(run / "q_history.npy")[-1], dtype=float)
    set_configuration(model, data, q)
    scene = _protocol_scene(
        str(summary.get("camera_scene_version", summary["scene_version"]))
    )
    target = np.asarray(scene.waypoints[-1], dtype=float)
    robot = build_robot_certificate(model)
    robot_radii = np.asarray([item.radius for item in robot], dtype=float)
    mvt = _build_mvt_only(proxies, "ellipsoid", robot_radii, simd=True)
    controller = _new_controller(
        "ellipsoid",
        model,
        data,
        scene,
        robot,
        proxies,
        mvt,
        ellipsoid_pair_threads=int(summary["ellipsoid_support_pair_threads"]),
        ellipsoid_pair_affinity_mask=int(summary["ellipsoid_pair_affinity_mask"]),
    )
    controller.task_gain = float(summary["task_gain"])
    controller.max_task_speed = float(summary["max_task_speed_m_per_s"])
    controller.osqp_adaptive_row_threshold = int(
        summary["configured_osqp_adaptive_row_threshold"]
    )
    controller.osqp_absolute_tolerance = float(summary["osqp_absolute_tolerance"])
    controller.osqp_relative_tolerance = float(summary["osqp_relative_tolerance"])
    controller.osqp_max_iterations = int(summary["osqp_max_iterations"])

    initial_ee, _, _ = controller.task_feedback(target)
    initial_error = float(np.linalg.norm(target - initial_ee))
    errors: list[float] = []
    statuses: dict[str, int] = {}
    success_hold = 0
    first_hold_completion = None
    limiting_counts: dict[str, int] = {}
    try:
        for cycle in range(args.cycles):
            qdot, metrics = controller.solve(target)
            statuses[metrics.status] = statuses.get(metrics.status, 0) + 1
            q = q + np.asarray(qdot, dtype=float) * DT
            set_configuration(model, data, q)
            ee, _, _ = controller.task_feedback(target)
            error = float(np.linalg.norm(target - ee))
            errors.append(error)
            success_hold = success_hold + 1 if error < 0.001 else 0
            if success_hold >= 50 and first_hold_completion is None:
                first_hold_completion = cycle
            active = [row for row in controller.last_pair_records if row.qp_collision_row]
            if active:
                limiting = min(active, key=lambda row: abs(row.clearance))
                key = f"robot={limiting.robot_index},proxy_id={limiting.proxy_id}"
                limiting_counts[key] = limiting_counts.get(key, 0) + 1
    finally:
        mvt.close()

    result = {
        "diagnostic_only": True,
        "feeds_back_to_formal_run": False,
        "source_run": str(run.resolve()),
        "source_proxy_count": int(len(proxies.centers)),
        "continuation_cycles": int(args.cycles),
        "initial_error_m": initial_error,
        "minimum_error_m": float(np.min(errors)),
        "final_error_m": float(errors[-1]),
        "strict_less_than_1mm_hold50": first_hold_completion is not None,
        "first_hold_completion_cycle": first_hold_completion,
        "statuses": statuses,
        "most_common_limiting_pairs": sorted(
            limiting_counts.items(), key=lambda item: (-item[1], item[0])
        )[:10],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
