"""Run final-goal LiuQP on a frozen v5.3 directional-leaf snapshot.

The diagnostic replaces each compact environment proxy by the already
published CenterVox directional leaf that it covers.  It is read-only and
does not feed commands or results back into the source run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import mujoco
import numpy as np

from model import build_robot_certificate, set_configuration
from pointcloud_proxy import minkowski_outer_shapes
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
    parser.add_argument("--cycles", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = resolve_run(args.run)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    archive = np.load(run / "final_causal_proxies.npz", allow_pickle=False)
    leaf_centers = np.asarray(archive["filtered_points"], dtype=float)
    leaf_u = np.asarray(archive["filtered_uncertainty_shapes"], dtype=float)
    leaf_delta = np.asarray(archive["filtered_point_offsets"], dtype=float)
    archive.close()
    count = len(leaf_centers)
    core_radius = 1.0e-4
    core = np.repeat(
        (np.eye(3) * core_radius**2)[None, :, :], count, axis=0
    )
    outer = minkowski_outer_shapes(core, leaf_u)
    proxies = SimpleNamespace(
        proxy_ids=np.arange(count, dtype=np.int64),
        centers=leaf_centers,
        sphere_radii=np.full(count, core_radius, dtype=float),
        base_sphere_radii=np.full(count, core_radius, dtype=float),
        base_ellipsoid_shapes=core,
        ellipsoid_outer_shapes=outer,
        proxy_uncertainty_shapes=leaf_u,
        proxy_offset_radii=leaf_delta,
    )

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
    errors = []
    compute_ms = []
    success_hold = 0
    first_hold = None
    statuses: dict[str, int] = {}
    try:
        for cycle in range(args.cycles):
            started = time.perf_counter()
            qdot, metrics = controller.solve(target)
            compute_ms.append((time.perf_counter() - started) * 1000.0)
            statuses[metrics.status] = statuses.get(metrics.status, 0) + 1
            q = q + np.asarray(qdot, dtype=float) * DT
            set_configuration(model, data, q)
            ee, _, _ = controller.task_feedback(target)
            error = float(np.linalg.norm(target - ee))
            errors.append(error)
            success_hold = success_hold + 1 if error < 0.001 else 0
            if success_hold >= 50 and first_hold is None:
                first_hold = cycle
    finally:
        mvt.close()

    result = {
        "diagnostic_only": True,
        "feeds_back_to_formal_run": False,
        "source_run": str(run.resolve()),
        "directional_leaf_proxy_count": count,
        "source_compact_proxy_count": int(summary["final_proxy_count"]),
        "initial_error_m": float(np.linalg.norm(target - initial_ee)),
        "minimum_error_m": float(np.min(errors)),
        "final_error_m": float(errors[-1]),
        "strict_less_than_1mm_hold50": first_hold is not None,
        "first_hold_completion_cycle": first_hold,
        "controller_compute_ms_p50": float(np.percentile(compute_ms, 50)),
        "controller_compute_ms_p95": float(np.percentile(compute_ms, 95)),
        "controller_compute_ms_p99": float(np.percentile(compute_ms, 99)),
        "controller_compute_ms_max": float(np.max(compute_ms)),
        "cycles_over_20ms": int(np.count_nonzero(np.asarray(compute_ms) > 20.0)),
        "statuses": statuses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
