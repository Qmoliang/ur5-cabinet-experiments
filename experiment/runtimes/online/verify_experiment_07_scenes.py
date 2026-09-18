"""Static E7-D1 checks for Experiment 07 scene inheritance and reachability."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from audit_protocol_v3_static_geometry import (
    _collision_free,
    _contact_records,
    _find_contact_free_ik,
    _solve_position_ik,
)
from experiment_07_scenes import (
    experiment_07_birdcage_scene,
    experiment_07_drawer_scene,
)
from model import JOINT_NAMES, build_model, build_xml
from protocol_drawer_scene import formal_drawer_camera_v5_balanced_scene


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "formal_results" / "experiment_07" / "static_scene_gate"
KNOWN_V43_SUCCESS_Q = np.asarray(
    [-5.43462758222244, -2.15925369343159, -0.8008267965607978,
     -1.8212272577837971, 0.005535997518201547, -0.28719821275751417],
    dtype=float,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def box_geometry(box):
    return np.asarray((*box.center, *box.half_size), dtype=float)


def scene_report(scene, rng, known_seed=None):
    xml = build_xml(scene).encode("utf-8")
    model = build_model(scene)
    data = mujoco.MjData(model)
    lower = np.asarray([model.jnt_range[model.joint(name).id, 0] for name in JOINT_NAMES])
    upper = np.asarray([model.jnt_range[model.joint(name).id, 1] for name in JOINT_NAMES])
    target = np.asarray(scene.waypoints[-1], dtype=float)
    ik = []
    if known_seed is not None:
        q, error = _solve_position_ik(model, data, known_seed, target, lower, upper)
        if error <= 0.0015 and _collision_free(model, data, q):
            ik.append({
                "attempt": "historical_v4_3_success_q_seed",
                "error_m": error,
                "q": q.tolist(),
                "contacts": _contact_records(model, data),
            })
    if not ik:
        ik = _find_contact_free_ik(
            model, data, target, lower, upper, rng, attempts=160, maximum_records=3
        )
    minimum_box_z = min(box.center[2] - box.half_size[2] for box in scene.boxes)
    return {
        "scene": scene.name,
        "xml_sha256": sha256(xml),
        "box_count": len(scene.boxes),
        "minimum_box_z_m": float(minimum_box_z),
        "touches_ground": bool(abs(minimum_box_z) <= 1.0e-12),
        "target_m": list(scene.waypoints[-1]),
        "contact_free_ik_count": len(ik),
        "contact_free_ik": ik,
    }, xml


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = formal_drawer_camera_v5_balanced_scene()
    drawer = experiment_07_drawer_scene()
    birdcage = experiment_07_birdcage_scene()
    inherited_equal = len(drawer.boxes) == len(frozen.boxes) + 2 and all(
        np.array_equal(box_geometry(old), box_geometry(new))
        for old, new in zip(frozen.boxes, drawer.boxes[: len(frozen.boxes)])
    )
    inherited_names_equal = [box.name for box in frozen.boxes] == [
        box.name for box in drawer.boxes[: len(frozen.boxes)]
    ]
    target_equal = drawer.waypoints == frozen.waypoints
    q0_equal = drawer.q0 == frozen.q0

    rng = np.random.default_rng(7007)
    drawer_report, drawer_xml = scene_report(drawer, rng, KNOWN_V43_SUCCESS_Q)
    cage_report, cage_xml = scene_report(birdcage, rng)
    report = {
        "experiment": "07",
        "gate": "E7-D1",
        "mock_data": False,
        "drawer_interior_geometry_equal_to_v43": bool(inherited_equal),
        "drawer_inherited_names_equal": bool(inherited_names_equal),
        "drawer_target_equal": bool(target_equal),
        "drawer_q0_equal": bool(q0_equal),
        "drawer": drawer_report,
        "birdcage": cage_report,
    }
    report["passed"] = bool(
        inherited_equal
        and inherited_names_equal
        and target_equal
        and q0_equal
        and drawer_report["touches_ground"]
        and cage_report["touches_ground"]
        and drawer_report["contact_free_ik_count"] > 0
        and cage_report["contact_free_ik_count"] > 0
    )
    (OUTPUT / "experiment_07_drawer.xml").write_bytes(drawer_xml)
    (OUTPUT / "experiment_07_birdcage.xml").write_bytes(cage_xml)
    (OUTPUT / "scene_gate.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
