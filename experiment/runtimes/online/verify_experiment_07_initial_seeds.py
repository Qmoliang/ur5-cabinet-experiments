"""Audit the five frozen Experiment 07 initial configurations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from audit_protocol_v3_static_geometry import _collision_free, _contact_records
from model import build_model, set_configuration
from run_experiment_07 import FORMAL_INITIAL_OFFSETS_RAD, SCENES
from run_protocol_v3_async_online import _protocol_scene


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "formal_results" / "experiment_07" / "initial_seed_gate.json"


def main() -> None:
    scene_reports = []
    for short_name, scene_version in SCENES.items():
        scene = _protocol_scene(scene_version)
        model = build_model(scene)
        data = mujoco.MjData(model)
        q0 = np.asarray(scene.q0, dtype=float)
        seeds = []
        for seed, offset in enumerate(FORMAL_INITIAL_OFFSETS_RAD):
            q = q0 + offset
            free = _collision_free(model, data, q)
            set_configuration(model, data, q)
            seeds.append(
                {
                    "seed": seed,
                    "offset_rad": offset.tolist(),
                    "q_rad": q.tolist(),
                    "q_sha256": hashlib.sha256(
                        np.ascontiguousarray(q, dtype=np.float64).tobytes()
                    ).hexdigest(),
                    "collision_free": bool(free),
                    "contacts": _contact_records(model, data),
                }
            )
        scene_reports.append(
            {
                "scene": short_name,
                "scene_version": scene_version,
                "all_collision_free": all(row["collision_free"] for row in seeds),
                "seeds": seeds,
            }
        )
    report = {
        "experiment": "07",
        "gate": "formal-initial-seeds",
        "mock_data": False,
        "seed_count": len(FORMAL_INITIAL_OFFSETS_RAD),
        "scenes": scene_reports,
    }
    report["passed"] = all(row["all_collision_free"] for row in scene_reports)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

