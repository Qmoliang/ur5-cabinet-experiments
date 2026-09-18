"""Post-hoc exact MuJoCo contact audit for a saved joint trajectory."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import mujoco
import numpy as np

from model import build_model, set_configuration
from protocol_drawer_scene import (
    protocol_drawer_v6_scene,
    protocol_drawer_v9_scene,
    protocol_drawer_v10_scene,
    protocol_drawer_v11_scene,
    protocol_drawer_v6s2_scene,
    protocol_drawer_v6s3_scene,
    protocol_drawer_v6s4_scene,
)


SCENES = {
    "v6": protocol_drawer_v6_scene,
    "v9": protocol_drawer_v9_scene,
    "v10": protocol_drawer_v10_scene,
    "v11": protocol_drawer_v11_scene,
    "v6s2": protocol_drawer_v6s2_scene,
    "v6s3": protocol_drawer_v6s3_scene,
    "v6s4": protocol_drawer_v6s4_scene,
}


def _geom_label(model: mujoco.MjModel, geom_id: int) -> str:
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    body_id = int(model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return f"{geom_name or f'geom_{geom_id}'}@{body_name or f'body_{body_id}'}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--scene-version", choices=tuple(SCENES), required=True)
    args = parser.parse_args()

    scene = SCENES[args.scene_version]()
    model = build_model(scene)
    data = mujoco.MjData(model)
    q_history = np.load(args.run_dir / "q_history.npy")
    q_trajectory = np.vstack((np.asarray(scene.q0, dtype=float), q_history))
    pairs: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "penetrating_cycles": set(),
            "contacts": 0,
            "minimum_distance_m": 0.0,
            "first_cycle": None,
        }
    )
    penetrating_cycles: set[int] = set()
    for cycle, q in enumerate(q_trajectory):
        set_configuration(model, data, q)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if float(contact.dist) >= -1.0e-8:
                continue
            penetrating_cycles.add(cycle)
            labels = tuple(
                sorted(
                    (
                        _geom_label(model, int(contact.geom1)),
                        _geom_label(model, int(contact.geom2)),
                    )
                )
            )
            row = pairs[labels]
            row["penetrating_cycles"].add(cycle)
            row["contacts"] += 1
            row["minimum_distance_m"] = min(
                row["minimum_distance_m"], float(contact.dist)
            )
            if row["first_cycle"] is None:
                row["first_cycle"] = cycle
    pair_rows = []
    for labels, row in pairs.items():
        pair_rows.append(
            {
                "geom_pair": list(labels),
                "penetrating_cycles": len(row["penetrating_cycles"]),
                "contacts": row["contacts"],
                "minimum_distance_m": row["minimum_distance_m"],
                "first_cycle": row["first_cycle"],
            }
        )
    pair_rows.sort(key=lambda row: (row["first_cycle"], row["geom_pair"]))
    report = {
        "scene": scene.name,
        "trajectory_states": len(q_trajectory),
        "penetrating_cycles": len(penetrating_cycles),
        "first_penetrating_cycle": (
            None if not penetrating_cycles else min(penetrating_cycles)
        ),
        "pairs": pair_rows,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
