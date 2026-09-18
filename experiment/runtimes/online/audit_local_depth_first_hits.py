"""Audit the formal local depth frame without exposing geometry to control.

This simulation-only audit replays one recorded joint state and verifies that
every environment endpoint admitted to proxy construction is a first-visible
MuJoCo ray hit.  In the frozen drawer scene every obstacle starts at
``FRONT_X``; an admitted endpoint before that plane is therefore a ghost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from depth_camera_perception import UR5MountedDepthCamera, environment_endpoint_mask
from model import build_model, set_configuration
from protocol_drawer_scene import FRONT_X, formal_protocol_scene
from run_protocol_v3_async_online import CAMERA_NAMES
from run_protocol_v3_online_ablation import (
    CAMERA_PIXEL_STRIDE,
    PROXY_WORKSPACE_LOWER,
    PROXY_WORKSPACE_UPPER,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("--cycle", type=int, default=20)
    args = parser.parse_args()

    q_history = np.asarray(np.load(args.history), dtype=float)
    if not 0 <= args.cycle < len(q_history):
        raise IndexError(f"cycle {args.cycle} outside [0, {len(q_history)})")
    scene = formal_protocol_scene()
    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, q_history[args.cycle])

    started = time.perf_counter()
    with UR5MountedDepthCamera(
        model,
        camera_names=CAMERA_NAMES,
        width=160,
        height=120,
        pixel_stride=CAMERA_PIXEL_STRIDE,
        optical_depth_error_bound=0.003,
        occluding_self_filter=True,
        native_raycast=True,
    ) as camera:
        observations = camera.capture(data)
    capture_ms = (time.perf_counter() - started) * 1000.0

    rows = []
    ghost_count = 0
    admitted_count = 0
    for observation in observations:
        points = np.asarray(observation.points, dtype=float)
        keep = environment_endpoint_mask(observation)
        keep &= np.all(points >= PROXY_WORKSPACE_LOWER, axis=1)
        keep &= np.all(points <= PROXY_WORKSPACE_UPPER, axis=1)
        admitted = points[keep]
        ghosts = admitted[:, 0] < FRONT_X - 1e-10 if len(admitted) else np.zeros(0, bool)
        ghost_count += int(np.count_nonzero(ghosts))
        admitted_count += int(len(admitted))
        rows.append(
            {
                "camera": observation.camera_name,
                "all_first_hits": int(len(points)),
                "environment_workspace_hits": int(len(admitted)),
                "minimum_environment_x_m": (
                    None if not len(admitted) else float(np.min(admitted[:, 0]))
                ),
                "ghost_hits_before_front": int(np.count_nonzero(ghosts)),
            }
        )

    report = {
        "backend": "mujoco_mj_multiRay_first_visible",
        "history": str(args.history.resolve()),
        "cycle": int(args.cycle),
        "capture_ms": float(capture_ms),
        "drawer_front_x_m": float(FRONT_X),
        "environment_workspace_hits": admitted_count,
        "ghost_hits_before_front": ghost_count,
        "passed": bool(admitted_count > 0 and ghost_count == 0),
        "cameras": rows,
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
