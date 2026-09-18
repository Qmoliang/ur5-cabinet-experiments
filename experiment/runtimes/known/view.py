"""Interactive MuJoCo replay of the known-volume sphere/ellipsoid experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "core"))
sys.path.insert(0, str(ROOT))

from robot import build_robot_certificate, certificate_world_positions, set_configuration
from run import DT, load_scene

IDENTITY = np.eye(3).reshape(-1)
SPHERE_RGBA = np.asarray([0.95, 0.12, 0.05, 0.18], dtype=np.float32)
ELLIPSOID_RGBA = np.asarray([1.0, 0.55, 0.02, 0.23], dtype=np.float32)
ROBOT_RGBA = np.asarray([0.05, 0.75, 1.0, 0.20], dtype=np.float32)
TARGET_RGBA = np.asarray([0.1, 1.0, 0.15, 0.9], dtype=np.float32)


def shape_to_axes_rotation(shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, rotation = np.linalg.eigh(np.asarray(shape, dtype=float))
    order = np.argsort(values)
    rotation = rotation[:, order]
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return np.sqrt(np.maximum(values[order], 1e-18)), rotation


def append_geom(scene, geom_type, size, position, rotation, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        int(geom_type),
        np.asarray(size, dtype=float),
        np.asarray(position, dtype=float),
        np.asarray(rotation, dtype=float).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1


def load_result(case: str) -> dict:
    directory = ROOT / "results" / case
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    proxies = np.load(directory / "proxies.npz")
    return {
        "summary": summary,
        "proxies": {name: np.asarray(proxies[name]) for name in proxies.files},
        "q": np.load(directory / "q_history.npy"),
        "ee": np.load(directory / "ee_history.npy"),
    }


def check() -> None:
    for case in ("sphere", "ellipsoid"):
        result = load_result(case)
        assert result["summary"]["complete_solid_cell_volume_covered"]
        assert len(result["proxies"]["centers"]) == 368
        assert len(result["q"]) == 3001
        assert len(result["ee"]) == 3001
    print("viewer inputs verified: 368 matched solid cells, 3000 cycles per representation")


def run_viewer(initial: str, speed: float) -> None:
    results = {case: load_result(case) for case in ("sphere", "ellipsoid")}
    scene = load_scene()
    model = __import__("robot").build_model(scene)
    data = mujoco.MjData(model)
    robot = build_robot_certificate(model)
    target = np.asarray(scene.waypoints[-1])
    commands: queue.Queue[str] = queue.Queue()

    def on_key(keycode: int) -> None:
        try:
            commands.put(chr(keycode).upper())
        except ValueError:
            return

    case = initial
    frame = 0
    paused = False
    show_proxies = True
    show_robot = True
    last_step = time.perf_counter()
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        print("1 球体积包络 | 2 椭球体积包络 | V代理 | R机械臂证书 | 空格暂停 | N单步 | 0开头 | F末态")
        while viewer.is_running():
            single_step = False
            while not commands.empty():
                command = commands.get_nowait()
                if command == "1":
                    case, frame = "sphere", 0
                elif command == "2":
                    case, frame = "ellipsoid", 0
                elif command == "V":
                    show_proxies = not show_proxies
                elif command == "R":
                    show_robot = not show_robot
                elif command == " ":
                    paused = not paused
                elif command == "N":
                    paused, single_step = True, True
                elif command == "0":
                    frame = 0
                elif command == "F":
                    frame = len(results[case]["q"]) - 1
            current = results[case]
            if not paused or single_step:
                now = time.perf_counter()
                if single_step or now - last_step >= DT / speed:
                    frame = min(frame + 1, len(current["q"]) - 1)
                    last_step = now
            set_configuration(model, data, current["q"][frame])
            overlay = viewer.user_scn
            overlay.ngeom = 0
            proxies = current["proxies"]
            if show_proxies:
                if case == "sphere":
                    for center, radius in zip(proxies["centers"], proxies["sphere_radii"]):
                        append_geom(
                            overlay,
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.full(3, radius),
                            center,
                            IDENTITY,
                            SPHERE_RGBA,
                        )
                else:
                    for center, shape in zip(proxies["centers"], proxies["ellipsoid_shapes"]):
                        axes, rotation = shape_to_axes_rotation(shape)
                        append_geom(
                            overlay,
                            mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                            axes,
                            center,
                            rotation,
                            ELLIPSOID_RGBA,
                        )
            if show_robot:
                positions = certificate_world_positions(data, robot)
                for center, certificate in zip(positions, robot):
                    append_geom(
                        overlay,
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.full(3, certificate.radius),
                        center,
                        IDENTITY,
                        ROBOT_RGBA,
                    )
            append_geom(
                overlay,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, 0.008),
                target,
                IDENTITY,
                TARGET_RGBA,
            )
            error_mm = 1000.0 * float(np.linalg.norm(target - current["ee"][frame]))
            label = "球体积外包" if case == "sphere" else "椭球体积外包"
            viewer.set_texts(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    label,
                    f"cycle={max(frame - 1, 0)}  error={error_mm:.4f} mm\n"
                    f"solid cells={len(proxies['centers'])}  full-volume coverage=PASS\n"
                    "1/2切换 | V体积代理 | R机械臂球 | 空格暂停 | F末态",
                )
            )
            viewer.sync()
            time.sleep(0.001)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", choices=("sphere", "ellipsoid"), default="ellipsoid")
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.speed <= 0.0:
        raise ValueError("speed must be positive")
    if args.check:
        check()
    else:
        run_viewer(args.initial, args.speed)


if __name__ == "__main__":
    main()
