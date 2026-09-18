"""Interactive MuJoCo replay of the strict wrist-depth sphere/ellipsoid result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import queue
import time

import mujoco
import numpy as np

from certified_scene_compare_viewer import (
    ELLIPSOID_CORE_RGBA,
    ELLIPSOID_ENVELOPE_RGBA,
    IDENTITY,
    ROBOT_PROXY_RGBA,
    SPHERE_RGBA,
    TARGET_RGBA,
    _append_geom,
    _outer_offset_shape,
    _set_row,
    _shape_to_axes_rotation,
)
from model import build_robot_certificate, certificate_world_state


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_geometry_ablation"
PROXY_FILE = ROOT / "results_depth_camera" / "wrist_depth_proxy_set_cluster140.npz"
DT = 0.02
DEPTH_RGBA = np.array([0.10, 0.90, 0.95, 0.28], dtype=np.float32)
CAMERA_RGBA = np.array([0.75, 0.20, 1.00, 0.90], dtype=np.float32)
BARRIER_RGBA = np.array([0.95, 0.05, 0.60, 0.13], dtype=np.float32)


def _find_result(certificate: str, duration: int) -> Path:
    matches = sorted(
        RESULTS.glob(
            "shelf_drawer_strict_long_tool_"
            f"{certificate}_direct_*wrist_depth_frozen_c140_strict*"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    matches = [
        path
        for path in matches
        if json.loads((path / "summary.json").read_text(encoding="utf-8"))[
            "configured_duration_s"
        ]
        == duration
    ]
    if not matches:
        raise FileNotFoundError(f"strict {certificate} duration={duration}s result not found")
    return matches[0]


def _rows(path: Path) -> list[dict[str, str]]:
    with (path / "trajectory.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class StrictDepthComparison:
    def __init__(self) -> None:
        self.directories = {
            "sphere": _find_result("sphere", 60),
            "ellipsoid": _find_result("ellipsoid", 40),
        }
        self.model = mujoco.MjModel.from_xml_path(
            str(self.directories["sphere"] / "scene.xml")
        )
        self.data = mujoco.MjData(self.model)
        self.rows = {mode: _rows(path) for mode, path in self.directories.items()}
        self.summaries = {
            mode: json.loads((path / "summary.json").read_text(encoding="utf-8"))
            for mode, path in self.directories.items()
        }
        payload = np.load(PROXY_FILE)
        self.centers = payload["centers"]
        self.radii = payload["sphere_radii"]
        self.shapes = payload["ellipsoid_shapes"]
        self.offsets = payload["proxy_offset_radii"]
        self.depth_points = payload["filtered_points"][:: max(1, len(payload["filtered_points"]) // 1200)]
        self.camera_positions = payload["camera_positions"]
        self.outer_shapes = np.asarray(
            [
                _outer_offset_shape(shape, float(offset))
                for shape, offset in zip(self.shapes, self.offsets)
            ]
        )
        self.robot_spheres = build_robot_certificate(self.model)
        self.target = np.array([0.730, 0.40633091, 0.401])

    def populate(
        self,
        scene: mujoco.MjvScene,
        mode: str,
        show_obstacles: bool,
        show_robot: bool,
        show_core: bool,
        show_envelope: bool,
        show_depth: bool,
        show_cameras: bool,
        show_barrier: bool,
    ) -> tuple[int, int, int]:
        scene.ngeom = 0
        obstacle_count = robot_count = depth_count = 0
        if show_obstacles and mode == "sphere":
            for center, radius, offset in zip(self.centers, self.radii, self.offsets):
                _append_geom(
                    scene,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.full(3, radius + offset),
                    center,
                    IDENTITY,
                    SPHERE_RGBA,
                )
                obstacle_count += 1
        elif show_obstacles:
            if show_envelope:
                for center, shape in zip(self.centers, self.outer_shapes):
                    axes, rotation = _shape_to_axes_rotation(shape)
                    _append_geom(
                        scene,
                        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        axes,
                        center,
                        rotation,
                        ELLIPSOID_ENVELOPE_RGBA,
                    )
                    obstacle_count += 1
            if show_core:
                for center, shape in zip(self.centers, self.shapes):
                    axes, rotation = _shape_to_axes_rotation(shape)
                    _append_geom(
                        scene,
                        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        axes,
                        center,
                        rotation,
                        ELLIPSOID_CORE_RGBA,
                    )
                    obstacle_count += 1
        if show_robot:
            positions, _, radii = certificate_world_state(
                self.model, self.data, self.robot_spheres
            )
            for position, radius in zip(positions, radii):
                _append_geom(
                    scene,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.full(3, radius),
                    position,
                    IDENTITY,
                    ROBOT_PROXY_RGBA,
                )
                robot_count += 1
        if show_depth:
            for point in self.depth_points:
                _append_geom(
                    scene,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.full(3, 0.0018),
                    point,
                    IDENTITY,
                    DEPTH_RGBA,
                )
                depth_count += 1
        if show_cameras:
            for point in self.camera_positions:
                _append_geom(
                    scene,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.full(3, 0.008),
                    point,
                    IDENTITY,
                    CAMERA_RGBA,
                )
        if show_barrier:
            _append_geom(
                scene,
                mujoco.mjtGeom.mjGEOM_BOX,
                np.array([0.001, 0.077, 0.077]),
                np.array([0.718, self.target[1], self.target[2]]),
                IDENTITY,
                BARRIER_RGBA,
            )
        _append_geom(
            scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, 0.010),
            self.target,
            IDENTITY,
            TARGET_RGBA,
        )
        return obstacle_count, robot_count, depth_count

    def check(self) -> None:
        user_scene = mujoco.MjvScene(self.model, 9000)
        for mode in ("sphere", "ellipsoid"):
            _set_row(self.model, self.data, self.rows[mode][len(self.rows[mode]) // 2])
            counts = self.populate(
                user_scene, mode, True, True, True, True, True, True, True
            )
            print(
                mode,
                "frames=", len(self.rows[mode]),
                "matched_proxies=", len(self.centers),
                "robot_spheres=", len(self.robot_spheres),
                "drawn=", counts,
            )

    def run(self, initial: str, speed: float) -> None:
        import mujoco.viewer

        commands: queue.SimpleQueue[int] = queue.SimpleQueue()
        mode = initial
        frame = 0
        paused = False
        loop = True
        show_proxies = True
        show_obstacles = True
        show_robot = True
        show_core = True
        show_envelope = True
        show_depth = False
        show_cameras = False
        show_barrier = True
        step_once = False
        playback_speed = speed
        print("1 sphere | 2 ellipsoid | V all proxies | O obstacle proxies | R robot spheres")
        print("B exact robot body | C ellipsoid core | E uncertainty envelope")
        print("D CenterVox depth points | K scan camera positions | H closed-section plane")
        print("P pause | N one frame | 0 restart | L loop | -/= speed")
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=commands.put,
            show_left_ui=False,
            show_right_ui=True,
        ) as viewer:
            viewer.opt.geomgroup[1] = 1
            viewer.cam.lookat[:] = np.array([0.53, 0.405, 0.43])
            viewer.cam.distance = 1.28
            viewer.cam.azimuth = 145.0
            viewer.cam.elevation = -18.0
            while viewer.is_running():
                while not commands.empty():
                    code = commands.get()
                    char = chr(code).upper() if 0 <= code < 128 else ""
                    if char == "1":
                        mode, frame, paused = "sphere", 0, False
                    elif char == "2":
                        mode, frame, paused = "ellipsoid", 0, False
                    elif char == "V":
                        show_proxies = not show_proxies
                    elif char == "O":
                        show_obstacles = not show_obstacles
                    elif char == "R":
                        show_robot = not show_robot
                    elif char == "B":
                        viewer.opt.geomgroup[1] = 1 - int(viewer.opt.geomgroup[1])
                    elif char == "C":
                        show_core = not show_core
                    elif char == "E":
                        show_envelope = not show_envelope
                    elif char == "D":
                        show_depth = not show_depth
                    elif char == "K":
                        show_cameras = not show_cameras
                    elif char == "H":
                        show_barrier = not show_barrier
                    elif char == "P":
                        paused = not paused
                    elif char == "N":
                        paused, step_once = True, True
                    elif char == "0":
                        frame = 0
                    elif char == "L":
                        loop = not loop
                    elif char in ("-", "_"):
                        playback_speed = max(0.125, playback_speed / 2.0)
                    elif char in ("=", "+"):
                        playback_speed = min(8.0, playback_speed * 2.0)
                rows = self.rows[mode]
                frame = min(frame, len(rows) - 1)
                row = rows[frame]
                started = time.perf_counter()
                with viewer.lock():
                    _set_row(self.model, self.data, row)
                    obstacles, robot, depth = self.populate(
                        viewer.user_scn,
                        mode,
                        show_proxies and show_obstacles,
                        show_proxies and show_robot,
                        show_core,
                        show_envelope,
                        show_depth,
                        show_cameras,
                        show_barrier,
                    )
                summary = self.summaries[mode]
                state = "SUCCESS" if summary["success"] else "STALLED / IMPOSSIBLE"
                viewer.set_texts(
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        f"strict wrist-depth | {mode} | {state}",
                        (
                            f"t={float(row['time_s']):.2f}s  goal error="
                            f"{1000*float(row['final_target_error_m']):.2f} mm  "
                            f"gate=5 mm  speed={playback_speed:g}x\n"
                            f"604 matched obstacle clusters; robot certificate={len(self.robot_spheres)} spheres\n"
                            f"overlays: obstacles={obstacles} robot={robot} depth={depth}; exact body group="
                            f"{int(viewer.opt.geomgroup[1])}\n"
                            "pink plane x=0.718 m: sphere continuous-clearance upper bound = -1.339 mm\n"
                            "V toggles proxies only; B toggles exact UR5/tool body independently"
                        ),
                    )
                )
                viewer.sync()
                if not paused or step_once:
                    frame += 1
                    step_once = False
                    if frame >= len(rows):
                        if loop:
                            frame = 0
                        else:
                            frame, paused = len(rows) - 1, True
                remaining = DT / playback_speed - (time.perf_counter() - started)
                if remaining > 0.0:
                    time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", choices=("sphere", "ellipsoid"), default="sphere")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    comparison = StrictDepthComparison()
    if args.check:
        comparison.check()
    else:
        comparison.run(args.initial, args.speed)


if __name__ == "__main__":
    main()
