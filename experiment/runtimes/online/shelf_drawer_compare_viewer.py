"""Interactive MuJoCo playback of the 150 mm drawer sphere/ellipsoid ablation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import queue
import time

import mujoco
import numpy as np

from model import build_robot_certificate, certificate_world_state
from pointcloud_proxy import build_shelf_drawer_proxy_set
from shelf_drawer_scene import shelf_drawer_narrow_scene


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_geometry_ablation"
DT = 0.02
IDENTITY = np.eye(3).reshape(-1)
OBSTACLE_SPHERE_RGBA = np.array([0.96, 0.24, 0.08, 0.17], dtype=np.float32)
OBSTACLE_ELLIPSOID_RGBA = np.array([0.98, 0.62, 0.05, 0.20], dtype=np.float32)
ROBOT_PROXY_RGBA = np.array([0.05, 0.78, 1.00, 0.20], dtype=np.float32)
EXACT_ROBOT_RGBA = np.array([0.20, 0.46, 0.82, 1.00], dtype=np.float32)
TARGET_RGBA = np.array([0.20, 0.95, 0.20, 0.85], dtype=np.float32)


@dataclass(frozen=True)
class RunSpec:
    key: str
    digit: str
    label: str
    directory: Path
    obstacle_geometry: str


RUN_SPECS = (
    RunSpec(
        "sphere",
        "1",
        "LiuQP spheres | 150 mm drawer | 60 s | blocked",
        RESULTS
        / "shelf_drawer_narrow_sphere_direct_pca_graph_mahal_cluster100mm_over025mm_dur060s",
        "sphere",
    ),
    RunSpec(
        "ellipsoid",
        "2",
        "LiuQP environment ellipsoids | same robot spheres | reaches goal",
        RESULTS
        / "shelf_drawer_narrow_ellipsoid_direct_pca_graph_mahal_cluster100mm_over025mm_dur030s",
        "ellipsoid",
    ),
)
SPEC_BY_KEY = {spec.key: spec for spec in RUN_SPECS}
SPEC_BY_DIGIT = {spec.digit: spec for spec in RUN_SPECS}


def load_rows(directory: Path) -> list[dict[str, str]]:
    path = directory / "trajectory.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing verified trajectory: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def set_row(model: mujoco.MjModel, data: mujoco.MjData, row: dict[str, str]) -> None:
    data.qpos[:6] = np.array([float(row[f"q{i}_rad"]) for i in range(1, 7)])
    data.qvel[:] = 0.0
    if model.nu:
        data.ctrl[:] = data.qpos[: model.nu]
    mujoco.mj_forward(model, data)


def shape_to_axes_rotation(shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, rotation = np.linalg.eigh(shape)
    order = np.argsort(values)
    rotation = rotation[:, order]
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return np.sqrt(np.maximum(values[order], 1.0e-18)), rotation


def append_geom(scene, geom_type, size, position, rotation, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError(f"Proxy overlay exceeds MjvScene maxgeom={scene.maxgeom}")
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


class DrawerComparison:
    def __init__(self) -> None:
        self.definition = shelf_drawer_narrow_scene()
        self.model = mujoco.MjModel.from_xml_path(str(RUN_SPECS[0].directory / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.rows = {spec.key: load_rows(spec.directory) for spec in RUN_SPECS}
        self.robot_spheres = build_robot_certificate(self.model)
        proxies = build_shelf_drawer_proxy_set(
            self.definition.boxes,
            cluster_size=0.100,
            maximum_aabb_overshoot=0.025,
        )
        self.obstacle_centers = proxies.centers
        self.obstacle_radii = proxies.sphere_radii
        self.obstacle_shapes = proxies.ellipsoid_shapes
        self.max_overshoot = proxies.maximum_ellipsoid_overshoot
        self.exact_robot_geom_ids = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) > 0
            and int(self.model.geom_type[geom_id])
            in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER))
        ]

    def populate(self, user_scene, spec, show_body, show_obstacles, show_robot):
        user_scene.ngeom = 0
        body_count = obstacle_count = robot_count = 0
        if show_body:
            for geom_id in self.exact_robot_geom_ids:
                append_geom(
                    user_scene,
                    self.model.geom_type[geom_id],
                    self.model.geom_size[geom_id],
                    self.data.geom_xpos[geom_id],
                    self.data.geom_xmat[geom_id],
                    EXACT_ROBOT_RGBA,
                )
                body_count += 1
        if show_obstacles:
            if spec.obstacle_geometry == "sphere":
                for center, radius in zip(self.obstacle_centers, self.obstacle_radii):
                    append_geom(
                        user_scene,
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.full(3, float(radius)),
                        center,
                        IDENTITY,
                        OBSTACLE_SPHERE_RGBA,
                    )
                    obstacle_count += 1
            else:
                for center, shape in zip(self.obstacle_centers, self.obstacle_shapes):
                    axes, rotation = shape_to_axes_rotation(shape)
                    append_geom(
                        user_scene,
                        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        axes,
                        center,
                        rotation,
                        OBSTACLE_ELLIPSOID_RGBA,
                    )
                    obstacle_count += 1
        if show_robot:
            positions, _, radii = certificate_world_state(
                self.model, self.data, self.robot_spheres
            )
            for position, radius in zip(positions, radii):
                append_geom(
                    user_scene,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.full(3, float(radius)),
                    position,
                    IDENTITY,
                    ROBOT_PROXY_RGBA,
                )
                robot_count += 1
        append_geom(
            user_scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, 0.018),
            np.asarray(self.definition.waypoints[-1]),
            IDENTITY,
            TARGET_RGBA,
        )
        return body_count, obstacle_count, robot_count

    def check(self) -> None:
        user_scene = mujoco.MjvScene(self.model, 5000)
        for spec in RUN_SPECS:
            rows = self.rows[spec.key]
            set_row(self.model, self.data, rows[len(rows) // 2])
            body, obstacles, robot = self.populate(user_scene, spec, True, True, True)
            assert obstacles == len(self.obstacle_centers) == 970
            assert robot == len(self.robot_spheres) == 68
            assert body == len(self.exact_robot_geom_ids)
            print(
                f"[{spec.digit}] {spec.key}: frames={len(rows)}, exact_body={body}, "
                f"obstacle_proxy={obstacles}, robot_spheres={robot}, overlay={user_scene.ngeom}"
            )
        print(f"maximum ellipsoid AABB overshoot={1000*self.max_overshoot:.2f} mm")

    def run(self, initial: str, speed: float) -> None:
        import mujoco.viewer

        commands: queue.SimpleQueue[int] = queue.SimpleQueue()

        def key_callback(keycode: int) -> None:
            commands.put(keycode)

        current = SPEC_BY_KEY[initial]
        frame = 0
        paused = False
        loop = True
        proxies_visible = True
        obstacle_proxies_enabled = True
        robot_proxies_enabled = True
        exact_body_visible = True
        step_once = False
        playback_speed = speed
        print("Keyboard shortcuts:")
        print("  1 / 2   = sphere LiuQP / environment-ellipsoid LiuQP")
        print("  V       = show/hide all collision proxies; exact body remains")
        print("  O / R   = toggle obstacle proxies / robot proxy spheres")
        print("  B       = toggle exact blue MuJoCo robot body")
        print("  P / N   = pause-play / one frame; 0 restart; L loop")
        print("  - / =   = slower / faster")

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=True,
        ) as viewer:
            # Hide duplicate model collision rendering.  Exact robot primitives
            # are redrawn independently, so V affects proxies only.
            viewer.opt.geomgroup[1] = 0
            viewer.cam.lookat[:] = np.array([0.50, 0.40, 0.48])
            viewer.cam.distance = 1.45
            viewer.cam.azimuth = 145.0
            viewer.cam.elevation = -20.0
            while viewer.is_running():
                while not commands.empty():
                    keycode = commands.get()
                    character = chr(keycode).upper() if 0 <= keycode < 128 else ""
                    if character in SPEC_BY_DIGIT:
                        current = SPEC_BY_DIGIT[character]
                        frame = 0
                        paused = False
                    elif character == "V":
                        proxies_visible = not proxies_visible
                    elif character == "O":
                        obstacle_proxies_enabled = not obstacle_proxies_enabled
                    elif character == "R":
                        robot_proxies_enabled = not robot_proxies_enabled
                    elif character == "B":
                        exact_body_visible = not exact_body_visible
                    elif character == "P":
                        paused = not paused
                    elif character == "N":
                        paused = True
                        step_once = True
                    elif character == "0":
                        frame = 0
                    elif character == "L":
                        loop = not loop
                    elif character in ("-", "_"):
                        playback_speed = max(0.125, playback_speed / 2.0)
                    elif character in ("=", "+"):
                        playback_speed = min(8.0, playback_speed * 2.0)

                rows = self.rows[current.key]
                frame = min(frame, len(rows) - 1)
                row = rows[frame]
                started = time.perf_counter()
                with viewer.lock():
                    set_row(self.model, self.data, row)
                    body, obstacles, robot = self.populate(
                        viewer.user_scn,
                        current,
                        exact_body_visible,
                        proxies_visible and obstacle_proxies_enabled,
                        proxies_visible and robot_proxies_enabled,
                    )
                viewer.set_texts(
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        current.label,
                        (
                            f"t={float(row['time_s']):.2f}s  "
                            f"goal error={1000*float(row['final_target_error_m']):.1f} mm  "
                            f"speed={playback_speed:g}x\n"
                            f"exact body={body}  obstacle proxy={obstacles}  "
                            f"robot spheres={robot}\n"
                            f"proxy fit=connected-normal PCA  max overshoot="
                            f"{1000*self.max_overshoot:.1f} mm\n"
                            "1 sphere | 2 ellipsoid | V proxies | O obstacles | "
                            "R robot balls | B body | P pause | N step | 0 restart"
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
                            frame = len(rows) - 1
                            paused = True
                remaining = DT / playback_speed - (time.perf_counter() - started)
                if remaining > 0.0:
                    time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--initial", choices=tuple(SPEC_BY_KEY), default="sphere")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    if args.speed <= 0.0:
        parser.error("--speed must be positive")
    comparison = DrawerComparison()
    if args.check:
        comparison.check()
    else:
        comparison.run(args.initial, args.speed)


if __name__ == "__main__":
    main()
