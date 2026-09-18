"""Replay certified sphere/ellipsoid runs in the drawer and FastIRIS-style scenes.

The opaque geometry is the exact MuJoCo world.  Obstacle overlays include the
continuous surface-cover radius used by the QP, rather than showing only the
PCA cores and leaving visually misleading gaps.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import time

import mujoco
import numpy as np

from fastiris_style_scenes import FASTIRIS_STYLE_SCENES
from model import build_robot_certificate, certificate_world_state
from pointcloud_proxy import build_box_scene_proxy_set
from shelf_drawer_scene import shelf_drawer_certified_scene


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_geometry_ablation"
DT = 0.02
IDENTITY = np.eye(3).reshape(-1)
SPHERE_RGBA = np.array([0.96, 0.20, 0.05, 0.17], dtype=np.float32)
ELLIPSOID_CORE_RGBA = np.array([1.00, 0.55, 0.02, 0.20], dtype=np.float32)
ELLIPSOID_ENVELOPE_RGBA = np.array([1.00, 0.90, 0.15, 0.075], dtype=np.float32)
ROBOT_PROXY_RGBA = np.array([0.05, 0.78, 1.00, 0.20], dtype=np.float32)
EXACT_ROBOT_RGBA = np.array([0.20, 0.46, 0.82, 1.00], dtype=np.float32)
TARGET_RGBA = np.array([0.20, 0.95, 0.20, 0.85], dtype=np.float32)


@dataclass(frozen=True)
class SceneSpec:
    key: str
    definition: object
    sphere_dir: str
    ellipsoid_dir: str
    camera_lookat: tuple[float, float, float]
    camera_distance: float


SCENES = {
    "drawer": SceneSpec(
        "drawer",
        shelf_drawer_certified_scene(),
        "shelf_drawer_certified_sphere_direct_pca_graph_mahal_cluster100mm_over025mm_pc006mm_vox006mm_surfacecover_dur060s",
        "shelf_drawer_certified_ellipsoid_direct_pca_graph_mahal_cluster100mm_over025mm_pc006mm_vox006mm_surfacecover_dur030s",
        (0.50, 0.40, 0.48),
        1.45,
    ),
    "iiwa_shelf": SceneSpec(
        "iiwa_shelf",
        FASTIRIS_STYLE_SCENES["fastiris_iiwa_shelf_style"],
        "fastiris_iiwa_shelf_style_sphere_direct_pca_graph_mahal_cluster100mm_over025mm_pc006mm_vox006mm_surfacecover",
        "fastiris_iiwa_shelf_style_ellipsoid_direct_pca_graph_mahal_cluster100mm_over025mm_pc006mm_vox006mm_surfacecover",
        (0.48, 0.0, 0.62),
        1.65,
    ),
    "4_shelves": SceneSpec(
        "4_shelves",
        FASTIRIS_STYLE_SCENES["fastiris_4_shelves_style"],
        "fastiris_4_shelves_style_sphere_direct_pca_graph_mahal_cluster100mm_over025mm_pc006mm_vox006mm_surfacecover",
        "fastiris_4_shelves_style_ellipsoid_direct_pca_graph_mahal_cluster100mm_over025mm_pc006mm_vox006mm_surfacecover",
        (0.48, 0.0, 0.64),
        1.65,
    ),
    "iiwa_bins": SceneSpec(
        "iiwa_bins",
        FASTIRIS_STYLE_SCENES["fastiris_iiwa_bins_style"],
        "fastiris_iiwa_bins_style_sphere_direct_pca_graph_mahal_cluster100mm_over025mm_pc006mm_vox006mm_surfacecover",
        "fastiris_iiwa_bins_style_ellipsoid_direct_pca_graph_mahal_cluster100mm_over025mm_pc006mm_vox006mm_surfacecover",
        (0.49, 0.0, 0.56),
        1.60,
    ),
}


def _load_rows(directory: Path) -> list[dict[str, str]]:
    with (directory / "trajectory.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _load_summary(directory: Path) -> dict:
    return json.loads((directory / "summary.json").read_text(encoding="utf-8"))


def _set_row(model: mujoco.MjModel, data: mujoco.MjData, row: dict[str, str]) -> None:
    data.qpos[:6] = np.array([float(row[f"q{i}_rad"]) for i in range(1, 7)])
    data.qvel[:] = 0.0
    if model.nu:
        data.ctrl[:] = data.qpos[: model.nu]
    mujoco.mj_forward(model, data)


def _shape_to_axes_rotation(shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, rotation = np.linalg.eigh(shape)
    order = np.argsort(values)
    rotation = rotation[:, order]
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return np.sqrt(np.maximum(values[order], 1.0e-18)), rotation


def _outer_offset_shape(shape: np.ndarray, radius: float) -> np.ndarray:
    """Outer ellipsoid for E(Q) (+) B(radius), used only for rendering.

    Young's inequality gives
      E(Q) (+) E(r^2 I) subset
      E((1 + 1/beta) Q + (1 + beta) r^2 I).
    The controller does not use this approximation; it evaluates the exact
    support sqrt(n'Qn) + radius.
    """

    axes = np.sqrt(np.maximum(np.linalg.eigvalsh(shape), 1.0e-18))
    beta = max(float(np.exp(np.mean(np.log(axes))) / radius), 1.0e-6)
    return (1.0 + 1.0 / beta) * shape + (1.0 + beta) * radius**2 * np.eye(3)


def _append_geom(scene, geom_type, size, position, rotation, rgba) -> None:
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


class CertifiedComparison:
    def __init__(self, scene_spec: SceneSpec) -> None:
        self.scene_spec = scene_spec
        self.directories = {
            "sphere": RESULTS / scene_spec.sphere_dir,
            "ellipsoid": RESULTS / scene_spec.ellipsoid_dir,
        }
        self.model = mujoco.MjModel.from_xml_path(
            str(self.directories["sphere"] / "scene.xml")
        )
        self.data = mujoco.MjData(self.model)
        self.rows = {key: _load_rows(path) for key, path in self.directories.items()}
        self.summaries = {
            key: _load_summary(path) for key, path in self.directories.items()
        }
        self.robot_spheres = build_robot_certificate(self.model)
        proxies = build_box_scene_proxy_set(
            scene_spec.definition.boxes,
            point_spacing=0.006,
            filter_size=0.006,
            cluster_size=0.100,
            maximum_aabb_overshoot=0.025,
            validation_spacing=0.0015,
        )
        self.centers = proxies.centers
        self.radii = proxies.sphere_radii
        self.shapes = proxies.ellipsoid_shapes
        self.cover_radius = proxies.surface_cover_radius
        self.outer_shapes = np.asarray(
            [_outer_offset_shape(shape, self.cover_radius) for shape in self.shapes]
        )
        self.max_overshoot = proxies.maximum_ellipsoid_overshoot
        self.exact_robot_geom_ids = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) > 0
            and int(self.model.geom_type[geom_id])
            in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER))
        ]

    def populate(
        self,
        user_scene,
        mode: str,
        show_body: bool,
        show_obstacles: bool,
        show_robot: bool,
        show_core: bool,
        show_envelope: bool,
    ) -> tuple[int, int, int]:
        user_scene.ngeom = 0
        body_count = obstacle_count = robot_count = 0
        if show_body:
            for geom_id in self.exact_robot_geom_ids:
                _append_geom(
                    user_scene,
                    self.model.geom_type[geom_id],
                    self.model.geom_size[geom_id],
                    self.data.geom_xpos[geom_id],
                    self.data.geom_xmat[geom_id],
                    EXACT_ROBOT_RGBA,
                )
                body_count += 1
        if show_obstacles and mode == "sphere":
            for center, radius in zip(self.centers, self.radii):
                _append_geom(
                    user_scene,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.full(3, radius + self.cover_radius),
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
                        user_scene,
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
                        user_scene,
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
                    user_scene,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.full(3, radius),
                    position,
                    IDENTITY,
                    ROBOT_PROXY_RGBA,
                )
                robot_count += 1
        _append_geom(
            user_scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, 0.018),
            np.asarray(self.scene_spec.definition.waypoints[-1]),
            IDENTITY,
            TARGET_RGBA,
        )
        return body_count, obstacle_count, robot_count

    def check(self) -> None:
        user_scene = mujoco.MjvScene(self.model, 6000)
        for mode in ("sphere", "ellipsoid"):
            rows = self.rows[mode]
            _set_row(self.model, self.data, rows[len(rows) // 2])
            body, obstacles, robot = self.populate(
                user_scene, mode, True, True, True, True, True
            )
            expected_obstacles = len(self.centers) * (2 if mode == "ellipsoid" else 1)
            assert obstacles == expected_obstacles
            assert robot == len(self.robot_spheres)
            assert body == len(self.exact_robot_geom_ids)
            print(
                f"{self.scene_spec.key}/{mode}: frames={len(rows)}, "
                f"base_proxies={len(self.centers)}, drawn_obstacles={obstacles}, "
                f"robot_spheres={robot}, cover_delta={1000*self.cover_radius:.3f} mm"
            )

    def run(self, initial: str, speed: float) -> None:
        import mujoco.viewer

        commands: queue.SimpleQueue[int] = queue.SimpleQueue()
        current = initial
        frame = 0
        paused = False
        loop = True
        proxies_visible = True
        obstacle_visible = True
        robot_visible = True
        body_visible = True
        core_visible = True
        envelope_visible = True
        step_once = False
        playback_speed = speed

        print("1 sphere | 2 ellipsoid | V all proxies | O obstacles | R robot spheres")
        print("B exact robot | C PCA cores | E certified display envelope")
        print("P pause | N step | 0 restart | L loop | -/= speed")

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=commands.put,
            show_left_ui=False,
            show_right_ui=True,
        ) as viewer:
            viewer.opt.geomgroup[1] = 0
            viewer.cam.lookat[:] = np.asarray(self.scene_spec.camera_lookat)
            viewer.cam.distance = self.scene_spec.camera_distance
            viewer.cam.azimuth = 145.0
            viewer.cam.elevation = -20.0
            while viewer.is_running():
                while not commands.empty():
                    keycode = commands.get()
                    char = chr(keycode).upper() if 0 <= keycode < 128 else ""
                    if char == "1":
                        current, frame, paused = "sphere", 0, False
                    elif char == "2":
                        current, frame, paused = "ellipsoid", 0, False
                    elif char == "V":
                        proxies_visible = not proxies_visible
                    elif char == "O":
                        obstacle_visible = not obstacle_visible
                    elif char == "R":
                        robot_visible = not robot_visible
                    elif char == "B":
                        body_visible = not body_visible
                    elif char == "C":
                        core_visible = not core_visible
                    elif char == "E":
                        envelope_visible = not envelope_visible
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

                rows = self.rows[current]
                frame = min(frame, len(rows) - 1)
                row = rows[frame]
                started = time.perf_counter()
                with viewer.lock():
                    _set_row(self.model, self.data, row)
                    body, obstacles, robot = self.populate(
                        viewer.user_scn,
                        current,
                        body_visible,
                        proxies_visible and obstacle_visible,
                        proxies_visible and robot_visible,
                        core_visible,
                        envelope_visible,
                    )
                summary = self.summaries[current]
                state = "success" if summary["success"] else "stalled/failure"
                viewer.set_texts(
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        f"{self.scene_spec.key} | {current} | {state}",
                        (
                            f"t={float(row['time_s']):.2f}s  "
                            f"goal error={1000*float(row['final_target_error_m']):.1f} mm  "
                            f"speed={playback_speed:g}x\n"
                            f"surface cover delta={1000*self.cover_radius:.2f} mm; "
                            f"base proxies={len(self.centers)}\n"
                            f"body={body} obstacle overlays={obstacles} robot spheres={robot}\n"
                            "ellipsoid pale shell = conservative DISPLAY outer bound; "
                            "QP uses exact support sqrt(n'Qn)+delta\n"
                            "1 sphere | 2 ellipsoid | V proxies | O obstacles | R robot | "
                            "B body | C core | E envelope"
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
    parser.add_argument("--scene", choices=tuple(SCENES), default="drawer")
    parser.add_argument("--initial", choices=("sphere", "ellipsoid"), default="sphere")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.speed <= 0.0:
        parser.error("--speed must be positive")
    comparison = CertifiedComparison(SCENES[args.scene])
    if args.check:
        comparison.check()
    else:
        comparison.run(args.initial, args.speed)


if __name__ == "__main__":
    main()
