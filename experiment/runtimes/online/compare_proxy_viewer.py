"""Interactive MuJoCo comparison viewer with collision-proxy overlays.

The exact scene is always visible. Keyboard shortcuts switch among the four
sphere-tree resolutions and the matched 204-obstacle-proxy full-ellipsoid run.
Obstacle and robot collision certificates can be shown or hidden independently.
No QP is re-solved: every motion comes from the verified stored trajectory CSV.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import queue
import time

import mujoco
import numpy as np

from ellipsoid_model import (
    build_robot_ellipsoid_certificate,
    ellipsoid_world_state,
)
from model import (
    build_robot_certificate,
    certificate_world_state,
)
from robust_slot_scene import (
    robust_slot_scene,
    sample_matched_slot_ellipsoids,
    sample_matched_slot_spheres,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_geometry_ablation"
DT = 0.02
IDENTITY = np.eye(3).reshape(-1)
OBSTACLE_RGBA = np.array([0.95, 0.25, 0.08, 0.14], dtype=np.float32)
ROBOT_RGBA = np.array([0.05, 0.75, 1.00, 0.18], dtype=np.float32)
ROBOT_ISOTROPIC_RGBA = np.array([0.78, 0.25, 0.95, 0.24], dtype=np.float32)
EXACT_ROBOT_RGBA = np.array([0.20, 0.46, 0.82, 1.00], dtype=np.float32)


@dataclass(frozen=True)
class RunSpec:
    key: str
    digit: str
    label: str
    directory: Path
    obstacle_geometry: str
    cell_size: tuple[float, float, float]
    expected_obstacle_proxies: int
    robot_geometry: str


RUN_SPECS = (
    RunSpec(
        "sphere204",
        "1",
        "Sphere LiuQP | 204 obstacle spheres | dz=0.3375 m | fail",
        RESULTS / "robust_slot_sphere_direct",
        "sphere",
        (0.04, 0.03, 0.36),
        204,
        "sphere",
    ),
    RunSpec(
        "sphere255",
        "2",
        "Sphere LiuQP | 255 obstacle spheres | dz=0.2700 m | fail",
        RESULTS
        / "resolution_sensitivity_sphere_z030_20260828"
        / "robust_slot_sphere_direct",
        "sphere",
        (0.04, 0.03, 0.30),
        255,
        "sphere",
    ),
    RunSpec(
        "sphere510",
        "3",
        "Sphere LiuQP | 510 obstacle spheres | dz=0.1350 m | fail",
        RESULTS
        / "resolution_sensitivity_sphere_z015_20260828"
        / "robust_slot_sphere_direct",
        "sphere",
        (0.04, 0.03, 0.15),
        510,
        "sphere",
    ),
    RunSpec(
        "sphere969",
        "4",
        "Sphere LiuQP | 969 obstacle spheres | dz=0.0711 m | success",
        RESULTS
        / "resolution_sensitivity_sphere_z0075_20260828"
        / "robust_slot_sphere_direct",
        "sphere",
        (0.04, 0.03, 0.075),
        969,
        "sphere",
    ),
    RunSpec(
        "ellipsoid204",
        "5",
        "Full-ellipsoid LiuQP | 204 obstacle ellipsoids | success",
        RESULTS / "robust_slot_ellipsoid_full_robot_direct",
        "ellipsoid",
        (0.04, 0.03, 0.36),
        204,
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
    values = values[order]
    rotation = rotation[:, order]
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return np.sqrt(np.maximum(values, 1.0e-18)), rotation


def append_geom(
    scene: mujoco.MjvScene,
    geom_type: int,
    size: np.ndarray,
    position: np.ndarray,
    rotation: np.ndarray,
    rgba: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError(f"Proxy overlay exceeds MjvScene maxgeom={scene.maxgeom}")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.asarray(size, dtype=float),
        np.asarray(position, dtype=float),
        np.asarray(rotation, dtype=float).reshape(-1),
        rgba,
    )
    geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1


class ProxyComparison:
    def __init__(self) -> None:
        self.scene_definition = robust_slot_scene()
        xml_path = RUN_SPECS[0].directory / "scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.rows = {spec.key: load_rows(spec.directory) for spec in RUN_SPECS}
        self.robot_spheres = build_robot_certificate(self.model)
        self.robot_ellipsoids = build_robot_ellipsoid_certificate(
            self.model, axial_overlap_factor=2.0
        )
        self.exact_robot_geom_ids = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) > 0
            and int(self.model.geom_type[geom_id])
            in (
                int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            )
        ]
        self.obstacles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for spec in RUN_SPECS:
            cell_size = np.asarray(spec.cell_size, dtype=float)
            if spec.obstacle_geometry == "sphere":
                centers, radii, _ = sample_matched_slot_spheres(
                    self.scene_definition.boxes, cell_size
                )
                parameters = radii
            else:
                centers, shapes, _ = sample_matched_slot_ellipsoids(
                    self.scene_definition.boxes, cell_size
                )
                parameters = shapes
            if len(centers) != spec.expected_obstacle_proxies:
                raise AssertionError(
                    f"{spec.key}: expected {spec.expected_obstacle_proxies} obstacle "
                    f"proxies, got {len(centers)}"
                )
            self.obstacles[spec.key] = (centers, parameters)

    def populate_proxy_scene(
        self,
        user_scene: mujoco.MjvScene,
        spec: RunSpec,
        show_exact_robot: bool,
        show_obstacles: bool,
        show_robot: bool,
    ) -> tuple[int, int, int, int, int]:
        user_scene.ngeom = 0
        body_count = obstacle_count = robot_count = 0
        anisotropic_count = isotropic_count = 0
        if show_exact_robot:
            for geom_id in self.exact_robot_geom_ids:
                append_geom(
                    user_scene,
                    int(self.model.geom_type[geom_id]),
                    self.model.geom_size[geom_id],
                    self.data.geom_xpos[geom_id],
                    self.data.geom_xmat[geom_id],
                    EXACT_ROBOT_RGBA,
                )
                body_count += 1
        centers, parameters = self.obstacles[spec.key]
        if show_obstacles:
            if spec.obstacle_geometry == "sphere":
                for center, radius in zip(centers, parameters):
                    append_geom(
                        user_scene,
                        int(mujoco.mjtGeom.mjGEOM_SPHERE),
                        np.full(3, float(radius)),
                        center,
                        IDENTITY,
                        OBSTACLE_RGBA,
                    )
                    obstacle_count += 1
            else:
                for center, shape in zip(centers, parameters):
                    axes, rotation = shape_to_axes_rotation(shape)
                    append_geom(
                        user_scene,
                        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
                        axes,
                        center,
                        rotation,
                        OBSTACLE_RGBA,
                    )
                    obstacle_count += 1
        if show_robot:
            if spec.robot_geometry == "sphere":
                positions, _, radii = certificate_world_state(
                    self.model, self.data, self.robot_spheres
                )
                for position, radius in zip(positions, radii):
                    append_geom(
                        user_scene,
                        int(mujoco.mjtGeom.mjGEOM_SPHERE),
                        np.full(3, float(radius)),
                        position,
                        IDENTITY,
                        ROBOT_RGBA,
                    )
                    robot_count += 1
            else:
                positions, _, _, shapes = ellipsoid_world_state(
                    self.model, self.data, self.robot_ellipsoids
                )
                for position, shape in zip(positions, shapes):
                    axes, rotation = shape_to_axes_rotation(shape)
                    isotropic = bool(np.ptp(axes) <= 1.0e-8)
                    append_geom(
                        user_scene,
                        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
                        axes,
                        position,
                        rotation,
                        ROBOT_ISOTROPIC_RGBA if isotropic else ROBOT_RGBA,
                    )
                    robot_count += 1
                    if isotropic:
                        isotropic_count += 1
                    else:
                        anisotropic_count += 1
        return (
            body_count,
            obstacle_count,
            robot_count,
            anisotropic_count,
            isotropic_count,
        )

    def check(self) -> None:
        user_scene = mujoco.MjvScene(self.model, 5000)
        for spec in RUN_SPECS:
            rows = self.rows[spec.key]
            set_row(self.model, self.data, rows[len(rows) // 2])
            body_count, obstacle_count, robot_count, anisotropic, isotropic = (
                self.populate_proxy_scene(user_scene, spec, True, True, True)
            )
            expected_robot = 57 if spec.robot_geometry == "sphere" else 73
            assert obstacle_count == spec.expected_obstacle_proxies
            assert robot_count == expected_robot
            assert body_count == len(self.exact_robot_geom_ids)
            assert user_scene.ngeom == body_count + obstacle_count + robot_count
            print(
                f"[{spec.digit}] {spec.key}: frames={len(rows)}, "
                f"body={body_count}, obstacle={obstacle_count}, "
                f"robot={robot_count} (aniso={anisotropic}, iso={isotropic}), "
                f"overlay={user_scene.ngeom}"
            )

    def run(self, initial: str = "sphere204", speed: float = 1.0) -> None:
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
        show_exact_robot = True
        step_once = False
        playback_speed = speed

        print("Keyboard shortcuts:")
        print("  1/2/3/4 = 204/255/510/969 obstacle-sphere LiuQP")
        print("  5       = 204-obstacle-ellipsoid full-ellipsoid LiuQP")
        print("  V       = show/hide all proxies (exact robot remains visible)")
        print("  O / R   = toggle obstacle / robot proxies")
        print("  B       = toggle the exact MuJoCo robot collision body")
        print("  P       = pause/play, N = one frame, 0 = restart, L = loop")
        print("  - / =   = slower/faster")

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=True,
        ) as viewer:
            # The official mesh assets are intentionally absent from this compact
            # reproduction. Hide the model's duplicate group-1 collision geoms and
            # draw the same exact primitives explicitly in user_scn so V can never
            # accidentally hide the robot body together with its certificates.
            viewer.opt.geomgroup[1] = 0
            viewer.cam.lookat[:] = np.array([0.27, 0.40, 0.55])
            viewer.cam.distance = 1.75
            viewer.cam.azimuth = 145.0
            viewer.cam.elevation = -22.0
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
                        show_exact_robot = not show_exact_robot
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
                    (
                        body_count,
                        obstacle_count,
                        robot_count,
                        anisotropic_count,
                        isotropic_count,
                    ) = self.populate_proxy_scene(
                        viewer.user_scn,
                        current,
                        show_exact_robot,
                        proxies_visible and obstacle_proxies_enabled,
                        proxies_visible and robot_proxies_enabled,
                    )
                error_mm = 1000.0 * float(row["final_target_error_m"])
                viewer.set_texts(
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        current.label,
                        (
                            f"t={float(row['time_s']):.2f}s  error={error_mm:.1f}mm  "
                            f"speed={playback_speed:g}x\n"
                            f"exact body={body_count}  obstacle proxy={obstacle_count}  "
                            f"robot proxy={robot_count}\n"
                            f"robot ellipsoids: anisotropic={anisotropic_count} "
                            f"isotropic/end-cap={isotropic_count}  "
                            f"paused={paused} loop={loop}\n"
                            "1-4 spheres | 5 ellipsoids | V all | O obstacle | "
                            "R robot | B body | P pause | N step | 0 restart"
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
    parser.add_argument("--check", action="store_true", help="validate all trajectories and overlays without opening a window")
    parser.add_argument("--initial", choices=tuple(SPEC_BY_KEY), default="sphere204")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    if args.speed <= 0.0:
        parser.error("--speed must be positive")
    comparison = ProxyComparison()
    if args.check:
        comparison.check()
    else:
        comparison.run(args.initial, args.speed)


if __name__ == "__main__":
    main()
