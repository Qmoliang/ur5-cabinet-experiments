"""Render final sphere and ellipsoid states for visual delivery QA."""
from __future__ import annotations

from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "core"))

from robot import build_model, build_robot_certificate, certificate_world_positions, set_configuration
from run import load_scene
from view import (
    ELLIPSOID_RGBA,
    IDENTITY,
    ROBOT_RGBA,
    SPHERE_RGBA,
    TARGET_RGBA,
    append_geom,
    load_result,
    shape_to_axes_rotation,
)


def main() -> None:
    scene = load_scene()
    model = build_model(scene)
    data = mujoco.MjData(model)
    robot = build_robot_certificate(model)
    target = np.asarray(scene.waypoints[-1])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray([0.52, 0.40633091, 0.60])
    camera.distance = 1.25
    camera.azimuth = 135.0
    camera.elevation = -18.0
    renderer = mujoco.Renderer(model, height=720, width=960)
    try:
        for case in ("sphere", "ellipsoid"):
            result = load_result(case)
            set_configuration(model, data, result["q"][-1])
            renderer.update_scene(data, camera=camera)
            proxies = result["proxies"]
            if case == "sphere":
                for center, radius in zip(proxies["centers"], proxies["sphere_radii"]):
                    append_geom(
                        renderer.scene,
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
                        renderer.scene,
                        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        axes,
                        center,
                        rotation,
                        ELLIPSOID_RGBA,
                    )
            for center, certificate in zip(certificate_world_positions(data, robot), robot):
                append_geom(
                    renderer.scene,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.full(3, certificate.radius),
                    center,
                    IDENTITY,
                    ROBOT_RGBA,
                )
            append_geom(
                renderer.scene,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, 0.008),
                target,
                IDENTITY,
                TARGET_RGBA,
            )
            Image.fromarray(renderer.render()).save(ROOT / "results" / f"preview_{case}.png")
    finally:
        renderer.close()
    print("rendered final-state previews for sphere and ellipsoid")


if __name__ == "__main__":
    main()
