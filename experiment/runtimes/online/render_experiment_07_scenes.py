"""Render the two Experiment 07 E7-D1 scene-review images."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from experiment_07_scenes import experiment_07_birdcage_scene, experiment_07_drawer_scene
from model import build_model, set_configuration


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figures" / "experiment_07"


def render(scene, filename, lookat, distance, azimuth, elevation):
    model = build_model(scene)
    data = mujoco.MjData(model)
    set_configuration(model, data, np.asarray(scene.q0, dtype=float))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    renderer = mujoco.Renderer(model, height=720, width=960)
    try:
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
    finally:
        renderer.close()
    Image.fromarray(image).save(OUTPUT / filename)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    render(
        experiment_07_drawer_scene(),
        "F11a_grounded_drawer_scene.png",
        (0.45, 0.36, 0.55),
        2.0,
        145.0,
        -20.0,
    )
    render(
        experiment_07_birdcage_scene(),
        "F11b_grounded_birdcage_scene.png",
        (0.48, 0.18, 0.52),
        2.15,
        150.0,
        -18.0,
    )


if __name__ == "__main__":
    main()
