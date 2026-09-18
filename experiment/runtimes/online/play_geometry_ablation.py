"""Replay or render the verified robust-slot trajectories.

Examples
--------
Interactive MuJoCo playback::

    python play_geometry_ablation.py --method sphere --mode viewer
    python play_geometry_ablation.py --method ellipsoid --mode viewer

Render MP4 files from the stored CSV trajectories::

    python play_geometry_ablation.py --method both --mode video
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_geometry_ablation"
RUNS = {
    "sphere": RESULTS / "robust_slot_sphere_direct",
    "ellipsoid": RESULTS / "robust_slot_ellipsoid_full_robot_direct",
}
LABELS = {
    "sphere": "Sphere LiuQP — fixed 204-obstacle-sphere level",
    "ellipsoid": "Full-ellipsoid LiuQP — reaches target",
}
DT = 0.02


def load_rows(directory: Path) -> list[dict[str, str]]:
    with (directory / "trajectory.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def set_row(model: mujoco.MjModel, data: mujoco.MjData, row: dict[str, str]) -> None:
    data.qpos[:6] = np.array([float(row[f"q{i}_rad"]) for i in range(1, 7)])
    data.qvel[:] = 0.0
    if model.nu:
        data.ctrl[:] = data.qpos[: model.nu]
    mujoco.mj_forward(model, data)


def play_viewer(method: str, speed: float) -> None:
    import mujoco.viewer

    directory = RUNS[method]
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    data = mujoco.MjData(model)
    rows = load_rows(directory)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for row in rows:
            if not viewer.is_running():
                break
            started = time.perf_counter()
            set_row(model, data, row)
            viewer.sync()
            remaining = DT / speed - (time.perf_counter() - started)
            if remaining > 0.0:
                time.sleep(remaining)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


def render_video(method: str, output_dir: Path, speed: float) -> Path:
    directory = RUNS[method]
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    data = mujoco.MjData(model)
    rows = load_rows(directory)
    width, height = 960, 720
    source_fps = 1.0 / DT
    output_fps = 25
    stride = max(1, int(round(source_fps / (output_fps * speed))))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.27, 0.40, 0.55])
    camera.distance = 1.75
    camera.azimuth = 145.0
    camera.elevation = -22.0
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"robust_slot_{method}_replay.mp4"
    renderer = mujoco.Renderer(model, height=height, width=width)
    writer = imageio.get_writer(
        output,
        fps=output_fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    try:
        for row in rows[::stride]:
            set_row(model, data, row)
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            error_mm = 1000.0 * float(row["final_target_error_m"])
            time_s = float(row["time_s"])
            draw.rectangle((12, 12, 590, 70), fill=(0, 0, 0, 165))
            draw.text((24, 22), LABELS[method], fill=(255, 255, 255))
            draw.text(
                (24, 44),
                f"t = {time_s:5.2f} s    goal error = {error_mm:7.2f} mm",
                fill=(255, 255, 255),
            )
            writer.append_data(np.asarray(image))
    finally:
        writer.close()
        renderer.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("sphere", "ellipsoid", "both"), default="both")
    parser.add_argument("--mode", choices=("viewer", "video"), default="viewer")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures" / "geometry_ablation" / "videos",
    )
    args = parser.parse_args()
    if args.speed <= 0.0:
        parser.error("--speed must be positive")
    methods = ("sphere", "ellipsoid") if args.method == "both" else (args.method,)
    for method in methods:
        if args.mode == "viewer":
            play_viewer(method, args.speed)
        else:
            print(render_video(method, args.output_dir, args.speed))


if __name__ == "__main__":
    main()
