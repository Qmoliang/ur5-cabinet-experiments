"""Render audited v5 MuJoCo trajectories with causal point/proxy overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import formal_protocol_v3_viewer as viewer_data
from formal_protocol_v3_viewer import (
    DEFAULT_ELLIPSOID,
    DEFAULT_SPHERE,
    FormalComparison,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "figures" / "formal_v5" / "videos"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(
    comparison: FormalComparison,
    mode: str,
    output: Path,
    *,
    speed: float,
    fps: int,
    width: int,
    height: int,
) -> dict:
    run = comparison.runs[mode]
    stride = max(1, int(round(50.0 * speed / fps)))
    cycles = list(range(0, len(run.cycles), stride))
    if cycles[-1] != len(run.cycles) - 1:
        cycles.append(len(run.cycles) - 1)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.42, 0.40, 0.52])
    camera.distance = 1.42
    camera.azimuth = 0.0
    camera.elevation = -15.0
    option = mujoco.MjvOption()
    option.geomgroup[1] = 0
    renderer = mujoco.Renderer(comparison.model, height=height, width=width)
    writer = imageio.get_writer(
        output,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    label = (
        "Sphere LiuQP: certificate-closed, failed"
        if mode == "sphere"
        else "Ellipsoid LiuQP: exact support-sum, reached"
    )
    try:
        for cycle in cycles:
            comparison.set_cycle(mode, cycle)
            renderer.update_scene(
                comparison.data, camera=camera, scene_option=option
            )
            counts = comparison.populate(
                renderer.scene,
                mode,
                cycle,
                clear_scene=False,
                show_points=True,
                show_obstacles=True,
                show_robot=True,
                show_core=True,
                show_outer=False,
                show_uncertainty=False,
                show_map=False,
                show_candidates=False,
                show_active=True,
            )
            frame = renderer.render()
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image, "RGBA")
            row = run.cycles[cycle]
            draw.rectangle((12, 12, width - 12, 86), fill=(0, 0, 0, 178))
            draw.text((24, 20), label, fill=(255, 255, 255, 255))
            draw.text(
                (24, 42),
                (
                    f"t={float(row['time_s']):5.2f}s  "
                    f"error={1000.0*float(row['error_m']):7.2f}mm  "
                    f"QP={row['status']}"
                ),
                fill=(255, 255, 255, 255),
            )
            draw.text(
                (24, 63),
                (
                    f"causal frame={counts['proxy_snapshot']}  "
                    f"points={counts['observed_point_count']}  "
                    f"proxies={counts['proxy_count']}  "
                    f"QP-active={counts['active_proxy_count']}"
                ),
                fill=(220, 245, 255, 255),
            )
            writer.append_data(np.asarray(image))
    finally:
        writer.close()
        renderer.close()
    return {
        "mode": mode,
        "output": str(output.resolve()),
        "sha256": sha256(output),
        "source_cycles": len(run.cycles),
        "rendered_frames": len(cycles),
        "fps": fps,
        "playback_speed": speed,
        "duration_s": len(cycles) / fps,
        "overlays": "causal observed points, all generated proxies, QP-active proxies, robot certificate spheres",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sphere-dir", type=Path, default=DEFAULT_SPHERE)
    parser.add_argument("--ellipsoid-dir", type=Path, default=DEFAULT_ELLIPSOID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--mode", choices=("sphere", "ellipsoid", "both"), default="both"
    )
    args = parser.parse_args()
    if args.speed <= 0.0 or args.fps <= 0:
        parser.error("speed and fps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    viewer_data.MAX_DRAWN_POINTS = 1200
    comparison = FormalComparison(args.sphere_dir, args.ellipsoid_dir)
    modes = ("sphere", "ellipsoid") if args.mode == "both" else (args.mode,)
    results = [
        render(
            comparison,
            mode,
            args.output / f"formal_v5_{mode}_causal_replay.mp4",
            speed=args.speed,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
        for mode in modes
    ]
    sources = [
        args.sphere_dir / "scene.xml",
        args.sphere_dir / "q_history.npy",
        args.sphere_dir / "causal_observability_snapshots.npz",
        args.sphere_dir / "causal_proxy_snapshots.npz",
        args.ellipsoid_dir / "q_history.npy",
        args.ellipsoid_dir / "causal_observability_snapshots.npz",
        args.ellipsoid_dir / "causal_proxy_snapshots.npz",
    ]
    manifest = {
        "real_or_mock": "real",
        "online_controller_rerun": False,
        "sources": {str(path.resolve()): sha256(path) for path in sources},
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
        "videos": results,
    }
    path = args.output / "video_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
