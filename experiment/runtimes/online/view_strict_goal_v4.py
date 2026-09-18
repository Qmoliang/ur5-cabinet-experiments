"""Open the strict 1 mm adaptive/30/50/70 mm comparison in MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from formal_protocol_v3_viewer import main as viewer_main


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "formal_results" / "final_two_camera" / "formal_strict_goal_v4"


def _run_directory(representation: str, scale: str) -> Path:
    scale_token = "" if scale == "adaptive" else f"-cr{int(scale)}mm"
    matches = sorted(
        path
        for path in RESULTS.iterdir()
        if path.is_dir()
        and f"-{representation}-" in path.name
        and scale_token in path.name
        and (
            scale != "adaptive"
            or not any(token in path.name for token in ("-cr30mm", "-cr50mm", "-cr70mm"))
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {representation} {scale} run, found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale", choices=("adaptive", "30", "50", "70"), default="adaptive"
    )
    parser.add_argument(
        "--initial", choices=("sphere", "ellipsoid"), default="ellipsoid"
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    sphere = _run_directory("sphere", args.scale)
    ellipsoid = _run_directory("ellipsoid", args.scale)
    sys.argv = [
        "formal_protocol_v3_viewer.py",
        "--sphere-dir",
        str(sphere),
        "--ellipsoid-dir",
        str(ellipsoid),
        "--initial",
        args.initial,
        "--speed",
        str(args.speed),
    ]
    if args.check:
        sys.argv.append("--check")
    viewer_main()


if __name__ == "__main__":
    main()
