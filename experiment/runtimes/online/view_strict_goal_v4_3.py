"""Open the final v4.3 no-core-reinflation comparison in MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from formal_protocol_v3_viewer import main as viewer_main


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "formal_results" / "final_two_camera"
SUFFIX = (
    "camera_quarter_cv7.5mm_cr70mm_u1p05_stride1_vmax180mmps_"
    "affinity_perception_heavy_formal_redesign_no_coreinflate_r70_paired"
)
SPHERE_ROOT = (
    RESULTS / f"formal_strict_goal_v4_3_surface_core_sphere_{SUFFIX}"
)
ELLIPSOID_ROOT = (
    RESULTS / f"formal_strict_goal_v4_3_surface_core_ellipsoid_{SUFFIX}"
)


def _single_run(root: Path) -> Path:
    matches = [path for path in root.iterdir() if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one run below {root}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial", choices=("sphere", "ellipsoid"), default="sphere"
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    sys.argv = [
        "formal_protocol_v3_viewer.py",
        "--sphere-dir",
        str(_single_run(SPHERE_ROOT)),
        "--ellipsoid-dir",
        str(_single_run(ELLIPSOID_ROOT)),
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
