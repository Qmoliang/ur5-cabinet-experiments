"""Open the final v4.2h sphere/ellipsoid causal comparison in MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from formal_protocol_v3_viewer import main as viewer_main


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "formal_results" / "final_two_camera"
SPHERE_ROOT = (
    RESULTS
    / "formal_strict_goal_v4_2h_matched_sphere_camera_quarter_"
    "affinity_isolated_12_2_18_formal_paired"
)
ELLIPSOID_ROOT = (
    RESULTS
    / "formal_strict_goal_v4_2h_reachable_partitioned_thin_camera_quarter_"
    "affinity_isolated_12_2_18_formal_paired"
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
