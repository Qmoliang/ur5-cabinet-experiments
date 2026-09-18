"""View the corrected adaptive ellipsoid v4.1 as a single formal replay."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from formal_protocol_v3_viewer import main as viewer_main


ROOT = Path(__file__).resolve().parent
NEW_RESULTS = (
    ROOT
    / "formal_results"
    / "final_two_camera"
    / "formal_strict_goal_v4_1_original_contact_semantics"
)


def _only_match(root: Path, representation: str) -> Path:
    matches = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and f"-{representation}-" in path.name
        and not any(token in path.name for token in ("-cr30mm", "-cr50mm", "-cr70mm"))
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one adaptive {representation} run in {root}, found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial", choices=("sphere", "ellipsoid"), default="ellipsoid"
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    sys.argv = [
        "formal_protocol_v3_viewer.py",
        "--sphere-dir",
        str(_only_match(NEW_RESULTS, "ellipsoid")),
        "--ellipsoid-dir",
        str(_only_match(NEW_RESULTS, "ellipsoid")),
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
