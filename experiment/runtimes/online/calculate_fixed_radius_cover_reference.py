"""Analytic regular-grid reference for fixed-radius drawer certificates.

This is a geometry-only audit.  It does not run LiuQP, perception, or MuJoCo
control and must never be used to create online proxies.
"""

from __future__ import annotations

import argparse
import json
import math

from protocol_drawer_scene import formal_protocol_scene


def minimum_volume_grid(lengths_m: tuple[float, float, float], radius_m: float):
    """Minimize nx*ny*nz over axis-aligned equal cells covered by radius R."""

    if radius_m <= 0.0:
        raise ValueError("geometric radius must be positive")
    upper = tuple(max(1, math.ceil(length / (2.0 * radius_m))) + 40 for length in lengths_m)
    best = None
    for nx in range(1, upper[0] + 1):
        for ny in range(1, upper[1] + 1):
            for nz in range(1, upper[2] + 1):
                count = nx * ny * nz
                if best is not None and count > best[0]:
                    continue
                required = math.sqrt(
                    (lengths_m[0] / (2.0 * nx)) ** 2
                    + (lengths_m[1] / (2.0 * ny)) ** 2
                    + (lengths_m[2] / (2.0 * nz)) ** 2
                )
                candidate = (count, required, nx, ny, nz)
                if required <= radius_m + 1.0e-15 and (
                    best is None or candidate < best
                ):
                    best = candidate
    if best is None:
        raise AssertionError("grid search bound was insufficient")
    return best


def calculate(radius_mm: float, reserve_mm: float) -> dict[str, object]:
    radius_m = radius_mm / 1000.0
    reserve_m = reserve_mm / 1000.0
    geometric_radius = radius_m - reserve_m
    rows = []
    total = 0
    drawer_total = 0
    for index, box in enumerate(formal_protocol_scene().boxes):
        lengths = tuple(2.0 * float(value) for value in box.half_size)
        count, required, nx, ny, nz = minimum_volume_grid(
            lengths, geometric_radius
        )
        total += count
        if index < 5:
            drawer_total += count
        rows.append(
            {
                "obstacle": box.name,
                "size_mm": [round(value * 1000.0, 6) for value in lengths],
                "grid": [nx, ny, nz],
                "spheres": count,
                "required_geometric_radius_mm": required * 1000.0,
            }
        )
    return {
        "definition": "minimum axis-aligned equal-cell regular grid per analytic box",
        "fixed_sphere_radius_mm": radius_mm,
        "certificate_reserve_mm": reserve_mm,
        "available_geometric_radius_mm": geometric_radius * 1000.0,
        "drawer_five_components_spheres": drawer_total,
        "all_eight_components_spheres": total,
        "reject_above_two_times_reference": 2 * total,
        "components": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius-mm", type=float, default=70.0)
    parser.add_argument("--reserve-mm", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(calculate(args.radius_mm, args.reserve_mm), indent=2))


if __name__ == "__main__":
    main()
