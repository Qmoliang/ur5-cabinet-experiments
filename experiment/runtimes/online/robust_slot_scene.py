"""Robust narrow-slot scene and matched anisotropic obstacle certificates.

The exact MuJoCo wall has a real vertical slot.  Both proxy trees use exactly
the same cell centers and cell partition.  A sphere encloses each rectangular
cell by its half diagonal, whereas the ellipsoid is the minimum-volume
axis-aligned ellipsoid enclosing the same cell.  Consequently the comparison
changes only the support geometry, not the obstacle mesh or sampling density.
"""

from __future__ import annotations

import numpy as np

from model import BoxObstacle, SceneDefinition


SLOT_CENTER_Y = 0.40633091
SLOT_WIDTH = 0.40
WALL_CENTER_X = 0.42
WALL_HALF_THICKNESS = 0.02
WORKSPACE_Y_MIN = -0.95
WORKSPACE_Y_MAX = 0.95
WORKSPACE_Z_MIN = 0.0
WORKSPACE_Z_MAX = 1.35

# Matched anisotropic cells.  Long z cells make the circumscribed balls
# strongly over-conservative in y, while ellipsoid y support stays small.
CELL_SIZE = np.array([0.04, 0.03, 0.36])


def robust_slot_scene() -> SceneDefinition:
    left_edge = SLOT_CENTER_Y - 0.5 * SLOT_WIDTH
    right_edge = SLOT_CENTER_Y + 0.5 * SLOT_WIDTH
    z_center = 0.5 * (WORKSPACE_Z_MIN + WORKSPACE_Z_MAX)
    z_half = 0.5 * (WORKSPACE_Z_MAX - WORKSPACE_Z_MIN)
    boxes = (
        BoxObstacle(
            "slot_wall_negative_y",
            (WALL_CENTER_X, 0.5 * (WORKSPACE_Y_MIN + left_edge), z_center),
            (
                WALL_HALF_THICKNESS,
                0.5 * (left_edge - WORKSPACE_Y_MIN),
                z_half,
            ),
            "obstacle",
        ),
        BoxObstacle(
            "slot_wall_positive_y",
            (WALL_CENTER_X, 0.5 * (right_edge + WORKSPACE_Y_MAX), z_center),
            (
                WALL_HALF_THICKNESS,
                0.5 * (WORKSPACE_Y_MAX - right_edge),
                z_half,
            ),
            "obstacle",
        ),
    )
    return SceneDefinition(
        name="robust_slot",
        description=(
            "Full-height through-wall with a 0.400 m exact vertical slot; "
            "matched anisotropic cells make the sphere proxy close the slot "
            "while the ellipsoid proxy preserves it."
        ),
        boxes=boxes,
        # Position-IK solution with the same start point and a compact wrist
        # cross-section; shared unchanged by both certificate geometries.
        q0=(
            -1.34973100,
            -1.45289466,
            1.66063433,
            -0.387646052,
            -2.86824511,
            0.0,
        ),
        waypoints=((0.20, SLOT_CENTER_Y, 0.401), (0.56, SLOT_CENTER_Y, 0.401)),
        duration=18.0,
    )


def _matched_cells(
    boxes: tuple[BoxObstacle, ...],
    cell_size: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    requested_cell_size = CELL_SIZE if cell_size is None else np.asarray(cell_size, dtype=float)
    centers: list[np.ndarray] = []
    steps: list[np.ndarray] = []
    owners: list[int] = []
    for owner, box in enumerate(boxes):
        half = np.asarray(box.half_size, dtype=float)
        count = np.maximum(1, np.ceil((2.0 * half) / requested_cell_size).astype(int))
        step = 2.0 * half / count
        axes = [
            np.linspace(-half[i] + 0.5 * step[i], half[i] - 0.5 * step[i], count[i])
            for i in range(3)
        ]
        for x in axes[0]:
            for y in axes[1]:
                for z in axes[2]:
                    centers.append(np.asarray(box.center) + np.array([x, y, z]))
                    steps.append(step.copy())
                    owners.append(owner)
    return np.asarray(centers), np.asarray(steps), np.asarray(owners, dtype=int)


def sample_matched_slot_spheres(
    boxes: tuple[BoxObstacle, ...],
    cell_size: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers, steps, owners = _matched_cells(boxes, cell_size)
    radii = 0.5 * np.linalg.norm(steps, axis=1)
    return centers, radii, owners


def sample_matched_slot_ellipsoids(
    boxes: tuple[BoxObstacle, ...],
    cell_size: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers, steps, owners = _matched_cells(boxes, cell_size)
    semi_axes = np.sqrt(3.0) * 0.5 * steps
    shapes = np.asarray([np.diag(axes**2) for axes in semi_axes])
    return centers, shapes, owners


def proxy_opening_certificate(robot_radius: float, safety_margin: float = 0.006) -> dict[str, float | bool]:
    """Closed-form transverse opening for the identical-cell comparison."""

    scene = robust_slot_scene()
    _, steps, _ = _matched_cells(scene.boxes)
    # Boundary cells on both wall halves have the same designed dimensions.
    step = steps[0]
    sphere_support_y = float(0.5 * np.linalg.norm(step))
    ellipsoid_support_y = float(np.sqrt(3.0) * 0.5 * step[1])
    sphere_opening = SLOT_WIDTH - 2.0 * (
        robot_radius + sphere_support_y + safety_margin
    )
    ellipsoid_opening = SLOT_WIDTH - 2.0 * (
        robot_radius + ellipsoid_support_y + safety_margin
    )
    return {
        "exact_slot_width_m": SLOT_WIDTH,
        "robot_radius_m": float(robot_radius),
        "safety_margin_m": float(safety_margin),
        "cell_step_x_m": float(step[0]),
        "cell_step_y_m": float(step[1]),
        "cell_step_z_m": float(step[2]),
        "sphere_obstacle_support_y_m": sphere_support_y,
        "ellipsoid_obstacle_support_y_m": ellipsoid_support_y,
        "sphere_modeled_opening_m": float(sphere_opening),
        "ellipsoid_modeled_opening_m": float(ellipsoid_opening),
        "sphere_model_is_closed": bool(sphere_opening <= 0.0),
        "ellipsoid_model_is_open": bool(ellipsoid_opening > 0.0),
    }
