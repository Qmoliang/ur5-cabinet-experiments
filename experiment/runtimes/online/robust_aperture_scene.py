"""Through-plate scene with a provably sphere-closed transverse slot."""

from __future__ import annotations

import numpy as np

from model import BoxObstacle, SceneDefinition


APERTURE_CENTER_X = -0.13399814
APERTURE_CENTER_Y = 0.40633091
APERTURE_WIDTH_Y = 0.35
PLATE_CENTER_Z = 0.65
PLATE_HALF_THICKNESS = 0.02
WORKSPACE_X_MIN = -0.95
WORKSPACE_X_MAX = 0.95
WORKSPACE_Y_MIN = -0.95
WORKSPACE_Y_MAX = 0.95
CELL_SIZE = np.array([0.30, 0.03, 0.04])


def robust_aperture_scene() -> SceneDefinition:
    lower_edge = APERTURE_CENTER_Y - 0.5 * APERTURE_WIDTH_Y
    upper_edge = APERTURE_CENTER_Y + 0.5 * APERTURE_WIDTH_Y
    x_center = 0.5 * (WORKSPACE_X_MIN + WORKSPACE_X_MAX)
    x_half = 0.5 * (WORKSPACE_X_MAX - WORKSPACE_X_MIN)
    boxes = (
        BoxObstacle(
            "plate_negative_y",
            (x_center, 0.5 * (WORKSPACE_Y_MIN + lower_edge), PLATE_CENTER_Z),
            (x_half, 0.5 * (lower_edge - WORKSPACE_Y_MIN), PLATE_HALF_THICKNESS),
            "obstacle",
        ),
        BoxObstacle(
            "plate_positive_y",
            (x_center, 0.5 * (upper_edge + WORKSPACE_Y_MAX), PLATE_CENTER_Z),
            (x_half, 0.5 * (WORKSPACE_Y_MAX - upper_edge), PLATE_HALF_THICKNESS),
            "obstacle",
        ),
    )
    return SceneDefinition(
        name="robust_aperture",
        description=(
            "Full-width horizontal plate with a 0.350 m exact slot; matched "
            "anisotropic cells close the sphere model and preserve the "
            "ellipsoid model while the UR5 tool moves along its wrist axis."
        ),
        boxes=boxes,
        q0=(-1.5708, -1.75, 1.95, -1.75, -1.5708, 0.0),
        waypoints=(
            (APERTURE_CENTER_X, APERTURE_CENTER_Y, 0.52),
            (APERTURE_CENTER_X, APERTURE_CENTER_Y, 0.82),
        ),
        duration=18.0,
    )


def _matched_cells(boxes: tuple[BoxObstacle, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers: list[np.ndarray] = []
    steps: list[np.ndarray] = []
    owners: list[int] = []
    for owner, box in enumerate(boxes):
        half = np.asarray(box.half_size, dtype=float)
        count = np.maximum(1, np.ceil((2.0 * half) / CELL_SIZE).astype(int))
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


def sample_matched_aperture_spheres(boxes: tuple[BoxObstacle, ...]):
    centers, steps, owners = _matched_cells(boxes)
    return centers, 0.5 * np.linalg.norm(steps, axis=1), owners


def sample_matched_aperture_ellipsoids(boxes: tuple[BoxObstacle, ...]):
    centers, steps, owners = _matched_cells(boxes)
    semi_axes = np.sqrt(3.0) * 0.5 * steps
    return centers, np.asarray([np.diag(axes**2) for axes in semi_axes]), owners


def aperture_opening_certificate(robot_radius: float, safety_margin: float = 0.006):
    scene = robust_aperture_scene()
    _, steps, _ = _matched_cells(scene.boxes)
    step = steps[0]
    sphere_support_y = float(0.5 * np.linalg.norm(step))
    ellipsoid_support_y = float(np.sqrt(3.0) * 0.5 * step[1])
    sphere_opening = APERTURE_WIDTH_Y - 2.0 * (robot_radius + sphere_support_y + safety_margin)
    ellipsoid_opening = APERTURE_WIDTH_Y - 2.0 * (robot_radius + ellipsoid_support_y + safety_margin)
    return {
        "exact_slot_width_m": APERTURE_WIDTH_Y,
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
