"""Experiment 07 scene definitions.

The drawer keeps the frozen v4.3 interior and task geometry.  Only supports
below the existing cabinet are added so the furniture reaches the floor.  The
birdcage is a local IRIS-ZO-style generalization scene, not an official IRIS-ZO
benchmark asset.
"""

from __future__ import annotations

from model import BoxObstacle, SceneDefinition
from protocol_drawer_scene import (
    BACK_X,
    DRAWER_Y,
    DRAWER_Z,
    FRONT_X,
    INNER_HALF_Y,
    INNER_HALF_Z,
    Q0_V3,
    TARGET_X,
    WALL,
    formal_drawer_camera_v5_balanced_scene,
)


DRAWER_SCENE_NAME = "experiment_07_drawer"
BIRDCAGE_SCENE_NAME = "experiment_07_birdcage"


def experiment_07_drawer_scene() -> SceneDefinition:
    """Grounded cabinet with the exact frozen v4.3 drawer interior."""

    frozen = formal_drawer_camera_v5_balanced_scene()
    half_x = 0.5 * (BACK_X - FRONT_X)
    center_x = 0.5 * (BACK_X + FRONT_X)
    recolored = tuple(
        BoxObstacle(box.name, box.center, box.half_size, "experiment_07_cabinet")
        for box in frozen.boxes
    )
    # Existing cabinet side panels end at z=0.33 m.  These two supports extend
    # them to the ground without entering the drawer interior or open front.
    ground_supports = (
        BoxObstacle(
            "cabinet_ground_left",
            (BACK_X - 0.025, DRAWER_Y - 0.36, 0.165),
            (0.025, 0.025, 0.165),
            "experiment_07_cabinet",
        ),
        BoxObstacle(
            "cabinet_ground_right",
            (BACK_X - 0.025, DRAWER_Y + 0.36, 0.165),
            (0.025, 0.025, 0.165),
            "experiment_07_cabinet",
        ),
    )
    return SceneDefinition(
        name=DRAWER_SCENE_NAME,
        description=(
            "Experiment 07A grounded cabinet: exact v4.3 drawer interior, "
            "Q0 and deep final target, with two external supports to the floor."
        ),
        boxes=recolored + ground_supports,
        q0=frozen.q0,
        waypoints=frozen.waypoints,
        duration=frozen.duration,
    )


def _birdcage_boxes() -> tuple[BoxObstacle, ...]:
    """Large grounded bar cage with a 180 mm front entrance."""

    material = "experiment_07_cage"
    boxes: list[BoxObstacle] = [
        BoxObstacle("birdcage_base", (0.605, 0.18, 0.035), (0.255, 0.42, 0.035), material),
        BoxObstacle("birdcage_back_frame", (0.86, 0.18, 0.47), (0.018, 0.42, 0.40), material),
        BoxObstacle("birdcage_left_frame", (0.605, -0.24, 0.47), (0.255, 0.018, 0.40), material),
        BoxObstacle("birdcage_right_frame", (0.605, 0.60, 0.47), (0.255, 0.018, 0.40), material),
        BoxObstacle("birdcage_top_front", (0.35, 0.18, 0.87), (0.018, 0.42, 0.018), material),
        BoxObstacle("birdcage_top_back", (0.86, 0.18, 0.87), (0.018, 0.42, 0.018), material),
        BoxObstacle("birdcage_top_left", (0.605, -0.24, 0.87), (0.255, 0.018, 0.018), material),
        BoxObstacle("birdcage_top_right", (0.605, 0.60, 0.87), (0.255, 0.018, 0.018), material),
    ]
    # Front bars leave one 180 mm opening centered at y=0.18 m.  The target is
    # behind this opening.  Rear and side bars make the object visibly enclosed.
    front_y = (-0.22, -0.12, -0.02, 0.08, 0.28, 0.38, 0.48, 0.58)
    for index, y in enumerate(front_y):
        boxes.append(
            BoxObstacle(
                f"birdcage_front_bar_{index:02d}",
                (0.35, y, 0.47),
                (0.012, 0.012, 0.40),
                material,
            )
        )
    for index, y in enumerate((-0.20, -0.10, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)):
        boxes.append(
            BoxObstacle(
                f"birdcage_back_bar_{index:02d}",
                (0.842, y, 0.47),
                (0.012, 0.012, 0.40),
                material,
            )
        )
    for side_name, y in (("left", -0.222), ("right", 0.582)):
        for index, x in enumerate((0.43, 0.50, 0.57, 0.64, 0.71, 0.78)):
            boxes.append(
                BoxObstacle(
                    f"birdcage_{side_name}_bar_{index:02d}",
                    (x, y, 0.47),
                    (0.012, 0.012, 0.40),
                    material,
                )
            )
    # Stepped roof rails create a birdcage silhouette while retaining box-only
    # geometry supported by the current scene model.
    for index, (x, z, half_y) in enumerate(
        ((0.43, 0.93, 0.34), (0.50, 0.98, 0.27), (0.605, 1.02, 0.20), (0.71, 0.98, 0.27), (0.78, 0.93, 0.34))
    ):
        boxes.append(
            BoxObstacle(
                f"birdcage_roof_rail_{index:02d}",
                (x, 0.18, z),
                (0.018, half_y, 0.018),
                material,
            )
        )
    return tuple(boxes)


def experiment_07_birdcage_scene() -> SceneDefinition:
    """Stage-B1 final-goal-only pregrasp task through the front opening."""

    return SceneDefinition(
        name=BIRDCAGE_SCENE_NAME,
        description=(
            "Experiment 07B grounded giant birdcage: UR5 reaches through the "
            "front bar opening to a pregrasp target inside."
        ),
        boxes=_birdcage_boxes(),
        q0=Q0_V3,
        waypoints=((0.63, 0.18, 0.47),),
        duration=45.0,
    )


EXPERIMENT_07_SCENES = {
    "drawer": experiment_07_drawer_scene,
    "birdcage": experiment_07_birdcage_scene,
}




