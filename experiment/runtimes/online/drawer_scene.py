"""Mechanism scene for sphere-versus-ellipsoid certificate ablation."""

from model import BoxObstacle, SceneDefinition


def drawer_scene() -> SceneDefinition:
    center_y = 0.40633091
    boxes = (
        BoxObstacle(
            "drawer_left_rail",
            (0.20, center_y - 0.090, 0.401),
            (0.42, 0.015, 0.015),
            "shelf",
        ),
        BoxObstacle(
            "drawer_right_rail",
            (0.20, center_y + 0.090, 0.401),
            (0.42, 0.015, 0.015),
            "shelf",
        ),
    )
    return SceneDefinition(
        name="drawer",
        description=(
            "Thin long drawer rails with a 0.150 m exact opening and a "
            "sphere-certificate-valid initial state."
        ),
        boxes=boxes,
        q0=(-1.5708, -1.75, 1.95, -1.75, -1.5708, 0.0),
        waypoints=(
            (0.10, center_y, 0.401),
            (0.34, center_y, 0.401),
            (0.56, center_y, 0.401),
        ),
        duration=16.0,
    )
