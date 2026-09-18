"""UR5e adaptations of three static shelf/bin environment classes in FastIRIS.

The FastIRIS paper names IIWAShelf, 4Shelves, and IIWABins but does not publish
MuJoCo assets or dimensions.  These scenes are therefore transparent geometric
reconstructions of those environment classes, not exact benchmark replicas.
"""

from __future__ import annotations

from model import BoxObstacle, SceneDefinition


Q0 = (-1.5708, -1.75, 1.95, -1.75, -1.5708, 0.0)


def fastiris_iiwa_shelf_style_scene() -> SceneDefinition:
    boxes = (
        BoxObstacle("shelf_back", (0.72, 0.00, 0.60), (0.025, 0.48, 0.46), "shelf"),
        BoxObstacle("shelf_left", (0.55, 0.50, 0.60), (0.19, 0.025, 0.46), "shelf"),
        BoxObstacle("shelf_right", (0.55, -0.50, 0.60), (0.19, 0.025, 0.46), "shelf"),
        BoxObstacle("shelf_bottom", (0.55, 0.00, 0.14), (0.19, 0.50, 0.025), "shelf"),
        BoxObstacle("shelf_middle", (0.55, 0.00, 0.56), (0.19, 0.50, 0.025), "shelf"),
        BoxObstacle("shelf_top", (0.55, 0.00, 1.06), (0.19, 0.50, 0.025), "shelf"),
        BoxObstacle("upper_divider", (0.55, 0.00, 0.81), (0.19, 0.025, 0.225), "shelf"),
    )
    return SceneDefinition(
        name="fastiris_iiwa_shelf_style",
        description="UR5e reconstruction of the FastIRIS IIWAShelf environment class.",
        boxes=boxes,
        q0=Q0,
        waypoints=((0.25, -0.24, 0.78), (0.43, -0.24, 0.78), (0.52, -0.24, 0.78)),
        duration=24.0,
    )


def fastiris_4_shelves_style_scene() -> SceneDefinition:
    boxes = (
        BoxObstacle("four_back", (0.72, 0.00, 0.63), (0.025, 0.48, 0.43), "shelf"),
        BoxObstacle("four_left", (0.55, 0.50, 0.63), (0.19, 0.025, 0.43), "shelf"),
        BoxObstacle("four_right", (0.55, -0.50, 0.63), (0.19, 0.025, 0.43), "shelf"),
        BoxObstacle("four_bottom", (0.55, 0.00, 0.20), (0.19, 0.50, 0.025), "shelf"),
        BoxObstacle("four_middle", (0.55, 0.00, 0.63), (0.19, 0.50, 0.025), "shelf"),
        BoxObstacle("four_top", (0.55, 0.00, 1.06), (0.19, 0.50, 0.025), "shelf"),
        BoxObstacle("four_vertical", (0.55, 0.00, 0.63), (0.19, 0.025, 0.43), "shelf"),
    )
    return SceneDefinition(
        name="fastiris_4_shelves_style",
        description="UR5e 2x2-cubby reconstruction of the FastIRIS 4Shelves class.",
        boxes=boxes,
        q0=Q0,
        waypoints=((0.24, -0.24, 0.84), (0.42, -0.24, 0.84), (0.53, -0.24, 0.84)),
        duration=24.0,
    )


def fastiris_iiwa_bins_style_scene() -> SceneDefinition:
    boxes = (
        BoxObstacle("bins_back", (0.73, 0.00, 0.56), (0.025, 0.48, 0.31), "shelf"),
        BoxObstacle("bins_left", (0.57, 0.50, 0.56), (0.18, 0.025, 0.31), "shelf"),
        BoxObstacle("bins_right", (0.57, -0.50, 0.56), (0.18, 0.025, 0.31), "shelf"),
        BoxObstacle("bins_bottom", (0.57, 0.00, 0.25), (0.18, 0.50, 0.025), "shelf"),
        BoxObstacle("bins_top", (0.57, 0.00, 0.87), (0.18, 0.50, 0.025), "shelf"),
        BoxObstacle("bins_divider_left", (0.57, -0.165, 0.56), (0.18, 0.018, 0.31), "shelf"),
        BoxObstacle("bins_divider_right", (0.57, 0.165, 0.56), (0.18, 0.018, 0.31), "shelf"),
    )
    return SceneDefinition(
        name="fastiris_iiwa_bins_style",
        description="UR5e three-bin reconstruction of the FastIRIS IIWABins class.",
        boxes=boxes,
        q0=Q0,
        waypoints=((0.24, 0.00, 0.56), (0.42, 0.00, 0.56), (0.54, 0.00, 0.56)),
        duration=24.0,
    )


FASTIRIS_STYLE_SCENES = {
    "fastiris_iiwa_shelf_style": fastiris_iiwa_shelf_style_scene(),
    "fastiris_4_shelves_style": fastiris_4_shelves_style_scene(),
    "fastiris_iiwa_bins_style": fastiris_iiwa_bins_style_scene(),
}
