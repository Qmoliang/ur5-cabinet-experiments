"""Open-drawer shelf scene for point-cloud sphere/ellipsoid ablation."""

from model import BoxObstacle, SceneDefinition


CENTER_Y = 0.40633091
CENTER_Z = 0.401


def _make_shelf_drawer_scene(inner_half: float, name: str) -> SceneDefinition:
    """A long open drawer inside a shelf, with a target near its back.

    The drawer is already open and static.  The task is pure collision-free
    insertion, not contact-rich drawer opening or grasp dynamics.  Its exact
    interior is 180 mm wide and 180 mm high.  A slender 350 mm retrieval-tool
    extension enters about 320 mm while the bulky wrist stays outside.
    """

    wall_half = 0.015
    drawer_x = 0.615
    drawer_half_x = 0.185  # front x=0.430, back x=0.800
    boxes = (
        # Pulled-out drawer: bottom and two long side walls.
        BoxObstacle(
            "open_drawer_bottom",
            (drawer_x, CENTER_Y, CENTER_Z - inner_half - wall_half),
            (drawer_half_x, 0.120, wall_half),
            "shelf",
        ),
        BoxObstacle(
            "open_drawer_left_wall",
            (drawer_x, CENTER_Y - inner_half - wall_half, CENTER_Z),
            (drawer_half_x, wall_half, inner_half),
            "shelf",
        ),
        BoxObstacle(
            "open_drawer_right_wall",
            (drawer_x, CENTER_Y + inner_half + wall_half, CENTER_Z),
            (drawer_half_x, wall_half, inner_half),
            "shelf",
        ),
        BoxObstacle(
            "open_drawer_back",
            (0.815, CENTER_Y, CENTER_Z),
            (wall_half, 0.120, inner_half + wall_half),
            "shelf",
        ),
        # The shelf board immediately above the open drawer forms the fourth
        # side of the long insertion channel.
        BoxObstacle(
            "shelf_board_above_drawer",
            (drawer_x, CENTER_Y, CENTER_Z + inner_half + wall_half),
            (drawer_half_x, 0.380, wall_half),
            "shelf",
        ),
        # Cabinet frame.  These pieces make the scene visibly a shelf rather
        # than two abstract rails, but do not encode any proxy orientation.
        BoxObstacle(
            "shelf_left_outer",
            (drawer_x, CENTER_Y - 0.430, 0.585),
            (drawer_half_x, 0.020, 0.330),
            "shelf",
        ),
        BoxObstacle(
            "shelf_right_outer",
            (drawer_x, CENTER_Y + 0.430, 0.585),
            (drawer_half_x, 0.020, 0.330),
            "shelf",
        ),
        BoxObstacle(
            "shelf_top",
            (drawer_x, CENTER_Y, 0.915),
            (drawer_half_x, 0.450, 0.020),
            "shelf",
        ),
        BoxObstacle(
            "shelf_back_panel",
            (0.815, CENTER_Y, 0.585),
            (0.020, 0.450, 0.350),
            "shelf",
        ),
        # Handle sits below the channel so it is visible but is not an
        # artificial obstacle directly in front of the opening.
        BoxObstacle(
            "open_drawer_handle",
            (0.385, CENTER_Y, CENTER_Z - 0.130),
            (0.012, 0.075, 0.012),
            "obstacle",
        ),
    )
    return SceneDefinition(
        name=name,
        description=(
            "IRIS-shelf-inspired cabinet with a static open drawer; the UR5e "
            f"must insert a slender retrieval tool through a {int(2000 * inner_half)} mm "
            f"x {int(2000 * inner_half)} mm channel "
            "to a target near the drawer back."
        ),
        boxes=boxes,
        # Exact-collision-free IK with tool tip at x=0.34 and tool axis along
        # drawer +x.  Both controllers start from this identical pose.
        q0=(-1.63037709, -1.53134835, 1.75428579, -0.22293743, -3.20117342, 1.57079633),
        waypoints=(
            (0.34, CENTER_Y, CENTER_Z),
            (0.43, CENTER_Y, CENTER_Z),
            (0.72, CENTER_Y, CENTER_Z),
        ),
        duration=20.0,
    )


def shelf_drawer_scene() -> SceneDefinition:
    """The original 180 mm open-drawer development scene."""

    return _make_shelf_drawer_scene(0.090, "shelf_drawer")


def shelf_drawer_narrow_scene() -> SceneDefinition:
    """A 150 mm channel used for the proxy-closure stress test."""

    return _make_shelf_drawer_scene(0.075, "shelf_drawer_narrow")


def shelf_drawer_certified_scene() -> SceneDefinition:
    """A 190 mm channel reserved for continuous-surface-certified runs."""

    return _make_shelf_drawer_scene(0.095, "shelf_drawer_certified")
