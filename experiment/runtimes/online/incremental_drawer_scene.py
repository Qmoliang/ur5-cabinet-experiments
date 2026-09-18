"""Frozen no-long-tool drawer scene for the incremental-depth experiment."""

from model import BoxObstacle, SceneDefinition


DRAWER_Y = 0.40633091
DRAWER_Z = 0.580
INNER_HALF_Y = 0.090
INNER_HALF_Z = 0.090
PROTOCOL_V3_INNER_HALF = 0.105
PROTOCOL_V4_INNER_HALF = 0.114
PROTOCOL_V5_INNER_HALF = 0.120
WALL = 0.015
FRONT_X = 0.350
BACK_X = 0.800
HALF_X = 0.5 * (BACK_X - FRONT_X)
CENTER_X = 0.5 * (BACK_X + FRONT_X)


def _drawer_scene(
    *,
    name: str,
    inner_half: float,
    target_x: float,
    duration: float,
    drawer_y: float = DRAWER_Y,
    inner_half_y: float | None = None,
    inner_half_z: float | None = None,
) -> SceneDefinition:
    """UR5 starts below the open drawer; a compact wrist must enter it.

    Geometry is fixed before controller testing.  The only controller target is
    the final point near the drawer back; ``waypoints`` intentionally contains
    exactly that one point.
    """

    inner_y = inner_half if inner_half_y is None else float(inner_half_y)
    inner_z = inner_half if inner_half_z is None else float(inner_half_z)
    boxes = (
        BoxObstacle(
            "drawer_bottom",
            (CENTER_X, drawer_y, DRAWER_Z - inner_z - WALL),
            (HALF_X, inner_y + 0.045, WALL),
            "shelf",
        ),
        BoxObstacle(
            "drawer_left",
            (CENTER_X, drawer_y - inner_y - WALL, DRAWER_Z),
            (HALF_X, WALL, inner_z),
            "shelf",
        ),
        BoxObstacle(
            "drawer_right",
            (CENTER_X, drawer_y + inner_y + WALL, DRAWER_Z),
            (HALF_X, WALL, inner_z),
            "shelf",
        ),
        BoxObstacle(
            "drawer_ceiling",
            (CENTER_X, drawer_y, DRAWER_Z + inner_z + WALL),
            (HALF_X, 0.32, WALL),
            "shelf",
        ),
        BoxObstacle(
            "drawer_back",
            (BACK_X + WALL, drawer_y, DRAWER_Z),
            (WALL, inner_y + 0.045, inner_z + WALL),
            "shelf",
        ),
        BoxObstacle(
            "cabinet_left",
            (CENTER_X, drawer_y - 0.36, 0.67),
            (HALF_X, 0.025, 0.34),
            "shelf",
        ),
        BoxObstacle(
            "cabinet_right",
            (CENTER_X, drawer_y + 0.36, 0.67),
            (HALF_X, 0.025, 0.34),
            "shelf",
        ),
        BoxObstacle(
            "cabinet_top",
            (CENTER_X, drawer_y, 1.01),
            (HALF_X, 0.385, 0.025),
            "shelf",
        ),
    )
    return SceneDefinition(
        name=name,
        description=(
            "UR5e with a 120 mm compact gripper starts below a static open "
            "drawer and must raise the wrist, cross the front plane, and "
            "insert the wrist/forearm toward one final goal."
        ),
        boxes=boxes,
        q0=(-1.63037709, -1.53134835, 1.75428579, -0.22293743, -3.20117342, 1.57079633),
        waypoints=((target_x, drawer_y, DRAWER_Z),),
        duration=duration,
    )


def incremental_drawer_scene() -> SceneDefinition:
    """Retained failed v1: its 0.70 m target is not physically auditable."""

    return _drawer_scene(
        name="incremental_drawer_frozen_v1",
        inner_half=INNER_HALF_Y,
        target_x=0.700,
        duration=35.0,
    )


def incremental_drawer_v2_candidate_scene() -> SceneDefinition:
    """Versioned 240 mm opening/170 mm insertion candidate.

    This is not promoted to frozen until continuous proxy certificates and an
    independent exact-geometry reachability audit both pass.
    """

    return _drawer_scene(
        name="incremental_drawer_candidate_v2",
        inner_half=0.120,
        target_x=0.520,
        duration=35.0,
    )


def incremental_drawer_v3_protocol_scene() -> SceneDefinition:
    """Versioned 210 mm opening selected after the v2 closure gate failed.

    The opening remains substantially wider than the physical wrist/camera,
    while the matched sphere certificate is expected to close a mandatory
    cross-section.  No target or path coordinate is changed from v2.
    """

    return _drawer_scene(
        name="incremental_drawer_protocol_v3",
        inner_half=PROTOCOL_V3_INNER_HALF,
        target_x=0.520,
        duration=35.0,
    )


def incremental_drawer_v4_candidate_scene() -> SceneDefinition:
    """228 mm candidate paired with 100 mm data-driven surface buckets.

    v3's 210 mm opening failed the full-robot proxy reachability gate by
    7.71 mm even though exact geometry had contact-free goal IK solutions.
    The v4 width is fixed before its controller trials; acceptance additionally
    requires the matched sphere proxies to prove a closed mandatory section.
    """

    return _drawer_scene(
        name="incremental_drawer_candidate_v4",
        inner_half=PROTOCOL_V4_INNER_HALF,
        target_x=0.520,
        duration=35.0,
    )


def incremental_drawer_v5_candidate_scene() -> SceneDefinition:
    """240 mm opening candidate paired with 200 mm surface buckets.

    v4/160 mm passed the sphere-section gate but its full-robot ellipsoid
    certificate missed the goal by 4.77 mm.  v5 restores the 240 mm physical
    opening while increasing only the shared, data-driven proxy bucket scale.
    """

    return _drawer_scene(
        name="incremental_drawer_candidate_v5",
        inner_half=PROTOCOL_V5_INNER_HALF,
        target_x=0.520,
        duration=35.0,
    )


def incremental_drawer_v6_deep_candidate_scene() -> SceneDefinition:
    """240 mm opening with a goal 300 mm behind the drawer front."""

    return _drawer_scene(
        name="incremental_drawer_deep_candidate_v6",
        inner_half=PROTOCOL_V5_INNER_HALF,
        target_x=0.650,
        duration=45.0,
    )


def incremental_drawer_v7_deep_candidate_scene() -> SceneDefinition:
    """Deep 300 mm insertion through a 280 mm physical drawer opening."""

    return _drawer_scene(
        name="incremental_drawer_deep_candidate_v7",
        inner_half=0.140,
        target_x=0.650,
        duration=45.0,
    )


def incremental_drawer_v8_aligned_candidate_scene() -> SceneDefinition:
    """Versioned lateral alignment after v7 forearm-sidewall failure.

    Width, insertion depth, controller target semantics and all sensing rules
    remain unchanged.  The entire drawer is translated rigidly by 36.33 mm in
    y so that the natural below-to-opening lift approaches its centerline.
    """

    return _drawer_scene(
        name="incremental_drawer_aligned_candidate_v8",
        inner_half=0.140,
        target_x=0.650,
        duration=45.0,
        drawer_y=0.370,
    )


def incremental_drawer_v9_aligned_narrow_candidate_scene() -> SceneDefinition:
    """264 mm aligned opening after v8 missed sphere closure by 5.97 mm."""

    return _drawer_scene(
        name="incremental_drawer_aligned_narrow_candidate_v9",
        inner_half=0.132,
        target_x=0.650,
        duration=45.0,
        drawer_y=0.370,
    )


def incremental_drawer_v10_rectangular_candidate_scene() -> SceneDefinition:
    """Wide lateral, low vertical drawer for a below-to-inside approach.

    The 340 x 230 mm physical opening gives the forearm lateral room while
    retaining a narrow vertical section where spherical surface proxies may
    close space that thin, data-oriented ellipsoids preserve.
    """

    return _drawer_scene(
        name="incremental_drawer_rectangular_candidate_v10",
        inner_half=0.140,
        inner_half_y=0.170,
        inner_half_z=0.115,
        target_x=0.650,
        duration=45.0,
        drawer_y=DRAWER_Y,
    )


def incremental_drawer_v11_low_rectangular_candidate_scene() -> SceneDefinition:
    """340 x 190 mm opening after v10 missed sphere closure by 17.16 mm."""

    return _drawer_scene(
        name="incremental_drawer_low_rectangular_candidate_v11",
        inner_half=0.140,
        inner_half_y=0.170,
        inner_half_z=0.095,
        target_x=0.650,
        duration=45.0,
        drawer_y=DRAWER_Y,
    )


def incremental_drawer_v12_certified_candidate_scene() -> SceneDefinition:
    """340 x 182 mm opening after v11 missed closure by 2.56 mm."""

    return _drawer_scene(
        name="incremental_drawer_certified_candidate_v12",
        inner_half=0.140,
        inner_half_y=0.170,
        inner_half_z=0.091,
        target_x=0.650,
        duration=45.0,
        drawer_y=DRAWER_Y,
    )
