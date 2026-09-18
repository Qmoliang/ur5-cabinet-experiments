"""Frozen protocol-v1 UR5 drawer scene with no gripper or long tool."""

from __future__ import annotations

from model import BoxObstacle, SceneDefinition


DRAWER_Y = 0.40633091
DRAWER_Z = 0.580
FRONT_X = 0.350
BACK_X = 0.800
INNER_HALF_Y = 0.170  # 340 mm physical width.
INNER_HALF_Z_V1 = 0.091  # Rejected v1: sphere section remained open.
INNER_HALF_Z = 0.070  # v2: 140 mm physical height, frozen before control runs.
INNER_HALF_Z_CLOSED128 = 0.064  # v4.2f: 128 mm, physical UR5 still fits.
WALL = 0.015
TARGET_X_V1_V2 = 0.650  # Rejected: 300 mm target was ellipsoid-infeasible.
TARGET_X = 0.630  # v3: 280 mm behind the open front plane.

Q0_V1_V2 = (
    -1.63037709,
    -1.53134835,
    1.75428579,
    -0.22293743,
    -3.20117342,
    1.57079633,
)
Q0_V3 = (
    -4.702522636775524,
    -1.4120550523623534,
    -1.8463744276536815,
    -2.056637227290278,
    0.004778088364271554,
    -0.4364774265209154,
)

# These values are frozen before formal controller trials.  A changed value
# requires a new scene/protocol version and a fresh geometry-gate report.
KNOWN_PROXY_CELL_SIZE = 0.100
SAFETY_MARGIN = 0.006
NEAR_DISTANCE = 0.040
# Liu/Yim Eq. (27) is a physical contact/overlap branch.  State classification
# uses the raw certified surface gap c_ij, not h_ij=c_ij-SAFETY_MARGIN.
CONTACT_DISTANCE = 0.0
SUCCESS_TOLERANCE = 0.018
CONTROL_HZ = 50.0
CAMERA_HZ = 30.0
# Robot calibration may certify only this small shell around the collision
# certificate at t=0.  It remains short of the drawer front and is
# never reapplied to future poses; later free-space growth must come from depth
# rays or an already-certified physical sweep.
INITIAL_CALIBRATED_FREE_PADDING = 0.005


def protocol_drawer_v1_scene() -> SceneDefinition:
    """Rejected geometry-gate candidate retained as negative evidence."""

    return _protocol_drawer_scene(
        name="protocol_drawer_v1",
        inner_half_z=INNER_HALF_Z_V1,
        target_x=TARGET_X_V1_V2,
        q0=Q0_V1_V2,
    )


def protocol_drawer_v2_scene() -> SceneDefinition:
    """Current formal scene; accepting it still requires all D2 gates."""

    return _protocol_drawer_scene(
        name="protocol_drawer_v2",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X_V1_V2,
        q0=Q0_V1_V2,
    )


def protocol_drawer_v3_scene() -> SceneDefinition:
    """Current formal scene frozen after path-free candidate screening."""

    return _protocol_drawer_scene(
        name="protocol_drawer_v3",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def protocol_drawer_v4_scene() -> SceneDefinition:
    """v4: v3 task geometry with a physically external wrist camera mount."""

    return _protocol_drawer_scene(
        name="protocol_drawer_v4",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def protocol_drawer_v5_scene() -> SceneDefinition:
    """v5 candidate: v4 task with a cross-observing wrist camera pair."""

    return _protocol_drawer_scene(
        name="protocol_drawer_v5",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def protocol_drawer_v6_scene() -> SceneDefinition:
    """v6 candidate: v5 positions with cross-view along horizontal FOV."""

    return _protocol_drawer_scene(
        name="protocol_drawer_v6",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def protocol_drawer_v7_scene() -> SceneDefinition:
    """v7 candidate: compact, cross-observing wrist camera pair."""

    return _protocol_drawer_scene(
        name="protocol_drawer_v7",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def protocol_drawer_v8_scene() -> SceneDefinition:
    """v8 candidate: collision-free radial offset for the v7 camera pair."""

    return _protocol_drawer_scene(
        name="protocol_drawer_v8",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def protocol_drawer_v9_scene() -> SceneDefinition:
    """v9 visibility candidate: v6 positions with outward wrist optical axes."""

    return _protocol_drawer_scene(
        name="protocol_drawer_v9",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def protocol_drawer_v10_scene() -> SceneDefinition:
    """v10 visibility candidate: v6 positions with parallel forward wrist views."""

    return _protocol_drawer_scene(
        name="protocol_drawer_v10",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def protocol_drawer_v11_scene() -> SceneDefinition:
    """v11 candidate: forward-shifted parallel cameras with correct optical faces."""

    return _protocol_drawer_scene(
        name="protocol_drawer_v11",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def protocol_drawer_v6s2_scene() -> SceneDefinition:
    """v6 wrist pair plus a mirrored second shoulder-mounted local view."""

    return _protocol_drawer_scene(
        name="protocol_drawer_v6s2",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def protocol_drawer_v6s3_scene() -> SceneDefinition:
    """v6 wrist pair plus two parallel shoulder views from opposite mounts."""

    return _protocol_drawer_scene(
        name="protocol_drawer_v6s3",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def protocol_drawer_v6s4_scene() -> SceneDefinition:
    """v6s3 with the second shoulder optical center on its front face."""

    return _protocol_drawer_scene(
        name="protocol_drawer_v6s4",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def protocol_drawer_camera_mount_grid_scene() -> SceneDefinition:
    """Non-formal scene used only to screen an upper-arm camera mount."""

    return _protocol_drawer_scene(
        name="protocol_drawer_camera_mount_grid",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def formal_drawer_two_camera_scene() -> SceneDefinition:
    """Frozen final task geometry with one wrist and one forearm D405."""

    return _protocol_drawer_scene(
        name="formal_drawer_two_camera",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def formal_uniform_radius_v3_1_scene() -> SceneDefinition:
    """Same frozen task with the visibility-only v3.1 forearm camera aim."""

    return _protocol_drawer_scene(
        name="formal_uniform_radius_v3_1",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def formal_drawer_camera_mid_scene() -> SceneDefinition:
    """Frozen task with a visibility-only midpoint forearm view."""

    return _protocol_drawer_scene(
        name="formal_drawer_camera_mid",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def formal_drawer_camera_quarter_scene() -> SceneDefinition:
    from grounded_scene import load_scene
    return load_scene()
    """Frozen task with the visibility-only quarter forearm view."""

    return _protocol_drawer_scene(
        name="formal_drawer_camera_quarter",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def formal_drawer_camera_blindspot_scene() -> SceneDefinition:
    """v4.3 camera-only candidate aimed at the cabinet-left outer edge."""

    return _protocol_drawer_scene(
        name="formal_drawer_camera_blindspot",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def formal_drawer_camera_compensated_scene() -> SceneDefinition:
    """Camera-only candidate with complementary wrist/forearm D405 aims."""

    return _protocol_drawer_scene(
        name="formal_drawer_camera_compensated",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def formal_drawer_camera_v5_balanced_scene() -> SceneDefinition:
    """v5 two-D405 mount frozen by visibility/collision-only auditing."""

    return _protocol_drawer_scene(
        name="formal_drawer_camera_v5_balanced",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def formal_drawer_camera_quarter_closed128_scene() -> SceneDefinition:
    """v4.2f narrow passage with the unchanged quarter-view camera layout.

    Only the physical drawer inner height changes from 140 mm to 128 mm.
    The target, robot, cameras, safety margin and proxy construction remain
    unchanged.  This scene is versioned separately so prior evidence is never
    overwritten or silently reinterpreted.
    """

    return _protocol_drawer_scene(
        name="formal_drawer_camera_quarter_closed128",
        inner_half_z=INNER_HALF_Z_CLOSED128,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def formal_drawer_camera_three_sixteenths_scene() -> SceneDefinition:
    """Frozen task with the visibility-only 3/16 forearm view."""

    return _protocol_drawer_scene(
        name="formal_drawer_camera_three_sixteenths",
        inner_half_z=INNER_HALF_Z,
        target_x=TARGET_X,
        q0=Q0_V3,
    )


def formal_protocol_scene() -> SceneDefinition:
    return formal_drawer_two_camera_scene()


def _protocol_drawer_scene(
    *,
    name: str,
    inner_half_z: float,
    target_x: float,
    q0: tuple[float, ...],
) -> SceneDefinition:
    half_x = 0.5 * (BACK_X - FRONT_X)
    center_x = 0.5 * (BACK_X + FRONT_X)
    boxes = (
        BoxObstacle(
            "drawer_bottom",
            (center_x, DRAWER_Y, DRAWER_Z - inner_half_z - WALL),
            (half_x, INNER_HALF_Y + 0.045, WALL),
            "shelf",
        ),
        BoxObstacle(
            "drawer_left",
            (center_x, DRAWER_Y - INNER_HALF_Y - WALL, DRAWER_Z),
            (half_x, WALL, inner_half_z),
            "shelf",
        ),
        BoxObstacle(
            "drawer_right",
            (center_x, DRAWER_Y + INNER_HALF_Y + WALL, DRAWER_Z),
            (half_x, WALL, inner_half_z),
            "shelf",
        ),
        BoxObstacle(
            "drawer_ceiling",
            (center_x, DRAWER_Y, DRAWER_Z + inner_half_z + WALL),
            (half_x, 0.32, WALL),
            "shelf",
        ),
        BoxObstacle(
            "drawer_back",
            (BACK_X + WALL, DRAWER_Y, DRAWER_Z),
            (WALL, INNER_HALF_Y + 0.045, inner_half_z + WALL),
            "shelf",
        ),
        BoxObstacle(
            "cabinet_left",
            (center_x, DRAWER_Y - 0.36, 0.67),
            (half_x, 0.025, 0.34),
            "shelf",
        ),
        BoxObstacle(
            "cabinet_right",
            (center_x, DRAWER_Y + 0.36, 0.67),
            (half_x, 0.025, 0.34),
            "shelf",
        ),
        BoxObstacle(
            "cabinet_top",
            (center_x, DRAWER_Y, 1.01),
            (half_x, 0.385, 0.025),
            "shelf",
        ),
    )
    return SceneDefinition(
        name=name,
        description=(
            "Formal no-gripper UR5e task: physical wrist and forearm depth "
            "views provide complementary causal local observations, and "
            "the wrist/end-effector center must move from below the open drawer "
            "to one final target deep inside; scene version is frozen "
            "before any closed-loop comparison."
        ),
        boxes=boxes,
        q0=q0,
        waypoints=((target_x, DRAWER_Y, DRAWER_Z),),
        duration=45.0,
    )
