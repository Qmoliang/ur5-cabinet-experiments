"""UR5e MuJoCo model and IRIS-inspired scene definitions.

The robot source is Google DeepMind's official MuJoCo Menagerie UR5e MJCF.
At load time we remove only mesh visual geoms so the reproduction remains
self-contained without the OBJ bundle.  The official body tree, joint axes,
link transforms, collision capsules, inertias, actuators, and home keyframe are
left unchanged.  Primitive geoms are recolored and used for both collision
checking and visualization.

The paper by Werner et al. benchmarks IIWAShelf, 4Shelves, and IIWABins in
Drake.  It does not publish reusable dimensions for those scenes.  ``shelf``
below is therefore an explicitly labelled geometric reconstruction of that
environment class.  ``cage`` is an additional LiuQP stress test inherited from
the local reproduction; it is not claimed to be an official IRIS benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
OFFICIAL_UR5E_XML = (
    ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "universal_robots_ur5e"
    / "ur5e.xml"
)

DT = 0.02  # 50 Hz sequential kinematic control.
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@dataclass(frozen=True)
class BoxObstacle:
    """Axis-aligned task-space obstacle represented by a MuJoCo box geom."""

    name: str
    center: tuple[float, float, float]
    half_size: tuple[float, float, float]
    material: str = "obstacle"


@dataclass(frozen=True)
class SceneDefinition:
    name: str
    description: str
    boxes: tuple[BoxObstacle, ...]
    q0: tuple[float, ...]
    waypoints: tuple[tuple[float, float, float], ...]
    duration: float


@dataclass(frozen=True)
class CertificateSphere:
    """Sphere fixed in one robot body, used by LiuQP Eqs. (23)-(27)."""

    body_id: int
    body_name: str
    source_geom_id: int
    local_center: np.ndarray
    radius: float


def _shelf_boxes() -> tuple[BoxObstacle, ...]:
    """Open-front, two-column shelf inspired by IRIS IIWAShelf/4Shelves."""

    # The shelf stands behind the robot in +x.  Internal clear width is 0.84 m,
    # clear height 0.78 m, and depth 0.34 m.  The target compartment is open at
    # the front; the back, sides, center divider, and boards are obstacles.
    return (
        BoxObstacle("shelf_back", (0.72, 0.00, 0.60), (0.025, 0.48, 0.46), "shelf"),
        BoxObstacle("shelf_left", (0.55, 0.50, 0.60), (0.19, 0.025, 0.46), "shelf"),
        BoxObstacle("shelf_right", (0.55, -0.50, 0.60), (0.19, 0.025, 0.46), "shelf"),
        BoxObstacle("shelf_bottom", (0.55, 0.00, 0.14), (0.19, 0.50, 0.025), "shelf"),
        BoxObstacle("shelf_middle", (0.55, 0.00, 0.56), (0.19, 0.50, 0.025), "shelf"),
        BoxObstacle("shelf_top", (0.55, 0.00, 1.06), (0.19, 0.50, 0.025), "shelf"),
        BoxObstacle("shelf_divider", (0.55, 0.00, 0.81), (0.19, 0.025, 0.225), "shelf"),
    )


def _cage_boxes() -> tuple[BoxObstacle, ...]:
    """Open-top cage scaled to the UR5e workspace (non-IRIS extension)."""

    return (
        BoxObstacle("cage_back", (0.68, 0.00, 0.34), (0.025, 0.50, 0.21), "cage"),
        BoxObstacle("cage_front", (0.12, 0.00, 0.34), (0.025, 0.50, 0.21), "cage"),
        BoxObstacle("cage_left", (0.40, 0.52, 0.34), (0.305, 0.025, 0.21), "cage"),
        BoxObstacle("cage_right", (0.40, -0.52, 0.34), (0.305, 0.025, 0.21), "cage"),
        BoxObstacle("cage_floor", (0.40, 0.00, 0.10), (0.305, 0.50, 0.025), "cage"),
    )


SCENES: dict[str, SceneDefinition] = {
    "shelf": SceneDefinition(
        name="shelf",
        description="IRIS-inspired open-front shelf; approach then insert into a compartment.",
        boxes=_shelf_boxes(),
        q0=(-1.5708, -1.75, 1.95, -1.75, -1.5708, 0.0),
        waypoints=((0.25, -0.24, 0.78), (0.43, -0.24, 0.78), (0.52, -0.24, 0.78)),
        duration=22.0,
    ),
    "cage": SceneDefinition(
        name="cage",
        description="Open-top cage extension; rise, cross the rim, then descend.",
        boxes=_cage_boxes(),
        q0=(-1.5708, -1.92, 2.08, -1.70, -1.5708, 0.0),
        waypoints=((0.00, 0.00, 0.95), (0.34, 0.09, 0.92), (0.31, 0.12, 0.53)),
        duration=22.0,
    ),
}


def _vec(values: tuple[float, float, float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _remove_mesh_visuals(root: ET.Element) -> None:
    """Remove OBJ-only visuals while preserving official primitive collisions."""

    asset = root.find("asset")
    if asset is not None:
        for mesh in list(asset.findall("mesh")):
            asset.remove(mesh)
    for body in root.findall(".//body"):
        for geom in list(body.findall("geom")):
            if geom.get("class") == "visual" or geom.get("mesh") is not None:
                body.remove(geom)
            elif geom.get("class") in {"collision", "eef_collision"}:
                geom.set("rgba", "0.49 0.678 0.8 1")
                geom.set("group", "1")
                geom.set("friction", "0.8 0.1 0.1")


def build_xml(scene: SceneDefinition) -> str:
    # This isolated rerun uses the single audited physical cabinet XML.
    return (ROOT / "assets/drawer.xml").read_text(encoding="utf-8")
    """Create the self-contained MJCF for one task-space scene."""

    if not OFFICIAL_UR5E_XML.exists():
        raise FileNotFoundError(
            f"Official UR5e MJCF is missing: {OFFICIAL_UR5E_XML}. "
            "See README.md for the raw-source download command."
        )
    root = ET.parse(OFFICIAL_UR5E_XML).getroot()
    root.set("model", f"ur5e_liuqp_{scene.name}")
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", f"{DT:.9g}")
    option.set("gravity", "0 0 0")
    _remove_mesh_visuals(root)

    # A wrist-mounted active depth camera. In MuJoCo a camera looks along its
    # local -Z axis. The xyaxes below choose camera +X=link +X and camera
    # +Y=link +Z, hence optical -Z points along wrist-link +Y: the same
    # direction as the attachment tool. Wrist joints 1 and 2 can therefore
    # pitch/pan the view without relying on a privileged fixed world camera.
    # The incremental drawer scene adds measured-scale collision housings below
    # so perception hardware remains part of both LiuQP certificates.
    wrist_camera_body = root.find(".//body[@name='wrist_3_link']")
    if wrist_camera_body is None:
        raise RuntimeError("UR5e wrist_3_link not found for depth camera")
    wrist_camera = ET.SubElement(
        wrist_camera_body,
        "camera",
        name="ur5_depth_wrist",
        pos="0 0.072 0.055",
        xyaxes="1 0 0 0 0 1",
        fovy="70",
    )

    if scene.name.startswith("protocol_drawer"):
        # Formal protocol hardware: one wrist D405-scale view and one
        # D405-scale shoulder view; no gripper, probe, or retrieval tool.
        # Both are moving local sensors with collision housings; neither is a
        # privileged fixed world camera.  The task site is
        # the physical end-effector/camera-body center requested by the
        # protocol, so reaching the Cartesian target requires the UR5 wrist
        # itself to enter the drawer.
        attachment = wrist_camera_body.find("site[@name='attachment_site']")
        if attachment is None:
            raise RuntimeError("UR5e attachment_site not found")
        attachment.set("pos", "0 0.070 0")
        # The optical center is on the housing front face, not inside its
        # opaque collision box.  The attachment/task site remains at the
        # housing center, so this changes sensing only and not the task point.
        external_wrist_mount = scene.name in {
            "protocol_drawer_v4",
            "protocol_drawer_v5",
            "protocol_drawer_v6",
            "protocol_drawer_v7",
            "protocol_drawer_v8",
            "protocol_drawer_v9",
            "protocol_drawer_v10",
            "protocol_drawer_v11",
            "protocol_drawer_v6s2",
            "protocol_drawer_v6s3",
            "protocol_drawer_v6s4",
            "protocol_drawer_camera_mount_grid",
        }
        wrist_mount_z = -0.065 if external_wrist_mount else 0.0
        if scene.name in {
            "protocol_drawer_v5",
            "protocol_drawer_v6",
            "protocol_drawer_v7",
            "protocol_drawer_v8",
            "protocol_drawer_v9",
            "protocol_drawer_v10",
            "protocol_drawer_v11",
            "protocol_drawer_v6s2",
            "protocol_drawer_v6s3",
            "protocol_drawer_v6s4",
            "protocol_drawer_camera_mount_grid",
        }:
            # Two D405-scale wrist cameras are mounted on opposite sides of
            # the wrist and yawed 20 degrees inward.  Each camera observes the
            # other's <70 mm near-field blind shell.  The layout was proposed
            # solely by the v4 strict-unknown visibility audit, before any
            # sphere/ellipsoid closed-loop comparison.
            wrist_camera.set("name", "ur5_depth_wrist")
            wrist_camera.set(
                "pos",
                (
                    "-0.065 0.081575 0"
                    if scene.name == "protocol_drawer_v11"
                    else (
                        "-0.0427873 0.0503016 -0.060"
                        if scene.name == "protocol_drawer_v8"
                        else (
                            "-0.0387125 0.0508890 -0.055"
                            if scene.name == "protocol_drawer_v7"
                            else "-0.060699212 0.051816813 0"
                        )
                    )
                ),
            )
            wrist_camera.set(
                "xyaxes",
                (
                    "0 0 1 -0.939692621 0.342020143 0"
                    if scene.name == "protocol_drawer_v5"
                    else (
                        (
                            "1 0 0 0 0 1"
                            if scene.name in {
                                "protocol_drawer_v10",
                                "protocol_drawer_v11",
                            }
                            else "0.939692621 0.342020143 0 0 0 1"
                        )
                        if scene.name in {
                            "protocol_drawer_v9",
                            "protocol_drawer_v10",
                            "protocol_drawer_v11",
                        }
                        else (
                        "0.819152044 -0.573576436 0 0 0 1"
                        if scene.name == "protocol_drawer_v8"
                        else (
                            "0.866025404 -0.5 0 0 0 1"
                            if scene.name == "protocol_drawer_v7"
                            else "0.939692621 -0.342020143 0 0 0 1"
                        )
                        )
                    )
                ),
            )
            wrist_camera.set("fovy", "58")
            ET.SubElement(
                wrist_camera_body,
                "camera",
                name="ur5_depth_wrist_right",
                pos=(
                    "0.065 0.081575 0"
                    if scene.name == "protocol_drawer_v11"
                    else (
                        "0.0427873 0.0503016 -0.060"
                        if scene.name == "protocol_drawer_v8"
                        else (
                            "0.0387125 0.0508890 -0.055"
                            if scene.name == "protocol_drawer_v7"
                            else "0.060699212 0.051816813 0"
                        )
                    )
                ),
                xyaxes=(
                    "0 0 1 -0.939692621 -0.342020143 0"
                    if scene.name == "protocol_drawer_v5"
                    else (
                        (
                            "1 0 0 0 0 1"
                            if scene.name in {
                                "protocol_drawer_v10",
                                "protocol_drawer_v11",
                            }
                            else "0.939692621 -0.342020143 0 0 0 1"
                        )
                        if scene.name in {
                            "protocol_drawer_v9",
                            "protocol_drawer_v10",
                            "protocol_drawer_v11",
                        }
                        else (
                        "0.819152044 0.573576436 0 0 0 1"
                        if scene.name == "protocol_drawer_v8"
                        else (
                            "0.866025404 0.5 0 0 0 1"
                            if scene.name == "protocol_drawer_v7"
                            else "0.939692621 0.342020143 0 0 0 1"
                        )
                        )
                    )
                ),
                fovy="58",
            )
            wrist_mount_z = 0.0
        else:
            wrist_camera.set("pos", f"0 0.082575 {wrist_mount_z:.9g}")
            wrist_camera.set("xyaxes", "1 0 0 0 0 1")
        wrist_camera.set("fovy", "58")
        if scene.name in {
            "protocol_drawer_v5",
            "protocol_drawer_v6",
            "protocol_drawer_v7",
            "protocol_drawer_v8",
            "protocol_drawer_v9",
            "protocol_drawer_v10",
            "protocol_drawer_v11",
            "protocol_drawer_v6s2",
            "protocol_drawer_v6s3",
            "protocol_drawer_v6s4",
            "protocol_drawer_camera_mount_grid",
        }:
            ET.SubElement(
                wrist_camera_body,
                "geom",
                name="protocol_d405_camera_housing",
                type="box",
                pos=(
                    "-0.065 0.070 0"
                    if scene.name == "protocol_drawer_v11"
                    else (
                        "-0.050 0.040 -0.060"
                        if scene.name == "protocol_drawer_v8"
                        else (
                            "-0.045 0.040 -0.055"
                            if scene.name == "protocol_drawer_v7"
                            else "-0.065 0.040 0"
                        )
                    )
                ),
                xyaxes=(
                    "0 0 1 -0.939692621 0.342020143 0"
                    if scene.name == "protocol_drawer_v5"
                    else (
                        (
                            "1 0 0 0 0 1"
                            if scene.name in {
                                "protocol_drawer_v10",
                                "protocol_drawer_v11",
                            }
                            else "0.939692621 0.342020143 0 0 0 1"
                        )
                        if scene.name in {
                            "protocol_drawer_v9",
                            "protocol_drawer_v10",
                            "protocol_drawer_v11",
                        }
                        else (
                        "0.819152044 -0.573576436 0 0 0 1"
                        if scene.name == "protocol_drawer_v8"
                        else (
                            "0.866025404 -0.5 0 0 0 1"
                            if scene.name == "protocol_drawer_v7"
                            else "0.939692621 -0.342020143 0 0 0 1"
                        )
                        )
                    )
                ),
                size="0.021075 0.021075 0.011575",
                rgba="0.10 0.12 0.15 1",
                group="1",
                friction="0.8 0.1 0.1",
            )
            ET.SubElement(
                wrist_camera_body,
                "geom",
                name="protocol_d405_camera_housing_right",
                type="box",
                pos=(
                    "0.065 0.070 0"
                    if scene.name == "protocol_drawer_v11"
                    else (
                        "0.050 0.040 -0.060"
                        if scene.name == "protocol_drawer_v8"
                        else (
                            "0.045 0.040 -0.055"
                            if scene.name == "protocol_drawer_v7"
                            else "0.065 0.040 0"
                        )
                    )
                ),
                xyaxes=(
                    "0 0 1 -0.939692621 -0.342020143 0"
                    if scene.name == "protocol_drawer_v5"
                    else (
                        (
                            "1 0 0 0 0 1"
                            if scene.name in {
                                "protocol_drawer_v10",
                                "protocol_drawer_v11",
                            }
                            else "0.939692621 -0.342020143 0 0 0 1"
                        )
                        if scene.name in {
                            "protocol_drawer_v9",
                            "protocol_drawer_v10",
                            "protocol_drawer_v11",
                        }
                        else (
                        "0.819152044 0.573576436 0 0 0 1"
                        if scene.name == "protocol_drawer_v8"
                        else (
                            "0.866025404 0.5 0 0 0 1"
                            if scene.name == "protocol_drawer_v7"
                            else "0.939692621 0.342020143 0 0 0 1"
                        )
                        )
                    )
                ),
                size="0.021075 0.021075 0.011575",
                rgba="0.10 0.12 0.15 1",
                group="1",
                friction="0.8 0.1 0.1",
            )
        else:
            ET.SubElement(
                wrist_camera_body,
                "geom",
                name="protocol_d405_camera_housing",
                type="box",
                pos=f"0 0.070 {wrist_mount_z:.9g}",
                xyaxes="1 0 0 0 0 1",
                size="0.021075 0.021075 0.011575",
                rgba="0.10 0.12 0.15 1",
                group="1",
                friction="0.8 0.1 0.1",
            )
        if not external_wrist_mount:
            # Retain the rejected v1-v3 rear aperture only so historical
            # scenes remain reproducible. It is deliberately absent from v4:
            # its first return was always the arm itself.
            rear_xyaxes = (
                "0.651339451 -0.653241877 0.386046590 "
                "0.694608556 0.308534214 -0.649865827"
            )
            ET.SubElement(
                wrist_camera_body,
                "camera",
                name="ur5_depth_wrist_rear",
                pos="0 0.057425 0",
                xyaxes=rear_xyaxes,
                fovy="68",
            )
        shoulder_body = root.find(".//body[@name='shoulder_link']")
        if shoulder_body is None:
            raise RuntimeError("UR5e shoulder_link not found")
        ET.SubElement(
            shoulder_body,
            "camera",
            name="ur5_depth_shoulder",
            pos="-0.007228563 -0.101674498 0.086046860",
            xyaxes=(
                "0.755099474 0.655610238 0 "
                "0.315259117 -0.363099873 0.876795399"
            ),
            # D405 official depth FOV is 87 x 58 degrees.  Formal capture uses
            # a 16:9 raster, so fovy=58 reproduces that calibrated frustum.
            fovy="58",
        )
        ET.SubElement(
            shoulder_body,
            "geom",
            name="protocol_d405_shoulder_housing",
            type="box",
            pos="0 -0.11 0.08",
            xyaxes=(
                "0.755099474 0.655610238 0 "
                "0.315259117 -0.363099873 0.876795399"
            ),
            size="0.021075 0.021075 0.011575",
            rgba="0.10 0.12 0.15 1",
            group="1",
            friction="0.8 0.1 0.1",
        )
        if scene.name in {
            "protocol_drawer_v6s2",
            "protocol_drawer_v6s3",
            "protocol_drawer_v6s4",
        }:
            secondary_shoulder_axes = (
                "-0.755099474 0.655610238 0 "
                "0.315259117 0.363099873 0.876795399"
                if scene.name == "protocol_drawer_v6s2"
                else (
                    "0.755099474 0.655610238 0 "
                    "0.315259117 -0.363099873 0.876795399"
                )
            )
            ET.SubElement(
                shoulder_body,
                "camera",
                name="ur5_depth_shoulder_right",
                pos=(
                    "-0.007228563 0.118325502 0.086046860"
                    if scene.name == "protocol_drawer_v6s4"
                    else "-0.007228563 0.101674498 0.086046860"
                ),
                xyaxes=secondary_shoulder_axes,
                fovy="68",
            )
            ET.SubElement(
                shoulder_body,
                "geom",
                name="protocol_d405_shoulder_housing_right",
                type="box",
                pos="0 0.11 0.08",
                xyaxes=secondary_shoulder_axes,
                size="0.021075 0.021075 0.011575",
                rgba="0.10 0.12 0.15 1",
                group="1",
                friction="0.8 0.1 0.1",
            )
        if scene.name == "protocol_drawer_camera_mount_grid":
            upper_arm_body = root.find(".//body[@name='upper_arm_link']")
            if upper_arm_body is None:
                raise RuntimeError("UR5e upper_arm_link not found")
            ET.SubElement(
                upper_arm_body,
                "camera",
                name="ur5_depth_upper_arm_candidate",
                pos="0.075 0 0.20",
                xyaxes="1 0 0 0 0 1",
                fovy="68",
            )
            ET.SubElement(
                upper_arm_body,
                "geom",
                name="protocol_d405_upper_arm_candidate_housing",
                type="box",
                pos="0.075 -0.012575 0.20",
                xyaxes="1 0 0 0 0 1",
                size="0.021075 0.021075 0.011575",
                rgba="0.10 0.12 0.15 1",
                group="1",
                friction="0.8 0.1 0.1",
            )

    if scene.name in {
        "formal_drawer_two_camera",
        "formal_uniform_radius_v3_1",
        "formal_drawer_camera_mid",
        "formal_drawer_camera_blindspot",
        "formal_drawer_camera_compensated",
        "formal_drawer_camera_v5_balanced",
        "experiment_07_drawer",
        "experiment_07_drawer_official",
        "experiment_07_birdcage",
        "formal_drawer_camera_quarter",
        "formal_drawer_camera_quarter_closed128",
        "formal_drawer_camera_three_sixteenths",
    }:
        # Final physical sensor layout: one forward-looking wrist D405 and
        # one oblique D405 mounted radially on forearm_link.  There is no
        # gripper, probe, rear virtual aperture, fixed world camera, or unused
        # camera housing.  Both housings are collision geometry and therefore
        # enter the same robot certificate used by sphere and ellipsoid LiuQP.
        attachment = wrist_camera_body.find("site[@name='attachment_site']")
        if attachment is None:
            raise RuntimeError("UR5e attachment_site not found")
        attachment.set("pos", "0 0.070 0")
        wrist_camera.set("pos", "0 0.082575 -0.065")
        wrist_camera.attrib.pop("xyaxes", None)
        if scene.name == "formal_drawer_camera_compensated":
            wrist_quat = (
                "0.4256266282470781 0.7011772209104119 "
                "0.2110473437101772 -0.5316497878455829"
            )
            wrist_housing_pos = (
                "-0.007116276788598071 0.07224732925800659 "
                "-0.06591019048173087"
            )
        else:
            wrist_quat = (
                "0.599512975022579 0.7813005203969409 "
                "-0.13776436178754506 -0.10571031278128071"
            )
            wrist_housing_pos = (
                "-0.0041543535760831035 0.0711610073572811 "
                "-0.0682546494921642"
            )
        wrist_camera.set("quat", wrist_quat)
        wrist_camera.set("fovy", "58")
        ET.SubElement(
            wrist_camera_body,
            "geom",
            name="formal_d405_wrist_housing",
            type="box",
            pos=wrist_housing_pos,
            quat=wrist_quat,
            size="0.021075 0.021075 0.011575",
            rgba="0.10 0.12 0.15 1",
            group="1",
            friction="0.8 0.1 0.1",
        )
        forearm_body = root.find(".//body[@name='forearm_link']")
        if forearm_body is None:
            raise RuntimeError("UR5e forearm_link not found")
        # Both 58-degree D405 views are frozen from the two-trajectory
        # visibility/collision audit, not from controller success.  The wrist
        # is yawed 20 degrees and pitched upward 15 degrees; the forearm view
        # is aimed toward the lower-left opening so the pair divides ceiling,
        # floor, and cabinet-edge coverage before either frozen path is near.
        if scene.name == "formal_uniform_radius_v3_1":
            # Chosen by the two-trajectory visibility/collision-only grid:
            # wrist mounting remains frozen and the forearm optical axis targets
            # [0.35, DRAWER_Y-0.16, DRAWER_Z-0.06].  Controller success was not
            # an input to candidate ranking.
            forearm_quat = (
                "0.4903445143620375 0.4486407527374915 "
                "0.5553017392557258 -0.4999237047767992"
            )
            forearm_housing_pos = (
                "0.0012072758130997321 0.06248543457523756 "
                "0.30975759055207197"
            )
        elif scene.name == "formal_drawer_camera_mid":
            # Spherical midpoint between the original frozen view and the
            # visibility-only v3.1 view.  It is screened only by causal
            # observed-before-risk coverage, never by controller success.
            forearm_quat = (
                "0.5009430066462927 0.4376884646140742 "
                "0.5286128503792367 -0.527307658250746"
            )
            forearm_housing_pos = (
                "0.0008566380435767171 0.06251773243732636 "
                "0.31072784913614746"
            )
        elif scene.name == "formal_drawer_camera_blindspot":
            # Selected by audit_v4_3_forearm_blindspot_grid.py using only
            # frozen-trajectory visibility and camera-housing collision.
            forearm_quat = (
                "0.5311666153160751 0.4004262832359864 "
                "0.5152439122715451 -0.54041144448891"
            )
            forearm_housing_pos = (
                "0.0014407282830364047 0.06264790997181233 "
                "0.3118656899394073"
            )
        elif scene.name == "formal_drawer_camera_compensated":
            # Visibility/collision-only selection: forearm covers the cabinet
            # edge while the wrist view compensates the drawer front surfaces.
            forearm_quat = (
                "0.5311666153160751 0.4004262832359864 "
                "0.5152439122715451 -0.54041144448891"
            )
            forearm_housing_pos = (
                "0.0014407282830364047 0.06264790997181233 "
                "0.3118656899394073"
            )
        elif scene.name in {
            "formal_drawer_camera_v5_balanced",
            "experiment_07_drawer",
        "experiment_07_drawer_official",
            "experiment_07_birdcage",
        }:
            # Frozen before v5 closed-loop runs.  The two-trajectory
            # visibility/collision-only screen selected a -15 degree optical
            # roll.  The one remaining exact-ray edge sample is already
            # covered at cycle zero by a measured CenterVox certificate.
            forearm_quat = (
                "0.43664763451208494 0.35403818166274936 "
                "0.5897313911993957 -0.5798385080640641"
            )
            forearm_housing_pos = (
                "0.001313329124679112 0.06251204654977272 "
                "0.3106758773817978"
            )
        elif scene.name in {
            "formal_drawer_camera_quarter",
            "formal_drawer_camera_quarter_closed128",
        }:
            forearm_quat = (
                "0.5059250092158937 0.4319329943648225 "
                "0.514929627115348 -0.5406674139873654"
            )
            forearm_housing_pos = (
                "0.0006782482145986236 0.06253402911766576 "
                "0.3112126041781856"
            )
        elif scene.name == "formal_drawer_camera_three_sixteenths":
            forearm_quat = (
                "0.50713709970600485 0.43046543726458292 "
                "0.5114745061144883 -0.54397178145316816"
            )
            forearm_housing_pos = (
                "0.00063416258139019788 0.062538078662868105 "
                "0.31133385531369506"
            )
        else:
            forearm_quat = (
                "0.5106926718694792 0.4259945316165838 "
                "0.5010282490438904 -0.5537981108595377"
            )
            forearm_housing_pos = (
                "0.0005019056817649207 0.06255022729847515 "
                "0.31169760872022345"
            )
        ET.SubElement(
            forearm_body,
            "camera",
            name="ur5_depth_forearm",
            pos="0 0.075 0.31",
            quat=forearm_quat,
            fovy="58",
        )
        ET.SubElement(
            forearm_body,
            "geom",
            name="formal_d405_forearm_housing",
            type="box",
            pos=forearm_housing_pos,
            quat=forearm_quat,
            size="0.021075 0.021075 0.011575",
            rgba="0.10 0.12 0.15 1",
            group="1",
            friction="0.8 0.1 0.1",
        )

    if scene.name.startswith("incremental_drawer"):
        # Compact, fixed parallel-jaw end effector (120 mm reach) and one
        # D405-scale camera housing.  All pieces are collision geoms and are
        # automatically covered by build_robot_certificate; no long probe is
        # present and the attachment site starts outside the drawer.
        attachment = wrist_camera_body.find("site[@name='attachment_site']")
        if attachment is None:
            raise RuntimeError("UR5e attachment_site not found")
        attachment.set("pos", "0 0.12 0")
        # Eye-in-hand layout: the D405 is mounted radially above the compact
        # palm. Its optical center is outside the official wrist collision
        # cylinder and looks forward between the two fingers.
        wrist_camera.set("pos", "0 0.082575 0.065")
        wrist_camera.set("fovy", "58")
        shoulder_body = root.find(".//body[@name='shoulder_link']")
        if shoulder_body is None:
            raise RuntimeError("UR5e shoulder_link not found")
        ET.SubElement(
            shoulder_body,
            "camera",
            name="ur5_depth_shoulder",
            pos="0 -0.11 0.08",
            xyaxes=(
                "-0.65816089 -0.75287731 0 "
                "-0.44455929 0.38863110 0.80705198"
            ),
            fovy="68",
        )
        ET.SubElement(
            shoulder_body,
            "geom",
            name="shoulder_depth_camera_housing",
            type="capsule",
            fromto="0 -0.075 0.08 0 -0.11 0.08",
            size="0.025",
            rgba="0.10 0.12 0.15 1",
            group="1",
            friction="0.8 0.1 0.1",
        )
        # Intel's D405 maximum mechanical envelope is 42.15 x 42.15 x
        # 23.15 mm.  Represent it as a box; both certificate builders below
        # split that box into analytically covered cells.  This avoids the
        # previous capsule whose hemispherical ends made a nominal 23 mm body
        # 83 mm long.
        ET.SubElement(
            wrist_camera_body,
            "geom",
            name="d405_camera_housing",
            type="box",
            pos="0 0.070 0.065",
            size="0.021075 0.011575 0.021075",
            rgba="0.10 0.12 0.15 1",
            group="1",
            friction="0.8 0.1 0.1",
        )
        for name, start, end, radius in (
            ("compact_gripper_palm", (0.0, 0.0, 0.0), (0.0, 0.055, 0.0), 0.026),
            ("compact_gripper_left", (-0.035, 0.045, 0.0), (-0.035, 0.115, 0.0), 0.009),
            ("compact_gripper_right", (0.035, 0.045, 0.0), (0.035, 0.115, 0.0), 0.009),
        ):
            ET.SubElement(
                wrist_camera_body,
                "geom",
                name=name,
                type="capsule",
                fromto=f"{_vec(start)} {_vec(end)}",
                size=f"{radius:.9g}",
                rgba="0.10 0.12 0.15 1",
                group="1",
                friction="0.8 0.1 0.1",
            )

    if scene.name.startswith("shelf_drawer"):
        # Exact collision geometry for a slender retrieval tool.  Both robot
        # certificate builders discover this cylinder automatically.
        wrist = root.find(".//body[@name='wrist_3_link']")
        if wrist is None:
            raise RuntimeError("UR5e wrist_3_link not found")
        attachment = wrist.find("site[@name='attachment_site']")
        if attachment is None:
            raise RuntimeError("UR5e attachment_site not found")
        long_retrieval_tool = "long_tool" in scene.name
        attachment_distance = 0.60 if long_retrieval_tool else 0.45
        tool_half_length = 0.25 if long_retrieval_tool else 0.175
        tool_center = 0.35 if long_retrieval_tool else 0.275
        attachment.set("pos", f"0 {attachment_distance:.9g} 0")
        ET.SubElement(
            wrist,
            "geom",
            name="drawer_retrieval_tool",
            type="cylinder",
            pos=f"0 {tool_center:.9g} 0",
            quat="1 1 0 0",
            size=f"0.018 {tool_half_length:.9g}",
            rgba="0.12 0.16 0.20 1",
            group="1",
            friction="0.8 0.1 0.1",
        )

    asset = root.find("asset")
    assert asset is not None
    ET.SubElement(asset, "material", name="floor", rgba="0.28 0.30 0.34 1")
    ET.SubElement(asset, "material", name="shelf", rgba="0.48 0.31 0.18 1")
    ET.SubElement(asset, "material", name="cage", rgba="0.20 0.24 0.28 0.55")
    ET.SubElement(asset, "material", name="obstacle", rgba="0.55 0.22 0.18 1")
    ET.SubElement(
        asset,
        "material",
        name="experiment_07_cabinet",
        rgba="0.29 0.42 0.50 1",
    )
    ET.SubElement(
        asset,
        "material",
        name="experiment_07_cage",
        rgba="0.24 0.34 0.38 1",
    )

    worldbody = root.find("worldbody")
    assert worldbody is not None
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        pos="0 0 0",
        size="2 2 0.04",
        material="floor",
        contype="1",
        conaffinity="1",
    )
    for box in scene.boxes:
        ET.SubElement(
            worldbody,
            "geom",
            name=box.name,
            type="box",
            pos=_vec(box.center),
            size=_vec(box.half_size),
            material=box.material,
            contype="1",
            conaffinity="1",
        )
    ET.SubElement(
        worldbody,
        "site",
        name="liuqp_target",
        type="sphere",
        pos=_vec(scene.waypoints[-1]),
        size="0.025",
        rgba=(
            "0.95 0.42 0.10 0.85"
            if scene.name.startswith("experiment_07")
            else "0.10 0.90 0.25 0.65"
        ),
        group="0",
    )

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("azimuth", "145")
    global_visual.set("elevation", "-23")
    global_visual.set("offwidth", "960")
    global_visual.set("offheight", "720")

    if scene.name == "experiment_07_drawer_official":
        from official_drawer_07_3 import decorate
        root = decorate(root)

    return ET.tostring(root, encoding="unicode")


def build_model(scene: SceneDefinition) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_xml(scene))


def set_configuration(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> None:
    data.qpos[: len(JOINT_NAMES)] = np.asarray(q, dtype=float)
    data.qvel[:] = 0.0
    if model.nu:
        data.ctrl[:] = data.qpos[: model.nu]
    mujoco.mj_forward(model, data)


def attachment_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return data.site_xpos[model.site("attachment_site").id].copy()


def _quat_matrix(quat: np.ndarray) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def build_robot_certificate(model: mujoco.MjModel, axial_spacing: float = 0.035) -> list[CertificateSphere]:
    """Cover official UR5e capsule/cylinder collision geoms with spheres.

    A line segment swept by a radius-r ball is covered by discrete balls with
    radius sqrt(r^2 + (spacing/2)^2).  This small analytic inflation covers the
    gaps between adjacent centers instead of merely sampling the capsule axis.
    """

    certificate: list[CertificateSphere] = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id == 0:
            continue
        geom_type = int(model.geom_type[geom_id])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            half = np.asarray(model.geom_size[geom_id], dtype=float)
            counts = np.maximum(1, np.ceil((2.0 * half) / 0.024).astype(int))
            step = 2.0 * half / counts
            rotation = _quat_matrix(model.geom_quat[geom_id])
            body_name = model.body(body_id).name
            axes = [
                np.linspace(-half[axis] + 0.5 * step[axis], half[axis] - 0.5 * step[axis], counts[axis])
                for axis in range(3)
            ]
            cover_radius = float(np.linalg.norm(0.5 * step))
            for x in axes[0]:
                for y in axes[1]:
                    for z in axes[2]:
                        local = model.geom_pos[geom_id] + rotation @ np.array([x, y, z])
                        certificate.append(
                            CertificateSphere(
                                body_id=body_id,
                                body_name=body_name,
                                source_geom_id=geom_id,
                                local_center=local.copy(),
                                radius=cover_radius,
                            )
                        )
            continue
        if geom_type not in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            continue
        radius = float(model.geom_size[geom_id, 0])
        half_length = float(model.geom_size[geom_id, 1])
        count = max(2, int(np.ceil((2.0 * half_length) / axial_spacing)) + 1)
        axial = np.linspace(-half_length, half_length, count)
        spacing = 0.0 if count == 1 else float(axial[1] - axial[0])
        cover_radius = float(np.sqrt(radius * radius + 0.25 * spacing * spacing))
        rotation = _quat_matrix(model.geom_quat[geom_id])
        body_name = model.body(body_id).name
        for z in axial:
            local = model.geom_pos[geom_id] + rotation @ np.array([0.0, 0.0, z])
            certificate.append(
                CertificateSphere(
                    body_id=body_id,
                    body_name=body_name,
                    source_geom_id=geom_id,
                    local_center=local.copy(),
                    radius=cover_radius,
                )
            )
    return certificate


def certificate_world_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    certificate: list[CertificateSphere],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return robot sphere centers, translational Jacobians, and radii."""

    count = len(certificate)
    positions = np.zeros((count, 3))
    jacobians = np.zeros((count, 3, model.nv))
    radii = np.fromiter((sphere.radius for sphere in certificate), dtype=float)
    jac_rot = np.zeros((3, model.nv))
    for index, sphere in enumerate(certificate):
        rotation = data.xmat[sphere.body_id].reshape(3, 3)
        position = data.xpos[sphere.body_id] + rotation @ sphere.local_center
        positions[index] = position
        mujoco.mj_jac(model, data, jacobians[index], jac_rot, position, sphere.body_id)
    return positions, jacobians, radii


def certificate_world_positions(
    data: mujoco.MjData,
    certificate: list[CertificateSphere],
) -> np.ndarray:
    """Return only sphere centers for fast safety checks (no Jacobians)."""

    positions = np.empty((len(certificate), 3), dtype=float)
    for index, sphere in enumerate(certificate):
        rotation = data.xmat[sphere.body_id].reshape(3, 3)
        positions[index] = (
            data.xpos[sphere.body_id] + rotation @ sphere.local_center
        )
    return positions


def sample_box_sphere_tree(
    boxes: tuple[BoxObstacle, ...],
    cell_size: float = 0.075,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conservatively cover every obstacle box with a uniform sphere-tree level.

    Each voxel cell is enclosed by one sphere of radius equal to the cell's
    half-diagonal.  The union of spheres therefore contains the full box, which
    is conservative in exactly the way described around LiuQP Fig. 8.
    """

    centers: list[np.ndarray] = []
    radii: list[float] = []
    owners: list[int] = []
    for owner, box in enumerate(boxes):
        half = np.asarray(box.half_size, dtype=float)
        count = np.maximum(1, np.ceil((2.0 * half) / cell_size).astype(int))
        step = 2.0 * half / count
        axes = [
            np.linspace(-half[axis] + 0.5 * step[axis], half[axis] - 0.5 * step[axis], count[axis])
            for axis in range(3)
        ]
        cell_radius = float(0.5 * np.linalg.norm(step))
        for x in axes[0]:
            for y in axes[1]:
                for z in axes[2]:
                    centers.append(np.asarray(box.center) + np.array([x, y, z]))
                    radii.append(cell_radius)
                    owners.append(owner)
    return np.asarray(centers), np.asarray(radii), np.asarray(owners, dtype=int)
