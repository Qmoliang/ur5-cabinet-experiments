"""Wrist-mounted MuJoCo depth sensing for the UR5 LiuQP experiments.

The controller-facing output is an unlabelled XYZ cloud.  MuJoCo segmentation
IDs are retained only for simulation audits; proxy construction never receives
box names, analytic wall normals, or ground-truth obstacle dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np


@dataclass(frozen=True)
class DepthObservation:
    camera_name: str
    camera_position: np.ndarray
    camera_rotation: np.ndarray
    points: np.ndarray
    geom_ids: np.ndarray
    optical_depths: np.ndarray
    sample_radii: np.ndarray
    maximum_pixel_radius: float
    sample_uncertainty_shapes: np.ndarray | None = None
    # True for an observed environment surface endpoint.  False denotes a
    # robot-self return: its ray prefix is observed free, but the endpoint is
    # not an environment obstacle.  None preserves legacy all-environment
    # observations.
    endpoint_is_occupied: np.ndarray | None = None


def environment_endpoint_mask(observation: DepthObservation) -> np.ndarray:
    if observation.endpoint_is_occupied is None:
        return np.ones(len(observation.points), dtype=bool)
    mask = np.asarray(observation.endpoint_is_occupied, dtype=bool).reshape(-1)
    if len(mask) != len(observation.points):
        raise ValueError("endpoint_is_occupied must match depth point count")
    return mask


@dataclass(frozen=True)
class FusedDepthCloud:
    raw_points: np.ndarray
    filtered_points: np.ndarray
    voxel_residuals: np.ndarray
    filtered_cover_radii: np.ndarray
    cover_radius: float
    observation_count: int
    camera_positions: np.ndarray


@dataclass(frozen=True)
class ActiveWristScan:
    observations: list[DepthObservation]
    q_trajectory: np.ndarray
    duration: float
    penetrating_steps: int


class UR5MountedDepthCamera:
    """Metric depth renderer whose pose follows a UR5 body-mounted camera."""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_names: tuple[str, ...] = ("ur5_depth_wrist",),
        width: int = 160,
        height: int = 120,
        pixel_stride: int = 2,
        minimum_range: float = 0.035,
        maximum_range: float = 1.60,
        depth_noise_std: float = 0.0,
        optical_depth_error_bound: float = 0.0,
        seed: int = 0,
        occluding_self_filter: bool = False,
        native_raycast: bool = True,
    ) -> None:
        if width <= 0 or height <= 0 or pixel_stride <= 0:
            raise ValueError("image dimensions and pixel_stride must be positive")
        self.model = model
        self.camera_names = tuple(camera_names)
        self.camera_ids = tuple(model.camera(name).id for name in self.camera_names)
        self.width = int(width)
        self.height = int(height)
        self.pixel_stride = int(pixel_stride)
        self.minimum_range = float(minimum_range)
        self.maximum_range = float(maximum_range)
        self.depth_noise_std = float(depth_noise_std)
        self.optical_depth_error_bound = float(optical_depth_error_bound)
        if self.optical_depth_error_bound < 0.0:
            raise ValueError("optical_depth_error_bound must be nonnegative")
        self.occluding_self_filter = bool(occluding_self_filter)
        self.native_raycast = bool(native_raycast)
        self.rng = np.random.default_rng(seed)
        # MuJoCo's default offscreen MSAA blends reverse-Z depth across
        # sub-pixel silhouettes while the integer segmentation buffer retains
        # one object ID.  Back-projecting that mixed value along the pixel-
        # center ray can create a point many centimetres in front of every
        # physical geom.  A metric depth sensor must use one depth sample per
        # pixel; skipped-pixel footprint is already covered explicitly by the
        # directional uncertainty ellipsoid below.
        self.model.vis.quality.offsamples = 1
        # The XML default IPD is intended for stereo visualization.  This
        # class exposes one calibrated pinhole depth frame per named camera
        # and back-projects from data.cam_xpos, so the render eye and that
        # optical center must coincide.
        for camera_id in self.camera_ids:
            self.model.cam_ipd[camera_id] = 0.0
        self.renderer = (
            None
            if self.native_raycast
            else mujoco.Renderer(model, height=height, width=width)
        )
        self.scene_option = mujoco.MjvOption()
        # UR5 primitive collision geoms and the retrieval tool use group 1.
        # Hiding this group is a kinematics-based self filter, not privileged
        # obstacle segmentation. Environment geoms remain visible in group 0.
        # Legacy experiments hid group 1 before rendering. The frozen
        # incremental-drawer protocol leaves it visible so the robot casts a
        # real depth shadow, then removes robot-hit pixels by geom id.
        self.scene_option.geomgroup[1] = 1 if self.occluding_self_filter else 0
        self.robot_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) != 0
        }
        # Sites are annotations (target marker, attachment marker), not
        # collision geometry.  They must never enter the depth obstacle cloud.
        self.scene_option.sitegroup[:] = 0
        self.sample_v = np.arange(pixel_stride // 2, height, pixel_stride)
        self.sample_u = np.arange(pixel_stride // 2, width, pixel_stride)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()

    def __enter__(self) -> "UR5MountedDepthCamera":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def capture(self, data: mujoco.MjData) -> list[DepthObservation]:
        observations: list[DepthObservation] = []
        cy = 0.5 * (self.height - 1)
        cx = 0.5 * (self.width - 1)

        for name, camera_id in zip(self.camera_names, self.camera_ids):
            camera_position = data.cam_xpos[camera_id].copy()
            camera_rotation = data.cam_xmat[camera_id].reshape(3, 3).copy()
            fovy = math.radians(float(self.model.cam_fovy[camera_id]))
            fy = 0.5 * self.height / math.tan(0.5 * fovy)
            fx = fy

            grid_u, grid_v = np.meshgrid(self.sample_u, self.sample_v)
            flat_u = grid_u.reshape(-1)
            flat_v = grid_v.reshape(-1)
            local_directions = np.column_stack(
                [
                    (flat_u - cx) / fx,
                    -(flat_v - cy) / fy,
                    -np.ones(len(flat_u)),
                ]
            )
            local_norms = np.linalg.norm(local_directions, axis=1)
            ray_directions = local_directions @ camera_rotation.T
            if self.native_raycast:
                ray_units = np.ascontiguousarray(
                    ray_directions / local_norms[:, None], dtype=np.float64
                )
                radial_depths = np.full(len(ray_units), -1.0, dtype=np.float64)
                geom_ids = np.full(len(ray_units), -1, dtype=np.int32)
                mujoco.mj_multiRay(
                    self.model,
                    data,
                    np.ascontiguousarray(camera_position, dtype=np.float64),
                    # The Python binding mirrors MuJoCo's C API and expects
                    # the nray consecutive xyz vectors as one flat buffer.
                    ray_units.reshape(-1),
                    np.asarray(self.scene_option.geomgroup, dtype=np.uint8),
                    True,
                    -1,
                    geom_ids,
                    radial_depths,
                    None,
                    len(ray_units),
                    self.maximum_range,
                )
                optical_depths = radial_depths / local_norms
                finite = radial_depths >= 0.0
            else:
                if self.renderer is None:
                    raise RuntimeError("GPU depth renderer was not constructed")
                self.renderer.enable_depth_rendering()
                self.renderer.update_scene(
                    data, camera=name, scene_option=self.scene_option
                )
                depth = np.asarray(self.renderer.render(), dtype=float).copy()
                self.renderer.disable_depth_rendering()
                self.renderer.enable_segmentation_rendering()
                self.renderer.update_scene(
                    data, camera=name, scene_option=self.scene_option
                )
                segmentation = np.asarray(self.renderer.render()).copy()
                self.renderer.disable_segmentation_rendering()
                optical_depths = depth[flat_v, flat_u].astype(float, copy=True)
                finite = np.isfinite(optical_depths)
                radial_depths = optical_depths * local_norms
                object_types = segmentation[flat_v, flat_u, 1].astype(np.int32)
                geom_ids = np.where(
                    object_types == int(mujoco.mjtObj.mjOBJ_GEOM),
                    segmentation[flat_v, flat_u, 0].astype(np.int32),
                    -1,
                )
            if self.depth_noise_std > 0.0 and np.any(finite):
                optical_depths[finite] += self.rng.normal(
                    0.0, self.depth_noise_std, size=int(np.count_nonzero(finite))
                )
                radial_depths = optical_depths * local_norms
            keep = finite
            keep &= optical_depths > self.minimum_range
            keep &= radial_depths < self.maximum_range
            flat_u = flat_u[keep]
            flat_v = flat_v[keep]
            optical_depths = optical_depths[keep]
            local_directions = local_directions[keep]
            geom_ids = geom_ids[keep]

            ray_directions = local_directions @ camera_rotation.T
            points = camera_position[None, :] + (
                optical_depths[:, None] * ray_directions
            )
            if self.occluding_self_filter:
                robot_ids = np.fromiter(
                    self.robot_geom_ids, dtype=np.int32, count=len(self.robot_geom_ids)
                )
                endpoint_is_occupied = ~np.isin(geom_ids, robot_ids)
            else:
                endpoint_is_occupied = np.ones(len(geom_ids), dtype=bool)

            # The skipped-pixel cell and bounded optical-depth error form a
            # two/three-generator zonotope.  Batch construction evaluates the
            # same E(m G G') certificate as the scalar derivation.
            depth_bound = self.optical_depth_error_bound
            half_pixels = 0.5 * self.pixel_stride
            tangent_x = camera_rotation[:, 0]
            tangent_y = camera_rotation[:, 1]
            tangent_scale_x = (
                (optical_depths + depth_bound) * half_pixels / fx
            )
            tangent_scale_y = (
                (optical_depths + depth_bound) * half_pixels / fy
            )
            generator_x = tangent_scale_x[:, None] * tangent_x[None, :]
            generator_y = tangent_scale_y[:, None] * tangent_y[None, :]
            generator_count = 2 + int(depth_bound > 0.0)
            sample_uncertainty_shapes = generator_count * (
                np.einsum("ni,nj->nij", generator_x, generator_x)
                + np.einsum("ni,nj->nij", generator_y, generator_y)
            )
            if depth_bound > 0.0:
                generator_depth = depth_bound * ray_directions
                sample_uncertainty_shapes += generator_count * np.einsum(
                    "ni,nj->nij", generator_depth, generator_depth
                )
            sample_uncertainty_shapes = 0.5 * (
                sample_uncertainty_shapes
                + np.swapaxes(sample_uncertainty_shapes, 1, 2)
            )
            sample_radii = np.sqrt(
                np.maximum(
                    np.linalg.eigvalsh(sample_uncertainty_shapes)[:, -1], 0.0
                )
            )
            maximum_pixel_radius = (
                0.0 if not len(sample_radii) else float(np.max(sample_radii))
            )

            observations.append(
                DepthObservation(
                    camera_name=name,
                    camera_position=camera_position,
                    camera_rotation=camera_rotation,
                    points=np.asarray(points, dtype=float).reshape(-1, 3),
                    geom_ids=np.asarray(geom_ids, dtype=np.int32),
                    optical_depths=np.asarray(optical_depths, dtype=float),
                    sample_radii=np.asarray(sample_radii, dtype=float),
                    maximum_pixel_radius=float(maximum_pixel_radius),
                    sample_uncertainty_shapes=np.asarray(
                        sample_uncertainty_shapes, dtype=float
                    ).reshape(-1, 3, 3),
                    endpoint_is_occupied=np.asarray(endpoint_is_occupied, dtype=bool),
                )
            )

        return observations


def center_voxel_filter_with_residual(
    points: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """CenterVox plus the exact raw-to-representative residual per voxel."""

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return points.copy(), np.zeros(0)
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    origin = np.min(points, axis=0) - 1.0e-12
    keys = np.floor((points - origin) / voxel_size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys = keys[order]
    ordered_points = points[order]
    selected: list[np.ndarray] = []
    residuals: list[float] = []
    start = 0
    while start < len(ordered_points):
        end = start + 1
        while end < len(ordered_points) and np.array_equal(keys[end], keys[start]):
            end += 1
        voxel_center = origin + (keys[start].astype(float) + 0.5) * voxel_size
        candidates = ordered_points[start:end]
        selected_index = int(
            np.argmin(np.linalg.norm(candidates - voxel_center, axis=1))
        )
        representative = candidates[selected_index]
        selected.append(representative)
        residuals.append(
            float(np.max(np.linalg.norm(candidates - representative, axis=1)))
        )
        start = end
    return np.asarray(selected), np.asarray(residuals)


def center_voxel_filter_with_uncertainty(
    points: np.ndarray,
    point_radii: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CenterVox with a per-representative cover for uncertain raw samples."""

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    point_radii = np.asarray(point_radii, dtype=float).reshape(-1)
    if len(points) != len(point_radii):
        raise ValueError("point_radii must contain one value per point")
    if len(points) == 0:
        return points.copy(), np.zeros(0), np.zeros(0)
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    origin = np.min(points, axis=0) - 1.0e-12
    keys = np.floor((points - origin) / voxel_size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys = keys[order]
    ordered_points = points[order]
    ordered_radii = point_radii[order]
    selected: list[np.ndarray] = []
    residuals: list[float] = []
    cover_radii: list[float] = []
    start = 0
    while start < len(ordered_points):
        end = start + 1
        while end < len(ordered_points) and np.array_equal(keys[end], keys[start]):
            end += 1
        voxel_center = origin + (keys[start].astype(float) + 0.5) * voxel_size
        candidates = ordered_points[start:end]
        candidate_radii = ordered_radii[start:end]
        selected_index = int(
            np.argmin(np.linalg.norm(candidates - voxel_center, axis=1))
        )
        representative = candidates[selected_index]
        distances = np.linalg.norm(candidates - representative, axis=1)
        selected.append(representative)
        residuals.append(float(np.max(distances)))
        cover_radii.append(float(np.max(distances + candidate_radii)))
        start = end
    return (
        np.asarray(selected),
        np.asarray(residuals),
        np.asarray(cover_radii),
    )


def fuse_depth_observations(
    observations: list[DepthObservation],
    voxel_size: float = 0.006,
    calibrated_depth_bound: float = 0.003,
    workspace_lower: np.ndarray | None = None,
    workspace_upper: np.ndarray | None = None,
    accepted_geom_ids: set[int] | None = None,
) -> FusedDepthCloud:
    """Fuse views and return a conservative observed-surface error radius."""

    batches: list[np.ndarray] = []
    radius_batches: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    maximum_pixel_radius = 0.0
    for observation in observations:
        endpoint_mask = environment_endpoint_mask(observation)
        points = observation.points[endpoint_mask]
        geom_ids = observation.geom_ids[endpoint_mask]
        radii = (
            observation.sample_radii[endpoint_mask]
            + float(calibrated_depth_bound)
        )
        if accepted_geom_ids is not None and len(points):
            keep = np.fromiter(
                (int(value) in accepted_geom_ids for value in geom_ids),
                dtype=bool,
                count=len(observation.geom_ids),
            )
            points = points[keep]
            radii = radii[keep]
        if workspace_lower is not None and len(points):
            keep = np.all(points >= np.asarray(workspace_lower), axis=1)
            points = points[keep]
            radii = radii[keep]
        if workspace_upper is not None and len(points):
            keep = np.all(points <= np.asarray(workspace_upper), axis=1)
            points = points[keep]
            radii = radii[keep]
        if len(points):
            batches.append(points)
            radius_batches.append(radii)
        positions.append(observation.camera_position)
        maximum_pixel_radius = max(
            maximum_pixel_radius, observation.maximum_pixel_radius
        )
    raw = np.vstack(batches) if batches else np.zeros((0, 3))
    raw_radii = np.concatenate(radius_batches) if radius_batches else np.zeros(0)
    filtered, residuals, filtered_cover_radii = (
        center_voxel_filter_with_uncertainty(raw, raw_radii, voxel_size)
    )
    cover_radius = (
        float(np.max(filtered_cover_radii))
        if len(filtered_cover_radii)
        else float(calibrated_depth_bound)
    )
    return FusedDepthCloud(
        raw_points=raw,
        filtered_points=filtered,
        voxel_residuals=residuals,
        filtered_cover_radii=filtered_cover_radii,
        cover_radius=float(cover_radius),
        observation_count=len(observations),
        camera_positions=np.asarray(positions, dtype=float).reshape(-1, 3),
    )


def wrist_scan_configurations(
    q0: np.ndarray,
    wrist_1_offsets: tuple[float, ...] = (-0.24, 0.0, 0.24),
    wrist_2_offsets: tuple[float, ...] = (-0.55, 0.0, 0.55),
) -> list[np.ndarray]:
    """Small pan/pitch scan poses around a collision-checked start posture."""

    q0 = np.asarray(q0, dtype=float)
    if q0.shape != (6,):
        raise ValueError("UR5 scan seed must contain six joint positions")
    configurations: list[np.ndarray] = []
    for wrist_1 in wrist_1_offsets:
        for wrist_2 in wrist_2_offsets:
            q = q0.copy()
            q[3] += wrist_1
            q[4] += wrist_2
            configurations.append(q)
    return configurations


def acquire_active_wrist_scan(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: UR5MountedDepthCamera,
    configurations: list[np.ndarray],
    *,
    dt: float = 0.02,
    maximum_joint_speed: float = 0.45,
    return_to_start: bool = True,
) -> ActiveWristScan:
    """Execute a bounded-speed wrist scan and capture at every reached pose.

    The scan is kinematic, matching the sequential velocity-level LiuQP
    experiments.  Every intermediate pose is forwarded through MuJoCo and
    audited for penetration before an observation is accepted.
    """

    if dt <= 0.0 or maximum_joint_speed <= 0.0:
        raise ValueError("dt and maximum_joint_speed must be positive")
    start = np.asarray(data.qpos[:6], dtype=float).copy()
    targets = [np.asarray(q, dtype=float).copy() for q in configurations]
    if return_to_start:
        targets.append(start.copy())
    trajectory = [start.copy()]
    observations: list[DepthObservation] = []
    penetrating_steps = 0
    current = start.copy()
    for target in targets:
        if target.shape != (6,):
            raise ValueError("each scan configuration must have six joints")
        while float(np.max(np.abs(target - current))) > 1.0e-10:
            step = np.clip(
                target - current,
                -maximum_joint_speed * dt,
                maximum_joint_speed * dt,
            )
            current = current + step
            data.qpos[:6] = current
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            if any(
                float(data.contact[index].dist) < -1.0e-8
                for index in range(data.ncon)
            ):
                penetrating_steps += 1
            trajectory.append(current.copy())
        observations.extend(camera.capture(data))
    if penetrating_steps:
        raise RuntimeError(
            f"active wrist scan generated {penetrating_steps} penetrating steps"
        )
    return ActiveWristScan(
        observations=observations,
        q_trajectory=np.asarray(trajectory),
        duration=float((len(trajectory) - 1) * dt),
        penetrating_steps=int(penetrating_steps),
    )
