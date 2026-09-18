"""Extracted original v4.3 ellipsoid dependency; see docs/source_manifest.json."""
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
    endpoint_is_occupied: np.ndarray | None = None

def environment_endpoint_mask(observation: DepthObservation) -> np.ndarray:
    if observation.endpoint_is_occupied is None:
        return np.ones(len(observation.points), dtype=bool)
    mask = np.asarray(observation.endpoint_is_occupied, dtype=bool).reshape(-1)
    if len(mask) != len(observation.points):
        raise ValueError('endpoint_is_occupied must match depth point count')
    return mask

class UR5MountedDepthCamera:

    def __init__(self, model: mujoco.MjModel, camera_names: tuple[str, ...]=('ur5_depth_wrist',), width: int=160, height: int=120, pixel_stride: int=2, minimum_range: float=0.035, maximum_range: float=1.6, depth_noise_std: float=0.0, optical_depth_error_bound: float=0.0, seed: int=0, occluding_self_filter: bool=False, native_raycast: bool=True) -> None:
        if width <= 0 or height <= 0 or pixel_stride <= 0:
            raise ValueError('image dimensions and pixel_stride must be positive')
        self.model = model
        self.camera_names = tuple(camera_names)
        self.camera_ids = tuple((model.camera(name).id for name in self.camera_names))
        self.width = int(width)
        self.height = int(height)
        self.pixel_stride = int(pixel_stride)
        self.minimum_range = float(minimum_range)
        self.maximum_range = float(maximum_range)
        self.depth_noise_std = float(depth_noise_std)
        self.optical_depth_error_bound = float(optical_depth_error_bound)
        if self.optical_depth_error_bound < 0.0:
            raise ValueError('optical_depth_error_bound must be nonnegative')
        self.occluding_self_filter = bool(occluding_self_filter)
        self.native_raycast = bool(native_raycast)
        self.rng = np.random.default_rng(seed)
        self.model.vis.quality.offsamples = 1
        for camera_id in self.camera_ids:
            self.model.cam_ipd[camera_id] = 0.0
        self.renderer = None if self.native_raycast else mujoco.Renderer(model, height=height, width=width)
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[1] = 1 if self.occluding_self_filter else 0
        self.robot_geom_ids = {geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) != 0}
        self.scene_option.sitegroup[:] = 0
        self.sample_v = np.arange(pixel_stride // 2, height, pixel_stride)
        self.sample_u = np.arange(pixel_stride // 2, width, pixel_stride)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()

    def __enter__(self) -> 'UR5MountedDepthCamera':
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
            local_directions = np.column_stack([(flat_u - cx) / fx, -(flat_v - cy) / fy, -np.ones(len(flat_u))])
            local_norms = np.linalg.norm(local_directions, axis=1)
            ray_directions = local_directions @ camera_rotation.T
            if self.native_raycast:
                ray_units = np.ascontiguousarray(ray_directions / local_norms[:, None], dtype=np.float64)
                radial_depths = np.full(len(ray_units), -1.0, dtype=np.float64)
                geom_ids = np.full(len(ray_units), -1, dtype=np.int32)
                mujoco.mj_multiRay(self.model, data, np.ascontiguousarray(camera_position, dtype=np.float64), ray_units.reshape(-1), np.asarray(self.scene_option.geomgroup, dtype=np.uint8), True, -1, geom_ids, radial_depths, None, len(ray_units), self.maximum_range)
                optical_depths = radial_depths / local_norms
                finite = radial_depths >= 0.0
            else:
                if self.renderer is None:
                    raise RuntimeError('GPU depth renderer was not constructed')
                self.renderer.enable_depth_rendering()
                self.renderer.update_scene(data, camera=name, scene_option=self.scene_option)
                depth = np.asarray(self.renderer.render(), dtype=float).copy()
                self.renderer.disable_depth_rendering()
                self.renderer.enable_segmentation_rendering()
                self.renderer.update_scene(data, camera=name, scene_option=self.scene_option)
                segmentation = np.asarray(self.renderer.render()).copy()
                self.renderer.disable_segmentation_rendering()
                optical_depths = depth[flat_v, flat_u].astype(float, copy=True)
                finite = np.isfinite(optical_depths)
                radial_depths = optical_depths * local_norms
                object_types = segmentation[flat_v, flat_u, 1].astype(np.int32)
                geom_ids = np.where(object_types == int(mujoco.mjtObj.mjOBJ_GEOM), segmentation[flat_v, flat_u, 0].astype(np.int32), -1)
            if self.depth_noise_std > 0.0 and np.any(finite):
                optical_depths[finite] += self.rng.normal(0.0, self.depth_noise_std, size=int(np.count_nonzero(finite)))
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
            points = camera_position[None, :] + optical_depths[:, None] * ray_directions
            if self.occluding_self_filter:
                robot_ids = np.fromiter(self.robot_geom_ids, dtype=np.int32, count=len(self.robot_geom_ids))
                endpoint_is_occupied = ~np.isin(geom_ids, robot_ids)
            else:
                endpoint_is_occupied = np.ones(len(geom_ids), dtype=bool)
            depth_bound = self.optical_depth_error_bound
            half_pixels = 0.5 * self.pixel_stride
            tangent_x = camera_rotation[:, 0]
            tangent_y = camera_rotation[:, 1]
            tangent_scale_x = (optical_depths + depth_bound) * half_pixels / fx
            tangent_scale_y = (optical_depths + depth_bound) * half_pixels / fy
            generator_x = tangent_scale_x[:, None] * tangent_x[None, :]
            generator_y = tangent_scale_y[:, None] * tangent_y[None, :]
            generator_count = 2 + int(depth_bound > 0.0)
            sample_uncertainty_shapes = generator_count * (np.einsum('ni,nj->nij', generator_x, generator_x) + np.einsum('ni,nj->nij', generator_y, generator_y))
            if depth_bound > 0.0:
                generator_depth = depth_bound * ray_directions
                sample_uncertainty_shapes += generator_count * np.einsum('ni,nj->nij', generator_depth, generator_depth)
            sample_uncertainty_shapes = 0.5 * (sample_uncertainty_shapes + np.swapaxes(sample_uncertainty_shapes, 1, 2))
            sample_radii = np.sqrt(np.maximum(np.linalg.eigvalsh(sample_uncertainty_shapes)[:, -1], 0.0))
            maximum_pixel_radius = 0.0 if not len(sample_radii) else float(np.max(sample_radii))
            observations.append(DepthObservation(camera_name=name, camera_position=camera_position, camera_rotation=camera_rotation, points=np.asarray(points, dtype=float).reshape(-1, 3), geom_ids=np.asarray(geom_ids, dtype=np.int32), optical_depths=np.asarray(optical_depths, dtype=float), sample_radii=np.asarray(sample_radii, dtype=float), maximum_pixel_radius=float(maximum_pixel_radius), sample_uncertainty_shapes=np.asarray(sample_uncertainty_shapes, dtype=float).reshape(-1, 3, 3), endpoint_is_occupied=np.asarray(endpoint_is_occupied, dtype=bool)))
        return observations
