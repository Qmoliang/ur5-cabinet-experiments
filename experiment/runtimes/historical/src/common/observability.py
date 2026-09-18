"""Extracted original v4.3 ellipsoid dependency; see docs/source_manifest.json."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import mujoco
import numpy as np
from scipy.spatial import cKDTree
from robot import BoxObstacle, DT, SceneDefinition, build_model, build_robot_certificate, certificate_world_positions, set_configuration
from proxy_geometry import PointCloudProxySet, sample_box_surface_point_cloud

@dataclass(frozen=True)
class ObservedPointSnapshot:
    publish_cycle: int
    source_cycle: int
    points: np.ndarray
    cover_radii: np.ndarray

@dataclass(frozen=True)
class ObservabilityReport:
    audit_method: str
    passed: bool
    event_code: str
    observation_distance_m: float
    guard_time_s: float
    guard_cycles: int
    truth_surface_spacing_m: float
    truth_lipschitz_correction_m: float
    truth_surface_samples: int
    ever_at_risk_samples: int
    observed_before_deadline_samples: int
    late_samples: int
    never_observed_samples: int
    maximum_observation_lateness_cycles: int
    first_violation_cycle: int | None
    first_violation_point_m: list[float] | None
    first_violation_box: str | None
    first_violation_kind: str | None
    first_violation_seen_cycle: int | None
    first_violation_deadline_cycle: int | None
    at_risk_samples_by_box: dict[str, int]
    violation_samples_by_box: dict[str, int]
    late_samples_by_box: dict[str, int]
    never_observed_samples_by_box: dict[str, int]
    late_sample_bounds_by_box_m: dict[str, dict[str, list[float]] | None]
    never_observed_sample_bounds_by_box_m: dict[str, dict[str, list[float]] | None]
    snapshot_count: int
    truth_used_after_control_only: bool = True
    evaluator_feedback_to_control: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

@dataclass(frozen=True)
class BoxSurfaceRange:
    box: BoxObstacle
    start: int
    stop: int

def snapshot_from_proxies(proxies: PointCloudProxySet, *, publish_cycle: int, source_cycle: int, centervox_size: float) -> ObservedPointSnapshot:
    points = np.asarray(proxies.filtered_points, dtype=float).reshape(-1, 3)
    clusters = np.asarray(proxies.filtered_cluster_indices, dtype=np.int64).reshape(-1)
    if len(points) != len(clusters):
        raise ValueError('filtered points and cluster indices must match')
    if proxies.filtered_uncertainty_shapes is None:
        measurement = np.zeros(len(points), dtype=float)
    else:
        shapes = np.asarray(proxies.filtered_uncertainty_shapes, dtype=float).reshape(-1, 3, 3)
        if len(shapes) != len(points):
            raise ValueError('filtered uncertainty shapes must match points')
        measurement = np.sqrt(np.maximum(np.linalg.eigvalsh(shapes)[:, -1], 0.0))
    center_cover = math.sqrt(3.0) * float(centervox_size)
    cover = measurement + center_cover
    return ObservedPointSnapshot(publish_cycle=int(publish_cycle), source_cycle=int(source_cycle), points=points.copy(), cover_radii=cover.copy())

def load_observability_snapshots(path: str) -> list[ObservedPointSnapshot]:
    with np.load(path) as archive:
        publish = np.asarray(archive['publish_cycles'], dtype=np.int64)
        source = np.asarray(archive['source_cycles'], dtype=np.int64)
        offsets = np.asarray(archive['offsets'], dtype=np.int64)
        points = np.asarray(archive['points'], dtype=float)
        cover = np.asarray(archive['cover_radii'], dtype=float)
    if len(offsets) != len(publish) + 1 or len(source) != len(publish):
        raise ValueError('invalid observability snapshot offsets')
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(points):
        raise ValueError('observability snapshot point range is inconsistent')
    if len(points) != len(cover):
        raise ValueError('observability point and cover arrays must match')
    return [ObservedPointSnapshot(publish_cycle=int(publish[index]), source_cycle=int(source[index]), points=points[offsets[index]:offsets[index + 1]].copy(), cover_radii=cover[offsets[index]:offsets[index + 1]].copy()) for index in range(len(publish))]

def _first_seen_cycles(truth_points: np.ndarray, snapshots: list[ObservedPointSnapshot]) -> np.ndarray:
    truth_tree = cKDTree(truth_points)
    first_seen = np.full(len(truth_points), np.iinfo(np.int32).max, dtype=np.int32)
    for snapshot in sorted(snapshots, key=lambda item: item.publish_cycle):
        if not len(snapshot.points):
            continue
        neighborhoods = truth_tree.query_ball_point(np.asarray(snapshot.points, dtype=float), np.asarray(snapshot.cover_radii, dtype=float), workers=-1)
        nonempty = [np.asarray(item, dtype=np.int64) for item in neighborhoods if len(item)]
        if not nonempty:
            continue
        covered = np.unique(np.concatenate(nonempty))
        unseen = first_seen[covered] == np.iinfo(np.int32).max
        first_seen[covered[unseen]] = int(snapshot.publish_cycle)
    return first_seen

def _sample_truth_surfaces(boxes: tuple[BoxObstacle, ...], spacing: float) -> tuple[np.ndarray, list[BoxSurfaceRange]]:
    batches: list[np.ndarray] = []
    ranges: list[BoxSurfaceRange] = []
    start = 0
    for box in boxes:
        points = sample_box_surface_point_cloud((box,), spacing=spacing)
        stop = start + len(points)
        batches.append(points)
        ranges.append(BoxSurfaceRange(box=box, start=start, stop=stop))
        start = stop
    if not batches:
        return (np.empty((0, 3), dtype=float), ranges)
    return (np.concatenate(batches, axis=0), ranges)

def _nearest_surface_distance(center: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    clamped = np.clip(center, lower, upper)
    outside_distance = float(np.linalg.norm(center - clamped))
    if outside_distance > 0.0:
        return outside_distance
    face_distances = np.concatenate((center - lower, upper - center))
    return float(np.min(face_distances))

def _nearest_surface_patch_indices(center: np.ndarray, box: BoxObstacle, box_points: np.ndarray, tree: cKDTree, correction: float) -> np.ndarray:
    box_center = np.asarray(box.center, dtype=float)
    half = np.asarray(box.half_size, dtype=float)
    distance = _nearest_surface_distance(center, box_center - half, box_center + half)
    indices = tree.query_ball_point(center, distance + float(correction))
    if indices:
        return np.asarray(indices, dtype=np.int64)
    _, nearest = tree.query(center, k=1)
    return np.asarray([int(nearest)], dtype=np.int64)

def _filter_risk_patch_indices(center: np.ndarray, radius: float, box_points: np.ndarray, indices: np.ndarray, observation_distance: float, sampling_correction: float) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if not len(indices):
        return indices
    distances = np.linalg.norm(np.asarray(box_points, dtype=float)[indices] - np.asarray(center, dtype=float)[None, :], axis=1)
    limit = float(radius) + float(observation_distance) + float(sampling_correction) + 1e-12
    return indices[distances <= limit]

def _first_risk_cycles(scene: SceneDefinition, q_trajectory: np.ndarray, truth_points: np.ndarray, box_ranges: list[BoxSurfaceRange], observation_distance: float, lipschitz_correction: float) -> np.ndarray:
    model = build_model(scene)
    data = mujoco.MjData(model)
    robot = build_robot_certificate(model)
    radii = np.asarray([sphere.radius for sphere in robot], dtype=float)
    trees = [cKDTree(truth_points[item.start:item.stop]) for item in box_ranges]
    lowers = np.asarray([np.asarray(item.box.center) - np.asarray(item.box.half_size) for item in box_ranges], dtype=float)
    uppers = np.asarray([np.asarray(item.box.center) + np.asarray(item.box.half_size) for item in box_ranges], dtype=float)
    first_risk = np.full(len(truth_points), np.iinfo(np.int32).max, dtype=np.int32)
    query_extra = float(observation_distance) + float(lipschitz_correction)
    for cycle, q in enumerate(np.asarray(q_trajectory, dtype=float)):
        set_configuration(model, data, q)
        centers = certificate_world_positions(data, robot)
        candidates: list[np.ndarray] = []
        for center, radius in zip(centers, radii):
            outside_delta = np.maximum(np.maximum(lowers - center, center - uppers), 0.0)
            solid_distances = np.linalg.norm(outside_delta, axis=1)
            risk_boxes = np.flatnonzero(solid_distances <= float(radius) + query_extra)
            for box_index in risk_boxes:
                item = box_ranges[int(box_index)]
                local = _nearest_surface_patch_indices(center, item.box, truth_points[item.start:item.stop], trees[int(box_index)], lipschitz_correction)
                local = _filter_risk_patch_indices(center, float(radius), truth_points[item.start:item.stop], local, observation_distance, lipschitz_correction)
                if len(local):
                    candidates.append(local + item.start)
        if not candidates:
            continue
        nearby = np.unique(np.concatenate(candidates))
        unresolved = first_risk[nearby] == np.iinfo(np.int32).max
        first_risk[nearby[unresolved]] = int(cycle)
    return first_risk

def evaluate_observed_before_risk(scene: SceneDefinition, q_trajectory: np.ndarray, snapshots: list[ObservedPointSnapshot], *, observation_distance: float=0.1, guard_time: float=0.1, truth_surface_spacing: float=0.012) -> ObservabilityReport:
    if observation_distance <= 0.0 or guard_time < 0.0:
        raise ValueError('observation distance must be positive and guard nonnegative')
    if truth_surface_spacing <= 0.0:
        raise ValueError('truth surface spacing must be positive')
    truth, box_ranges = _sample_truth_surfaces(scene.boxes, spacing=float(truth_surface_spacing))
    correction = math.sqrt(2.0) * float(truth_surface_spacing) / 2.0
    first_seen = _first_seen_cycles(truth, snapshots)
    first_risk = _first_risk_cycles(scene, q_trajectory, truth, box_ranges, float(observation_distance), correction)
    sentinel = np.iinfo(np.int32).max
    risk_mask = first_risk != sentinel
    risk_indices = np.flatnonzero(risk_mask)
    guard_cycles = int(math.ceil(float(guard_time) / DT))
    deadlines = np.maximum(0, first_risk[risk_indices] - guard_cycles)
    seen = first_seen[risk_indices]
    never = seen == sentinel
    late = ~never & (seen > deadlines)
    valid = ~never & ~late
    violation = never | late
    first_violation_cycle = None
    first_violation_point = None
    first_violation_box = None
    first_violation_kind = None
    first_violation_seen_cycle = None
    first_violation_deadline_cycle = None
    maximum_lateness = 0
    if np.any(violation):
        violating_indices = risk_indices[violation]
        violation_cycles = first_risk[violating_indices]
        selected_local = int(np.argmin(violation_cycles))
        selected = int(violating_indices[selected_local])
        first_violation_cycle = int(first_risk[selected])
        first_violation_point = truth[selected].tolist()
        selected_risk_local = int(np.flatnonzero(risk_indices == selected)[0])
        first_violation_deadline_cycle = int(deadlines[selected_risk_local])
        if first_seen[selected] == sentinel:
            first_violation_kind = 'never_observed'
        else:
            first_violation_kind = 'published_late'
            first_violation_seen_cycle = int(first_seen[selected])
        for item in box_ranges:
            if item.start <= selected < item.stop:
                first_violation_box = item.box.name
                break
        finite_late = late & ~never
        if np.any(finite_late):
            maximum_lateness = int(np.max(seen[finite_late] - deadlines[finite_late]))
    passed = not np.any(violation)
    at_risk_by_box: dict[str, int] = {}
    violations_by_box: dict[str, int] = {}
    late_by_box: dict[str, int] = {}
    never_by_box: dict[str, int] = {}
    late_bounds_by_box: dict[str, dict[str, list[float]] | None] = {}
    never_bounds_by_box: dict[str, dict[str, list[float]] | None] = {}
    violation_global_mask = np.zeros(len(truth), dtype=bool)
    violation_global_mask[risk_indices] = violation
    late_global_mask = np.zeros(len(truth), dtype=bool)
    late_global_mask[risk_indices] = late
    never_global_mask = np.zeros(len(truth), dtype=bool)
    never_global_mask[risk_indices] = never
    for item in box_ranges:
        at_risk_by_box[item.box.name] = int(np.count_nonzero(risk_mask[item.start:item.stop]))
        violations_by_box[item.box.name] = int(np.count_nonzero(violation_global_mask[item.start:item.stop]))
        late_by_box[item.box.name] = int(np.count_nonzero(late_global_mask[item.start:item.stop]))
        never_by_box[item.box.name] = int(np.count_nonzero(never_global_mask[item.start:item.stop]))
        box_points = truth[item.start:item.stop]
        box_late = box_points[late_global_mask[item.start:item.stop]]
        box_never = box_points[never_global_mask[item.start:item.stop]]
        late_bounds_by_box[item.box.name] = None if not len(box_late) else {'minimum': np.min(box_late, axis=0).tolist(), 'maximum': np.max(box_late, axis=0).tolist()}
        never_bounds_by_box[item.box.name] = None if not len(box_never) else {'minimum': np.min(box_never, axis=0).tolist(), 'maximum': np.max(box_never, axis=0).tolist()}
    return ObservabilityReport(audit_method='nearest_solid_surface_witness_v2', passed=bool(passed), event_code='observed_before_risk' if passed else 'unobserved_hazard', observation_distance_m=float(observation_distance), guard_time_s=float(guard_time), guard_cycles=guard_cycles, truth_surface_spacing_m=float(truth_surface_spacing), truth_lipschitz_correction_m=float(correction), truth_surface_samples=int(len(truth)), ever_at_risk_samples=int(len(risk_indices)), observed_before_deadline_samples=int(np.count_nonzero(valid)), late_samples=int(np.count_nonzero(late)), never_observed_samples=int(np.count_nonzero(never)), maximum_observation_lateness_cycles=maximum_lateness, first_violation_cycle=first_violation_cycle, first_violation_point_m=first_violation_point, first_violation_box=first_violation_box, first_violation_kind=first_violation_kind, first_violation_seen_cycle=first_violation_seen_cycle, first_violation_deadline_cycle=first_violation_deadline_cycle, at_risk_samples_by_box=at_risk_by_box, violation_samples_by_box=violations_by_box, late_samples_by_box=late_by_box, never_observed_samples_by_box=never_by_box, late_sample_bounds_by_box_m=late_bounds_by_box, never_observed_sample_bounds_by_box_m=never_bounds_by_box, snapshot_count=len(snapshots))
