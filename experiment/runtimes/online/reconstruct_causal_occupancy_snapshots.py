"""Rebuild causal free/occupied/unknown map deltas after an online run.

The online controller saves only each published depth packet's source joint
configuration.  This script deterministically re-renders those same two local
cameras and repeats the native map updates after control has stopped.  Thus map
visualization cannot change perception publication timing or LiuQP commands.
Unknown space is represented exactly as the complement of the replayed sparse
FREE/OCCUPIED dictionary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import mujoco
import numpy as np

from depth_camera_perception import UR5MountedDepthCamera
from model import build_robot_certificate, certificate_world_positions, set_configuration
from native_occupancy import NativeIncrementalOccupancyMap
from protocol_drawer_scene import INITIAL_CALIBRATED_FREE_PADDING
from run_protocol_v3_online_ablation import MAP_VOXEL_SIZE


def _save_deltas(path: Path, snapshots: list[tuple[int, int, np.ndarray, np.ndarray]]) -> None:
    counts = np.asarray([len(item[2]) for item in snapshots], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    np.savez_compressed(
        path,
        publish_cycles=np.asarray([item[0] for item in snapshots], dtype=np.int64),
        source_cycles=np.asarray([item[1] for item in snapshots], dtype=np.int64),
        offsets=offsets,
        voxel_keys=np.concatenate([item[2] for item in snapshots], axis=0),
        voxel_states=np.concatenate([item[3] for item in snapshots], axis=0),
        voxel_size_m=np.asarray(MAP_VOXEL_SIZE, dtype=float),
        storage=np.asarray("discrete_state_change_deltas"),
        unknown_definition=np.asarray("all_voxel_keys_absent_from_sparse_FREE_OCCUPIED_map"),
    )


def reconstruct(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    source = np.load(run_dir / "causal_source_configurations.npz")
    publish_cycles = np.asarray(source["publish_cycles"], dtype=np.int64)
    source_cycles = np.asarray(source["source_cycles"], dtype=np.int64)
    q_values = np.asarray(source["q"], dtype=float)
    with (run_dir / "perception_frames.csv").open(newline="", encoding="utf-8") as stream:
        frame_rows = list(csv.DictReader(stream))
    if not (
        len(publish_cycles) == len(source_cycles) == len(q_values) == len(frame_rows)
    ):
        raise ValueError("published source configurations do not match frame rows")

    model = mujoco.MjModel.from_xml_path(str(run_dir / "scene.xml"))
    data = mujoco.MjData(model)
    robot = build_robot_certificate(model)
    robot_radii = np.asarray([item.radius for item in robot], dtype=float)
    camera_class = UR5MountedDepthCamera
    noise_metadata = summary.get("bounded_depth_noise_sensitivity")
    if noise_metadata is not None:
        os.environ["LIUQP_BOUNDED_DEPTH_NOISE_STD_M"] = repr(
            float(noise_metadata["standard_deviation_m"])
        )
        os.environ["LIUQP_BOUNDED_DEPTH_NOISE_BOUND_M"] = repr(
            float(noise_metadata["clipping_bound_m"])
        )
        os.environ["LIUQP_BOUNDED_DEPTH_NOISE_SEED"] = str(
            int(noise_metadata["seed"])
        )
        from run_protocol_v3_bounded_noise import BoundedNoiseDepthCamera

        camera_class = BoundedNoiseDepthCamera
    camera = camera_class(
        model,
        camera_names=tuple(summary["local_moving_cameras"]),
        width=int(summary["camera_width_px"]),
        height=int(summary["camera_height_px"]),
        pixel_stride=int(summary["camera_pixel_stride"]),
        minimum_range=float(summary["camera_minimum_range_m"]),
        optical_depth_error_bound=0.003,
        occluding_self_filter=True,
    )
    occupancy = NativeIncrementalOccupancyMap(
        MAP_VOXEL_SIZE, track_state_deltas=True
    )
    snapshots: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    replay: dict[tuple[int, int, int], int] = {}
    frame_reports = []
    try:
        for index, (publish_cycle, source_cycle, q, frame_row) in enumerate(
            zip(publish_cycles, source_cycles, q_values, frame_rows)
        ):
            if int(frame_row["source_cycle"]) != int(source_cycle):
                raise AssertionError("source cycle mismatch during map replay")
            if int(frame_row["publish_cycle"]) != int(publish_cycle):
                raise AssertionError("publish cycle mismatch during map replay")
            set_configuration(model, data, q)
            observations = camera.capture(data)
            stats = occupancy.integrate(observations)
            robot_positions = certificate_world_positions(data, robot)
            occupancy.mark_current_robot_free(
                robot_positions,
                robot_radii,
                padding=INITIAL_CALIBRATED_FREE_PADDING if index == 0 else 0.0,
            )
            keys, states = occupancy.snapshot_delta_arrays()
            snapshots.append(
                (int(publish_cycle), int(source_cycle), keys, states)
            )
            for key, state in zip(keys, states):
                packed = tuple(map(int, key))
                if int(state) == 0:
                    replay.pop(packed, None)
                else:
                    replay[packed] = int(state)
            expected_free = int(frame_row["free_voxels"])
            expected_occupied = int(frame_row["occupied_voxels"])
            if stats.free_voxels != expected_free or stats.occupied_voxels != expected_occupied:
                raise AssertionError(
                    "deterministic occupancy replay diverged from online frame "
                    f"{index}: replay free/occupied="
                    f"{stats.free_voxels}/{stats.occupied_voxels}, online="
                    f"{expected_free}/{expected_occupied}"
                )
            frame_reports.append(
                {
                    "snapshot": index,
                    "publish_cycle": int(publish_cycle),
                    "source_cycle": int(source_cycle),
                    "delta_state_changes": len(keys),
                    "known_sparse_states_after_robot_free": len(replay),
                    "free_voxels_before_current_robot_free": stats.free_voxels,
                    "occupied_voxels_before_current_robot_free": stats.occupied_voxels,
                }
            )
    finally:
        occupancy.close()
        camera.close()
        source.close()

    output = run_dir / "causal_occupancy_replay_deltas.npz"
    _save_deltas(output, snapshots)
    report = {
        "passed": True,
        "method": (
            "post_control_deterministic_bounded_noise_depth_and_native_map_replay"
            if noise_metadata is not None
            else "post_control_deterministic_depth_and_native_map_replay"
        ),
        "online_control_or_publication_timing_affected": False,
        "map_voxel_size_m": MAP_VOXEL_SIZE,
        "snapshot_count": len(snapshots),
        "total_state_change_rows": int(sum(len(item[2]) for item in snapshots)),
        "final_known_sparse_states": len(replay),
        "unknown_definition": "complement of sparse FREE/OCCUPIED keys",
        "all_online_frame_state_counts_matched": True,
        "output": str(output.resolve()),
        "frames": frame_reports,
    }
    (run_dir / "occupancy_replay_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    report = reconstruct(args.run_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
