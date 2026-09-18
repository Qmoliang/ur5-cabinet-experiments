"""Deterministic cProfile harness for one causal local-perception frame."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
from pathlib import Path

import numpy as np

from run_protocol_v3_async_online import PerceptionWorker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--centervox-size", type=float, default=0.0075)
    parser.add_argument(
        "--uncertainty-mode",
        choices=("separate_uncertainty", "partitioned_separate_uncertainty"),
        default="separate_uncertainty",
    )
    parser.add_argument("--source-configurations", type=Path)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--pstats", type=Path)
    parser.add_argument("--rows", type=int, default=40)
    args = parser.parse_args()
    worker = PerceptionWorker(
        "ellipsoid",
        "mvt_simd",
        centervox_size=args.centervox_size,
        maximum_uncertainty_union_inflation=1.05,
        certificate_radius_limit=0.07,
        uncertainty_fusion_mode=args.uncertainty_mode,
        direct_thin_axis_inflation=1.0,
        direct_tangent_subdivisions=1,
        direct_partition_mode="grid",
        sphere_cover_mode="matched",
        ellipsoid_cover_mode="adaptive_irredundant",
        radius_limit_representation="ellipsoid",
        camera_width=args.width,
        camera_height=args.height,
        camera_pixel_stride=args.stride,
        camera_names=("ur5_depth_wrist", "ur5_depth_forearm"),
        scene_version="camera_v5_balanced",
    )
    if args.source_configurations is None:
        source_cycles = np.arange(args.frames, dtype=np.int64)
        configurations = np.repeat(
            np.asarray(worker.scene.q0, dtype=float)[None, :], args.frames, axis=0
        )
    else:
        archive = np.load(args.source_configurations, allow_pickle=False)
        source_cycles = np.asarray(archive["source_cycles"], dtype=np.int64)[
            : args.frames
        ]
        configurations = np.asarray(archive["q"], dtype=float)[: args.frames]
        archive.close()
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        rows = []
        for frame, (source_cycle, q) in enumerate(
            zip(source_cycles, configurations)
        ):
            packet = worker.process(
                q,
                requested_frame=frame,
                source_cycle=int(source_cycle),
                source_time_s=float(source_cycle) * 0.02,
                transferable=True,
                include_dense_snapshot=False,
            )
            rows.append(packet.frame_row)
        profiler.disable()
        for row in rows:
            print(row)
        if args.pstats is not None:
            args.pstats.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(args.pstats)
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
            "cumulative"
        ).print_stats(args.rows)
        print(stream.getvalue())
    finally:
        worker.close()


if __name__ == "__main__":
    main()
