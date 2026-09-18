"""Run the frozen formal protocol with reproducible bounded depth noise.

This is a sensitivity wrapper, not a replacement controller. It leaves the
formal runner unchanged, perturbs only measured optical depth, and enlarges
the directional uncertainty certificate by the exact clipping bound. The
perturbation is a deterministic function of source joint configuration,
camera name, and seed so a causal map replay can reproduce it.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import run_protocol_v3_async_online as protocol
from depth_camera_perception import (
    DepthObservation,
    UR5MountedDepthCamera as BaseDepthCamera,
)


NOISE_STD_ENV = "LIUQP_BOUNDED_DEPTH_NOISE_STD_M"
NOISE_BOUND_ENV = "LIUQP_BOUNDED_DEPTH_NOISE_BOUND_M"
NOISE_SEED_ENV = "LIUQP_BOUNDED_DEPTH_NOISE_SEED"
BASELINE_OPTICAL_BOUND_M = 0.003


def _noise_parameters() -> tuple[float, float, int]:
    return (
        float(os.environ.get(NOISE_STD_ENV, "0")),
        float(os.environ.get(NOISE_BOUND_ENV, "0")),
        int(os.environ.get(NOISE_SEED_ENV, "0")),
    )


class BoundedNoiseDepthCamera(BaseDepthCamera):
    """Depth camera with a clipped Gaussian error covered by its certificate."""

    def __init__(self, *args, **kwargs) -> None:
        noise_std, noise_bound, seed = _noise_parameters()
        requested_bound = float(kwargs.pop("optical_depth_error_bound", 0.0))
        kwargs["depth_noise_std"] = 0.0
        kwargs["optical_depth_error_bound"] = requested_bound + noise_bound
        kwargs["seed"] = seed
        super().__init__(*args, **kwargs)
        self.bounded_noise_std = noise_std
        self.bounded_noise_bound = noise_bound
        self.bounded_noise_seed = seed

    def _rng_for(self, data, camera_name: str) -> np.random.Generator:
        q = np.ascontiguousarray(data.qpos, dtype="<f8")
        payload = (
            q.tobytes()
            + camera_name.encode("utf-8")
            + int(self.bounded_noise_seed).to_bytes(8, "little", signed=True)
        )
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return np.random.default_rng(int.from_bytes(digest, "little"))

    def capture(self, data) -> list[DepthObservation]:
        observations = super().capture(data)
        if self.bounded_noise_std == 0.0 or self.bounded_noise_bound == 0.0:
            return observations
        perturbed: list[DepthObservation] = []
        for observation in observations:
            if len(observation.points) == 0:
                perturbed.append(observation)
                continue
            rng = self._rng_for(data, observation.camera_name)
            errors = np.clip(
                rng.normal(
                    0.0,
                    self.bounded_noise_std,
                    size=len(observation.optical_depths),
                ),
                -self.bounded_noise_bound,
                self.bounded_noise_bound,
            )
            ray_directions = (
                observation.points - observation.camera_position[None, :]
            ) / observation.optical_depths[:, None]
            noisy_depths = observation.optical_depths + errors
            if np.any(noisy_depths <= 0.0):
                raise AssertionError("bounded sensitivity noise produced invalid depth")
            noisy_points = observation.camera_position[None, :] + (
                noisy_depths[:, None] * ray_directions
            )
            perturbed.append(
                replace(
                    observation,
                    points=np.asarray(noisy_points, dtype=float),
                    optical_depths=np.asarray(noisy_depths, dtype=float),
                )
            )
        return perturbed


# The process worker resolves this module global when constructing its camera.
# Applying the patch at import time also covers Windows spawn children.
protocol.UR5MountedDepthCamera = BoundedNoiseDepthCamera


def _run_directory(output: Path, representation: str) -> Path:
    matches = [
        path.parent
        for path in output.glob(f"*-{representation}-*/summary.json")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {representation} run under {output}, found {matches}"
        )
    return matches[0]


def _annotate(
    output: Path,
    representation: str,
    noise_std: float,
    noise_bound: float,
    seed: int,
) -> dict:
    run_dir = _run_directory(output, representation)
    path = run_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["bounded_depth_noise_sensitivity"] = {
        "distribution": "zero_mean_gaussian_clipped_to_closed_interval",
        "standard_deviation_m": noise_std,
        "clipping_bound_m": noise_bound,
        "seed": seed,
        "baseline_optical_error_bound_m": BASELINE_OPTICAL_BOUND_M,
        "certificate_total_optical_error_bound_m": (
            BASELINE_OPTICAL_BOUND_M + noise_bound
        ),
        "certificate_covers_every_injected_sample": True,
        "stateless_key": "source_q_float64_bytes,camera_name,seed",
        "controller_or_runner_modified": False,
    }
    summary["bounded_noise_wrapper_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representation", choices=("sphere", "ellipsoid", "both"), default="both"
    )
    parser.add_argument("--noise-std-mm", type=float, required=True)
    parser.add_argument("--noise-bound-mm", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--maximum-cycles", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    noise_std = args.noise_std_mm / 1000.0
    noise_bound = args.noise_bound_mm / 1000.0
    if noise_std < 0.0 or noise_bound < 0.0:
        raise ValueError("noise standard deviation and bound must be nonnegative")
    if noise_std > 0.0 and noise_bound == 0.0:
        raise ValueError("nonzero noise requires a positive deterministic bound")
    os.environ[NOISE_STD_ENV] = repr(noise_std)
    os.environ[NOISE_BOUND_ENV] = repr(noise_bound)
    os.environ[NOISE_SEED_ENV] = str(args.seed)

    representations = (
        ("sphere", "ellipsoid")
        if args.representation == "both"
        else (args.representation,)
    )
    summaries = []
    for representation in representations:
        protocol.run_one(
            representation,
            "mvt_simd",
            args.output,
            maximum_cycles=args.maximum_cycles,
            unknown_policy="observed_only",
            realtime_pacing=True,
            perception_executor="process",
            centervox_size=0.0075,
            maximum_uncertainty_union_inflation=1.25,
            camera_width=320,
            camera_height=180,
            camera_pixel_stride=1,
            camera_names=protocol.CAMERA_NAMES,
            scene_version="formal",
            record_dense_map_snapshots=False,
            sweep_guard_mode="post_control_audit",
        )
        summaries.append(
            _annotate(
                args.output,
                representation,
                noise_std,
                noise_bound,
                args.seed,
            )
        )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
