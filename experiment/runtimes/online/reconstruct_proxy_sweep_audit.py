"""Run the continuous causal-proxy sweep certificate after control stops."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from model import build_robot_certificate, certificate_world_positions, set_configuration
from run_protocol_v3_async_online import ContinuousProxyGuard


def reconstruct(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    archive = np.load(run_dir / "causal_proxy_snapshots.npz")
    publish_cycles = np.asarray(archive["publish_cycles"], dtype=np.int64)
    offsets = np.asarray(archive["offsets"], dtype=np.int64)
    fields = {
        name: np.asarray(archive[name])
        for name in (
            "proxy_ids",
            "centers",
            "sphere_radii",
            "ellipsoid_shapes",
            "ellipsoid_outer_shapes",
            "proxy_uncertainty_shapes",
            "uncertainty_offsets",
        )
    }
    q_after = np.asarray(np.load(run_dir / "q_history.npy"), dtype=float)
    source = np.load(run_dir / "causal_source_configurations.npz")
    q0 = np.asarray(source["q"][0], dtype=float)
    source.close()
    model = mujoco.MjModel.from_xml_path(str(run_dir / "scene.xml"))
    data = mujoco.MjData(model)
    robot = build_robot_certificate(model)
    radii = np.asarray([item.radius for item in robot], dtype=float)
    guards: dict[int, ContinuousProxyGuard] = {}
    rows = []
    unsafe = 0
    minimum = np.inf
    for cycle, end_q in enumerate(q_after):
        snapshot = int(np.searchsorted(publish_cycles, cycle, side="right") - 1)
        snapshot = max(snapshot, 0)
        if snapshot not in guards:
            start, stop = map(int, offsets[snapshot : snapshot + 2])
            proxies = SimpleNamespace(
                proxy_ids=fields["proxy_ids"][start:stop],
                centers=fields["centers"][start:stop],
                sphere_radii=fields["sphere_radii"][start:stop],
                base_ellipsoid_shapes=fields["ellipsoid_shapes"][start:stop],
                ellipsoid_outer_shapes=fields["ellipsoid_outer_shapes"][start:stop],
                proxy_uncertainty_shapes=fields["proxy_uncertainty_shapes"][start:stop],
                proxy_offset_radii=fields["uncertainty_offsets"][start:stop],
            )
            packet = SimpleNamespace(
                proxies=proxies,
                mvt=None,
                ellipsoid_eigenvalues=None,
                ellipsoid_rotations=None,
                uncertainty_eigenvalues=None,
            )
            guards[snapshot] = ContinuousProxyGuard(
                packet, summary["representation"]
            )
        begin_q = q0 if cycle == 0 else q_after[cycle - 1]
        set_configuration(model, data, begin_q)
        starts = certificate_world_positions(data, robot)
        set_configuration(model, data, end_q)
        ends = certificate_world_positions(data, robot)
        certificate = guards[snapshot].certify(starts, ends, radii)
        unsafe += int(not certificate.safe)
        minimum = min(minimum, certificate.minimum_clearance_m)
        rows.append(
            {
                "cycle": cycle,
                "time_s": cycle * 0.02,
                "active_generation": snapshot,
                **certificate.__dict__,
            }
        )
    archive.close()
    path = run_dir / "post_control_sweep_audit.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "passed": unsafe == 0,
        "method": "post_control_exact_causal_proxy_continuous_sweep_replay",
        "online_control_or_publication_timing_affected": False,
        "liu_qp_commands_modified": False,
        "cycles": len(rows),
        "unsafe_cycles": unsafe,
        "minimum_clearance_m": (
            None if not np.isfinite(minimum) else float(minimum)
        ),
        "output": str(path.resolve()),
    }
    (run_dir / "post_control_sweep_audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if unsafe:
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
