"""Aggregate the six formal uniform-radius online runs without altering them."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "formal_results" / "final_two_camera" / "formal_uniform_radius_v3_3"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cycle_columns(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times, errors = [], []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            times.append(float(row["time_s"]))
            errors.append(float(row["error_m"]))
    return np.asarray(times), np.asarray(errors)


def aggregate(results: Path = RESULTS) -> dict:
    rows = []
    traces = {}
    for run_dir in sorted(path for path in results.iterdir() if path.is_dir()):
        summary = _json(run_dir / "summary.json")
        section = _json(run_dir / "online_mandatory_section_audit.json")
        snapshot = _json(run_dir / "online_snapshot_q_plus_u_audit.json")
        candidate = _json(run_dir / "candidate_replay_report.json")
        sweep = _json(run_dir / "post_control_sweep_audit_report.json")
        occupancy = _json(run_dir / "recorded_occupancy_replay_report.json")
        representation = summary["representation"]
        scale_mm = int(round(summary["certificate_radius_limit_m"] * 1000.0))
        group = ("S" if representation == "sphere" else "E") + str(scale_mm)
        goal_safe = sum(bool(item["safe"]) for item in snapshot["goal_configurations"])
        all_audits = bool(
            summary["all_proxy_radius_limit_checks_passed"]
            and summary["all_centervox_coverage_checks_passed"]
            and summary["all_mvt_oracle_checks_passed"]
            and summary["exact_penetrating_cycles"] == 0
            and candidate["passed"]
            and candidate["all_per_cycle_candidate_counts_matched"]
            and sweep["passed"]
            and sweep["unsafe_cycles"] == 0
            and occupancy["passed"]
            and occupancy["cycles_match_source_proxy_and_frame_logs"]
        )
        realtime = bool(summary["control_compute_ms_p99"] <= 20.0)
        formal_pass = bool(
            summary["success"]
            and summary["observability_passed"]
            and realtime
            and all_audits
        )
        sphere_closed = bool(
            section["sphere_certificate_section"]["continuously_closed"]
        )
        ellipsoid_open = bool(
            section["ellipsoid_q_plus_u_section"]["certified_open"]
        )
        if representation == "sphere":
            interpretation = (
                "continuous_certificate_closure"
                if sphere_closed
                else "qp_stalled_without_continuous_section_closure"
            )
        else:
            interpretation = (
                "certified_section_open_but_online_qp_failed"
                if ellipsoid_open and not summary["success"]
                else "ellipsoid_outcome_other"
            )
        rows.append(
            {
                "group": group,
                "representation": representation,
                "certificate_scale_mm": scale_mm,
                "success": bool(summary["success"]),
                "formal_pass": formal_pass,
                "interpretation": interpretation,
                "final_error_mm": summary["final_error_m"] * 1000.0,
                "minimum_error_mm": summary["minimum_error_m"] * 1000.0,
                "maximum_ee_x_m": summary["maximum_ee_x_m"],
                "proxy_count": summary["final_proxy_count"],
                "published_perception_hz": summary[
                    "effective_published_perception_hz_wall"
                ],
                "control_p50_ms": summary["control_compute_ms_p50"],
                "control_p99_ms": summary["control_compute_ms_p99"],
                "control_realtime_20ms": realtime,
                "deadline_misses": summary["deadline_misses"],
                "observability_passed": bool(summary["observability_passed"]),
                "late_hazard_samples": summary["observability_late_samples"],
                "radius_limit_passed": bool(
                    summary["all_proxy_radius_limit_checks_passed"]
                ),
                "coverage_passed": bool(
                    summary["all_centervox_coverage_checks_passed"]
                ),
                "mvt_oracle_passed": bool(
                    summary["all_mvt_oracle_checks_passed"]
                ),
                "exact_penetrating_cycles": summary["exact_penetrating_cycles"],
                "continuous_sweep_passed": bool(sweep["passed"]),
                "sweep_minimum_clearance_mm": sweep["minimum_clearance_m"] * 1000.0,
                "candidate_replay_passed": bool(candidate["passed"]),
                "candidate_pair_rows": candidate["candidate_pair_rows"],
                "occupancy_replay_passed": bool(occupancy["passed"]),
                "sphere_section_continuous_upper_mm": section[
                    "sphere_certificate_section"
                ]["continuous_maximum_upper_bound_m"]
                * 1000.0,
                "sphere_section_closed": sphere_closed,
                "ellipsoid_section_cell_lower_mm": section[
                    "ellipsoid_q_plus_u_section"
                ]["whole_grid_cell_clearance_lower_bound_m"]
                * 1000.0,
                "ellipsoid_section_open": ellipsoid_open,
                "safe_frozen_goal_ik_count": goal_safe,
                "frozen_goal_ik_count": len(snapshot["goal_configurations"]),
                "all_non_outcome_audits_passed": all_audits,
                "run_directory": str(run_dir.resolve()),
            }
        )
        time_s, error_m = _cycle_columns(run_dir / "cycles.csv")
        traces[group] = {"time_s": time_s, "error_m": error_m}

    rows.sort(key=lambda row: (row["certificate_scale_mm"], row["representation"]))
    expected = {"S30", "E30", "S50", "E50", "S70", "E70"}
    if {row["group"] for row in rows} != expected:
        raise AssertionError("formal result directory does not contain exactly six groups")

    csv_path = results / "uniform_radius_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    colors = {30: "#0072B2", 50: "#009E73", 70: "#D55E00"}
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), constrained_layout=True)
    for row in rows:
        trace = traces[row["group"]]
        axes[0].plot(
            trace["time_s"],
            trace["error_m"] * 1000.0,
            color=colors[row["certificate_scale_mm"]],
            linestyle="-" if row["representation"] == "ellipsoid" else "--",
            linewidth=1.5,
            label=row["group"],
        )
    axes[0].axhline(18.0, color="black", linewidth=1.0, linestyle=":", label="18 mm gate")
    axes[0].set(xlabel="Simulation time (s)", ylabel="End-effector error (mm)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    labels = [row["group"] for row in rows]
    p99 = [row["control_p99_ms"] for row in rows]
    bar_colors = [colors[row["certificate_scale_mm"]] for row in rows]
    axes[1].bar(labels, p99, color=bar_colors, alpha=0.85)
    axes[1].axhline(20.0, color="black", linewidth=1.0, linestyle=":")
    axes[1].set(xlabel="Group", ylabel="Control compute p99 (ms)")
    axes[1].grid(axis="y", alpha=0.25)
    figure_path = results / "uniform_radius_error_timing.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    markdown = [
        "# 统一证书尺度正式在线结果 v3.3",
        "",
        "成功门为末端误差不超过 18 mm 并保持；实时门为控制计算 p99 不超过 20 ms。所有数字来自本目录的真实 CSV/JSON/NPZ。",
        "",
        "| 组 | 成功 | 末端最小误差 (mm) | 控制 p99 (ms) | 地图发布 (Hz) | 观测门 | 球截面闭塞 | 椭球截面开口 | 正式通过 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {group} | {success} | {minimum_error_mm:.3f} | {control_p99_ms:.3f} | "
            "{published_perception_hz:.3f} | {observability_passed} | "
            "{sphere_section_closed} | {ellipsoid_section_open} | {formal_pass} |".format(
                **row
            )
        )
    markdown.extend(
        [
            "",
            "结论：S50/S70 的实际在线球证书在必经截面形成连续闭塞，而同组实际 Q+U 椭球证书保留了经保守下界证明的开口；S30 没有连续闭塞证明。六组均未跨过冻结的 18 mm 成功门，因此本批结果不能声称椭球在线系统成功。所有冻结目标 IK 在各自最终在线代理快照下均存在负 Q+U 余量，说明当前剩余问题是在线保守代理/地图发布与 QP 的耦合，而不是截面表示能力本身。",
            "",
            "无效试跑（漏传并集膨胀 1.25）完整保存在相邻 `formal_uniform_radius_v3_3_no_ui_invalid`，不纳入本表。",
        ]
    )
    md_path = results / "uniform_radius_results.md"
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    source_files = [
        ROOT / "aggregate_uniform_radius_v3_3.py",
        ROOT / "audit_uniform_radius_online_sections.py",
        ROOT / "verify_recorded_causal_occupancy.py",
        ROOT / "native_mvt.py",
        ROOT / "native_mvt" / "native_mvt.cpp",
        ROOT / "native_mvt" / "native_mvt.dll",
    ]
    result = {
        "protocol": "uniform_radius_v3_3",
        "mock_data": False,
        "rows": rows,
        "artifacts": {
            "csv": str(csv_path.resolve()),
            "markdown": str(md_path.resolve()),
            "figure": str(figure_path.resolve()),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_files
        },
        "artifact_sha256": {
            path.name: _sha256(path) for path in (csv_path, md_path, figure_path)
        },
    }
    json_path = results / "uniform_radius_results.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    result = aggregate()
    print(json.dumps({"rows": len(result["rows"]), **result["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
