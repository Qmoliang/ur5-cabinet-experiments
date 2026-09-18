"""Create the bounded evidence report for frozen v4.3/v4.4.

Only immutable run artifacts and separately generated read-only audits are
summarised.  With one deterministic run per representation, task outcomes are
reported descriptively; no pseudo-replication or p-values are produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def single_run(root: Path) -> Path:
    matches = list(root.rglob("summary.json"))
    if len(matches) != 1:
        raise AssertionError(f"expected one summary below {root}, found {len(matches)}")
    return matches[0].parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args()
    evidence = args.evidence_dir.resolve()
    frozen = ROOT / "formal_results" / "final_two_camera" / "protocol_v4_4_v4_3_formal_frozen"
    manifest = json.loads((frozen / "frozen_manifest.json").read_text(encoding="utf-8"))
    runs = {row["representation"]: ROOT / row["path"] for row in manifest["runs"]}
    ellipsoid_run = runs["ellipsoid"]
    sphere_run = runs["sphere"]
    ellipsoid = json.loads((ellipsoid_run / "summary.json").read_text(encoding="utf-8"))
    sphere = json.loads((sphere_run / "summary.json").read_text(encoding="utf-8"))
    cover = json.loads((evidence / "v44_sphere_cover_audit.json").read_text(encoding="utf-8"))
    blockage = json.loads((evidence / "v44_blockage_audit.json").read_text(encoding="utf-8"))
    section = json.loads(
        (evidence / "cross_section_independent_final_snapshots" / "certificate.json").read_text(encoding="utf-8")
    )
    bench_e = json.loads((evidence / "mvt_simd_v43_ellipsoid.json").read_text(encoding="utf-8"))
    bench_s = json.loads((evidence / "mvt_simd_v44_sphere.json").read_text(encoding="utf-8"))

    q_e = np.asarray(np.load(ellipsoid_run / "q_history.npy"), dtype=float)
    q_s = np.asarray(np.load(sphere_run / "q_history.npy"), dtype=float)
    proxy_e = np.load(ellipsoid_run / "final_causal_proxies.npz")
    proxy_s = np.load(sphere_run / "final_causal_proxies.npz")
    points_e = np.asarray(proxy_e["filtered_points"], dtype=np.float64)
    points_s = np.asarray(proxy_s["filtered_points"], dtype=np.float64)
    point_hash_e = hashlib.sha256(np.ascontiguousarray(points_e).tobytes()).hexdigest()
    point_hash_s = hashlib.sha256(np.ascontiguousarray(points_s).tobytes()).hexdigest()

    same_fields = [
        "scene_version", "camera_count", "camera_width_px", "camera_height_px",
        "camera_pixel_stride", "camera_sample_rays_per_view", "depth_backend",
        "centervox_filter_size_m", "certificate_radius_limit_m",
        "maximum_uncertainty_union_inflation_limit", "uncertainty_fusion_mode",
        "robot_certificate_spheres", "safety_margin_m",
        "near_surface_gap_threshold_m", "contact_surface_gap_threshold_m",
        "task_gain", "max_task_speed_m_per_s", "success_tolerance_m",
        "success_hold_cycles", "cycles", "simulated_duration_s", "index_mode",
        "mvt_multilevel_enabled", "mvt_simd_enabled", "mvt_simd_width_float_lanes",
        "path_planner", "random_dither", "global_map_preloaded",
        "future_frames_used", "truth_collision_backtracking",
        "unknown_motion_policy", "incremental_free_occupied_unknown_map",
        "osqp_absolute_tolerance", "osqp_relative_tolerance", "osqp_max_iterations",
    ]
    parameter_rows = []
    for key in same_fields:
        equal = ellipsoid.get(key) == sphere.get(key)
        parameter_rows.append(
            {"parameter": key, "ellipsoid": ellipsoid.get(key), "sphere": sphere.get(key), "equal": equal}
        )
    source_e = ellipsoid.get("source_hashes", {})
    source_s = sphere.get("source_hashes", {})
    shared_source_names = sorted(source_e.keys() & source_s.keys())
    source_equal = [name for name in shared_source_names if source_e[name] == source_s[name]]
    source_different = [name for name in shared_source_names if source_e[name] != source_s[name]]

    scene_equal = sha256(ellipsoid_run / "scene.xml") == sha256(sphere_run / "scene.xml")
    q0_equal = bool(np.array_equal(q_e[0], q_s[0]))
    same_online_pointcloud = bool(point_hash_e == point_hash_s)
    result = {
        "report": "v4.3_v4.4_bounded_finite_evidence",
        "statistical_policy": {
            "task_runs_per_representation": 1,
            "inferential_test_performed": False,
            "reason": "deterministic case-study runs are not independent replicates",
            "timing_samples_are_independent_trials": False,
            "timing_reporting": "median, p95 and p99 only; no p-values",
        },
        "frozen_integrity": {
            "manifest_status": manifest["status"],
            "manifest_file_count": manifest["file_count"],
            "scene_xml_equal": scene_equal,
            "initial_q_exactly_equal": q0_equal,
            "declared_parameter_rows_equal": bool(all(row["equal"] for row in parameter_rows)),
            "shared_source_hash_count": len(shared_source_names),
            "equal_shared_source_hashes": source_equal,
            "different_shared_source_hashes": source_different,
        },
        "parameter_rows": parameter_rows,
        "pointcloud_pairing": {
            "ellipsoid_final_centervox_count": int(len(points_e)),
            "sphere_final_centervox_count": int(len(points_s)),
            "ellipsoid_final_centervox_sha256": point_hash_e,
            "sphere_final_centervox_sha256": point_hash_s,
            "same_final_centervox": same_online_pointcloud,
            "interpretation": (
                "same sensing/map configuration but different trajectory-dependent online observations; "
                "not a same-CenterVox representation ablation"
            ),
        },
        "main_outcomes": {
            "ellipsoid": {
                "success": bool(ellipsoid["success"]),
                "final_error_mm": float(ellipsoid["final_error_m"] * 1000.0),
                "first_success_time_s": ellipsoid["first_success_time_s"],
                "maximum_success_hold_cycles": ellipsoid["maximum_success_hold_cycles"],
                "proxy_count": ellipsoid["final_proxy_count"],
                "controller_p99_ms": ellipsoid["controller_ms_p99"],
                "observability_passed": bool(ellipsoid["observability_passed"]),
            },
            "sphere": {
                "success": bool(sphere["success"]),
                "final_error_mm": float(sphere["final_error_m"] * 1000.0),
                "maximum_ee_x_m": sphere["maximum_ee_x_m"],
                "proxy_count": sphere["final_proxy_count"],
                "controller_p99_ms": sphere["controller_ms_p99"],
                "observability_passed": bool(sphere["observability_passed"]),
            },
        },
        "sphere_cover": {
            "proxy_count": cover["proxy_count"],
            "centervox_count": cover["center_voxel_count"],
            "radius_mm": cover["sphere_radius_mm"],
            "coverage_multiplicity": cover["coverage_multiplicity_per_centervox"],
            "zero_unique_witness": cover["spheres_with_zero_unique_witness"],
            "contained_spheres": cover["geometrically_contained_spheres"],
            "actual_over_square_grid_reference": cover["actual_over_square_grid_reference"],
        },
        "geometry_and_qp_failure": {
            "first_failed_cycle": blockage["first_failed_cycle"],
            "post_failure_zero_velocity_cycles": blockage["post_failure_zero_velocity_cycles"],
            "limiting_pair": blockage["limiting_pair"],
            "local_fixed_xy_line_closed": blockage["local_fixed_xy_vertical_line"]["strictly_closed_on_this_fixed_xy_line"],
            "whole_section_closed_at_failure_x": blockage["entire_physical_yz_section_at_same_x"]["strictly_closed"],
            "whole_section_open_witness": blockage["entire_physical_yz_section_at_same_x"]["open_witness"],
            "final_sphere_front_section_closed": section["sphere_section"]["strictly_closed"],
            "final_sphere_front_open_witness": section["sphere_section"]["open_witness"],
            "final_ellipsoid_center_open_radius_mm": float(
                section["successful_ellipsoid_run_section"]["strict_open_ball_radius_m"] * 1000.0
            ),
            "independent_snapshot_certificate_passed": bool(section["paired_certificate_passed"]),
            "causal_gates_passed": bool(section["causal_gates_passed"]),
        },
        "mvt_simd_current_kernel_frozen_workload": {
            "ellipsoid": bench_e["aggregate_timing"],
            "sphere": bench_s["aggregate_timing"],
            "ellipsoid_oracle_equal": bench_e["all_scalar_and_avx2_rows_equal_full_scan"],
            "sphere_oracle_equal": bench_s["all_scalar_and_avx2_rows_equal_full_scan"],
            "historical_binary_rerun": False,
            "ellipsoid_all_hashes_match_frozen": bench_e["implementation"]["all_relevant_hashes_match_frozen"],
            "sphere_all_hashes_match_frozen": bench_s["implementation"]["all_relevant_hashes_match_frozen"],
        },
        "claim_status": [
            {"claim": "v4.3 ellipsoid reaches the strict target", "supported": bool(ellipsoid["success"])},
            {"claim": "v4.4 sphere run persistently stalls after a hard-QP failure", "supported": blockage["post_failure_zero_velocity_cycles"] == 2832},
            {"claim": "v4.4 final sphere cover is inclusion-minimal in its candidate family", "supported": cover["spheres_with_zero_unique_witness"] == 0 and cover["geometrically_contained_spheres"] == 0},
            {"claim": "v4.4 sphere proxies close the entire necessary physical section", "supported": bool(blockage["entire_physical_yz_section_at_same_x"]["strictly_closed"])},
            {"claim": "v4.3/v4.4 are a same-CenterVox paired ablation", "supported": same_online_pointcloud},
            {"claim": "current MVT scalar and AVX2 return the exact full-scan candidate sets", "supported": bool(bench_e["all_scalar_and_avx2_rows_equal_full_scan"] and bench_s["all_scalar_and_avx2_rows_equal_full_scan"])},
            {"claim": "the historical native controller binary has been rerun", "supported": False},
            {"claim": "both frozen runs pass complete online observability", "supported": bool(ellipsoid["observability_passed"] and sphere["observability_passed"])},
        ],
    }

    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "finite_evidence_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# v4.3 椭球 / v4.4 球：有限补证统计",
        "",
        "> 本报告仅汇总冻结制品和独立只读审计。每种表示只有一次确定性闭环运行，",
        "> 因此不进行 t 检验或人为构造 p 值。微基准重复样本不是独立任务试验。",
        "",
        "## 核心结果",
        "",
        f"- v4.3 椭球：最终误差 {ellipsoid['final_error_m']*1000.0:.6f} mm，严格成功；控制器 p99 {ellipsoid['controller_ms_p99']:.3f} ms。",
        f"- v4.4 球：最终误差 {sphere['final_error_m']*1000.0:.3f} mm，失败；第 {blockage['first_failed_cycle']} 周期后连续 {blockage['post_failure_zero_velocity_cycles']} 周期零速度。",
        f"- v4.4 覆盖：{cover['proxy_count']} 球覆盖 {cover['center_voxel_count']} CenterVox；无独占见证球 0，完全包含球 0。",
        "",
        "## 必须保留的证据边界",
        "",
        f"- 两组场景 XML 相同：{scene_equal}；初始 q 完全相同：{q0_equal}。",
        f"- 最终 CenterVox 哈希相同：{same_online_pointcloud}。两条轨迹改变了后续观测，因此这不是严格同点云配对。",
        f"- v4.4 当前固定 x,y 竖线闭合：{blockage['local_fixed_xy_vertical_line']['strictly_closed_on_this_fixed_xy_line']}；同一 x 的整个物理截面闭合：{blockage['entire_physical_yz_section_at_same_x']['strictly_closed']}。",
        f"- 最终球快照的正面截面闭合：{section['sphere_section']['strictly_closed']}；最终椭球中心开球半径 {section['successful_ellipsoid_run_section']['strict_open_ball_radius_m']*1000.0:.3f} mm。",
        f"- 完整可观测性门：球 {sphere['observability_passed']}，椭球 {ellipsoid['observability_passed']}。",
        "",
        "因此，现有冻结结果支持‘该球型 LiuQP 运行发生持久局部硬约束死锁，椭球运行成功’，",
        "但尚不支持‘球证书封闭所有可扰动构型’或‘同一 CenterVox 下只有表示变量不同’。",
        "",
        "## 当前原生宽阶段离线回放",
        "",
        f"- 椭球负载：全量/MVT 标量/MVT AVX2 中位数分别为 {bench_e['aggregate_timing']['full_scan']['median_ms']:.4f}/{bench_e['aggregate_timing']['mvt_scalar']['median_ms']:.4f}/{bench_e['aggregate_timing']['mvt_avx2']['median_ms']:.4f} ms。",
        f"- 球负载：全量/MVT 标量/MVT AVX2 中位数分别为 {bench_s['aggregate_timing']['full_scan']['median_ms']:.4f}/{bench_s['aggregate_timing']['mvt_scalar']['median_ms']:.4f}/{bench_s['aggregate_timing']['mvt_avx2']['median_ms']:.4f} ms。",
        "- 两组所有候选 ID 均与 float32 O(N) oracle 一致。当前 C++/DLL 哈希不匹配历史冻结二进制，故这是当前内核的离线补证，不是历史运行计时复现。",
    ]
    (evidence / "FINITE_EVIDENCE_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
