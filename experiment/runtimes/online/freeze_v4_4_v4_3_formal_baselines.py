"""Freeze the accepted v4.4 sphere and v4.3 ellipsoid result trees.

The source result trees are read-only inputs.  This script writes only a
content-addressed manifest and a short policy note in a separate directory.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "formal_results" / "final_two_camera"
OUTPUT = RESULTS / "protocol_v4_4_v4_3_formal_frozen"

SPHERE_ROOT = RESULTS / (
    "formal_strict_goal_v4_3_surface_core_sphere_"
    "camera_quarter_cv7.5mm_cr70mm_u1p05_stride1_vmax180mmps_"
    "affinity_perception_heavy_resolution_bounded_formal3_"
    "adaptive_irredundant_sphere_cover"
)
ELLIPSOID_ROOT = RESULTS / (
    "formal_strict_goal_v4_3_surface_core_ellipsoid_"
    "camera_quarter_cv7.5mm_cr70mm_u1p05_stride1_vmax180mmps_"
    "affinity_perception_heavy_formal_redesign_no_coreinflate_r70_paired"
)
REPORTS = (
    ROOT / "V4_4_ADAPTIVE_IRREDUNDANT_SPHERE_RESULTS.md",
    ROOT / "V4_3_NO_CORE_REINFLATION_FORMAL_RESULTS.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_run(root: Path) -> tuple[Path, dict]:
    summaries = list(root.rglob("summary.json"))
    if len(summaries) != 1:
        raise AssertionError(
            f"expected exactly one summary below {root}, found {len(summaries)}"
        )
    summary_path = summaries[0]
    return summary_path.parent, json.loads(summary_path.read_text(encoding="utf-8"))


def source_files() -> list[Path]:
    sources = (SPHERE_ROOT, ELLIPSOID_ROOT, *REPORTS)
    files: list[Path] = []
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_file():
            files.append(source)
        else:
            files.extend(path for path in source.rglob("*") if path.is_file())
    return sorted(set(files), key=lambda path: path.as_posix())


def run_record(label: str, root: Path) -> dict[str, object]:
    run_dir, summary = one_run(root)
    return {
        "label": label,
        "path": run_dir.relative_to(ROOT).as_posix(),
        "representation": summary["representation"],
        "success": bool(summary["success"]),
        "cycles": int(summary["cycles"]),
        "final_error_m": float(summary["final_error_m"]),
        "final_proxy_count": int(summary["final_proxy_count"]),
        "controller_ms_p99": float(summary["controller_ms_p99"]),
        "observability_passed": bool(summary["observability_passed"]),
        "observability_late_samples": int(summary["observability_late_samples"]),
        "observability_never_observed_samples": int(
            summary["observability_never_observed_samples"]
        ),
        "summary_sha256": sha256(run_dir / "summary.json"),
        "source_hashes_recorded_by_run": summary["source_hashes"],
    }


def main() -> None:
    files = source_files()
    manifest = {
        "protocol_identity": "formal_baseline_v4_4_sphere_v4_3_ellipsoid",
        "status": "frozen_content_addressed_formal_baseline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mutation_policy": (
            "Never overwrite or post-process the two source result trees in place. "
            "Any new algorithm, audit sidecar, or rerun belongs to formal_v5_0 or "
            "a later explicitly versioned result root."
        ),
        "fairness_policy_for_next_protocol": {
            "equal_proxy_count_required": False,
            "same_causal_pointcloud_required_for_representation_ablation": True,
            "same_centervox_rule_and_parameters_required": True,
            "same_coverage_and_safety_definition_required": True,
            "certificate_requirement": (
                "For each preregistered radius/longest-radius cap, publish an "
                "inclusion-minimal cover of the complete CenterVox certificate "
                "universe relative to the declared candidate family; do not densify."
            ),
        },
        "allowed_claim_boundary": (
            "These frozen runs establish an accepted engineering baseline only. "
            "They do not by themselves provide a same-pointcloud paired comparison, "
            "global minimum-cardinality cover, or complete online observability pass."
        ),
        "runs": [
            run_record("v4.4c_adaptive_irredundant_sphere", SPHERE_ROOT),
            run_record("v4.3_no_core_reinflation_ellipsoid", ELLIPSOID_ROOT),
        ],
        "file_count": len(files),
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "frozen_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# v4.4 球型 / v4.3 椭球型正式基线冻结\n\n"
        "本目录只保存内容寻址清单，不复制或改写原始结果。当前两组作为"
        "正式历史基线永久保留；新协议不要求球和椭球代理数量相同，而要求"
        "同一因果点云、同一 CenterVox 规则、同一覆盖与安全标准。每个预注册"
        "尺度下必须生成相对于声明候选族的不可约覆盖，禁止为了显示或控制而"
        "人为加密代理。\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(OUTPUT / "frozen_manifest.json"),
                "files": len(files),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
