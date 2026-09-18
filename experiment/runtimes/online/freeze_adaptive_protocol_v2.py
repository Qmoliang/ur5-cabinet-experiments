"""Freeze the accepted adaptive-radius online experiment as protocol v2.

The script does not copy, rename, or modify any source artifact. It writes a
content-addressed manifest whose hashes make later accidental changes visible.
Historical directory names containing v3 or v5 are engineering iteration
labels; the manifest assigns their paper-protocol identity.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "formal_results" / "final_two_camera"
OUTPUT = FINAL / "protocol_v2_adaptive_radius_frozen"

SOURCES = (
    FINAL / "formal_evidence_final_v5",
    FINAL / "formal_protocol_v3_evidence_manifest_v5.json",
    FINAL / "static_geometry_gate_final",
    FINAL / "known_packed_equivalence_final_v3",
    FINAL / "final_mvt_simd_benchmark_v5.json",
    ROOT / "figures" / "formal_v5",
    ROOT / "tables" / "formal_v5",
    ROOT / "FORMAL_V5_VIEWING_AND_REPRODUCTION.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files() -> list[Path]:
    files: list[Path] = []
    for source in SOURCES:
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_file():
            files.append(source)
        else:
            files.extend(path for path in source.rglob("*") if path.is_file())
    return sorted(set(files), key=lambda path: path.as_posix())


def main() -> None:
    files = source_files()
    runs = []
    online_root = FINAL / "formal_evidence_final_v5"
    for summary_path in sorted(online_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runs.append(
            {
                "path": summary_path.parent.relative_to(ROOT).as_posix(),
                "representation": summary["representation"],
                "success": bool(summary["success"]),
                "cycles": int(summary["cycles"]),
                "final_error_m": float(summary["final_error_m"]),
                "final_proxy_count": int(summary["final_proxy_count"]),
                "source_summary_sha256": sha256(summary_path),
            }
        )
    manifest = {
        "paper_protocol_identity": "protocol_v2_adaptive_radius_online",
        "historical_engineering_labels": [
            "formal_protocol_v3_final_two_camera",
            "formal_evidence_final_v5",
        ],
        "status": "frozen_content_addressed_reference",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mutation_policy": (
            "Do not overwrite source artifacts. Any changed hash creates a new "
            "protocol version and invalidates this manifest."
        ),
        "radius_policy": {
            "sphere": "adaptive cluster enclosing radius",
            "ellipsoid": "adaptive PCA shape plus directional uncertainty",
            "uniform_radius": False,
        },
        "runs": runs,
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
    manifest_path = OUTPUT / "frozen_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# 协议 v2：自适应半径在线实验冻结说明\n\n"
        "本目录不复制或改写原始数据；frozen_manifest.json 以 SHA-256 "
        "固定当前已接受实验的全部原始证据。历史路径中的 v3/v5 是工程迭代号，"
        "论文协议身份统一解释为 protocol_v2_adaptive_radius_online。\n\n"
        "该版本仍使用融合双相机点云和逐簇自适应球/椭球尺度，保留为有效的"
        "自适应代理实验。后续统一半径 30/50/70 mm 对照写入独立 v3 目录，"
        "不得覆盖本清单引用的任何文件。\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
