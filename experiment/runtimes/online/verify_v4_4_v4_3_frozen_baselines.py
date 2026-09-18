"""Verify every source artifact named by the frozen v4.4/v4.3 manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = (
    ROOT
    / "formal_results"
    / "final_two_camera"
    / "protocol_v4_4_v4_3_formal_frozen"
    / "frozen_manifest.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures: list[dict[str, object]] = []
    for expected in manifest["files"]:
        path = ROOT / expected["path"]
        if not path.is_file():
            failures.append({"path": expected["path"], "error": "missing"})
            continue
        actual_bytes = path.stat().st_size
        actual_hash = sha256(path)
        if (
            actual_bytes != int(expected["bytes"])
            or actual_hash != expected["sha256"]
        ):
            failures.append(
                {
                    "path": expected["path"],
                    "error": "content_mismatch",
                    "expected_bytes": expected["bytes"],
                    "actual_bytes": actual_bytes,
                    "expected_sha256": expected["sha256"],
                    "actual_sha256": actual_hash,
                }
            )
    result = {
        "protocol_identity": manifest["protocol_identity"],
        "verified_files": len(manifest["files"]) - len(failures),
        "expected_files": len(manifest["files"]),
        "passed": not failures,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
