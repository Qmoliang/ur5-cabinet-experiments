"""Build the native AVX2 MVT DLL with the local MinGW compiler."""

from __future__ import annotations

from pathlib import Path
import subprocess
import struct


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "native_mvt" / "native_mvt.cpp"
OUTPUT = ROOT / "native_mvt" / "native_mvt.dll"


def main() -> None:
    if struct.calcsize("P") != 8:
        raise RuntimeError("the native MVT must be built for 64-bit Python")
    batch = ROOT / "build_native_mvt.bat"
    subprocess.run(["cmd.exe", "/d", "/c", str(batch)], check=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
