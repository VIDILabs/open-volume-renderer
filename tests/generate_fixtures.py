"""Emit a small self-contained scene fixture used by both the C++ and
Python test suites.

Produces, under ``--output-dir``:

    synthetic_volume.raw    # 32^3 float32 LE gaussian blob
    synthetic_scene.json    # minimal VIDi-format scene referencing it

The script is idempotent: it rewrites the same files each invocation
without touching anything else, so re-running it in-place is safe.

Called from:
    * CMake / CTest fixture (see tests/CMakeLists.txt)
    * pytest conftest.py as a fallback when CMake hasn't generated it yet
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


DIM = 32


def _gaussian_blob(dim: int) -> bytes:
    import math
    buf = bytearray(dim * dim * dim * 4)
    idx = 0
    inv = 1.0 / (dim - 1)
    for z in range(dim):
        fz = z * inv * 2.0 - 1.0
        for y in range(dim):
            fy = y * inv * 2.0 - 1.0
            for x in range(dim):
                fx = x * inv * 2.0 - 1.0
                v = math.exp(-4.0 * (fx * fx + fy * fy + fz * fz))
                struct.pack_into("<f", buf, idx, v)
                idx += 4
    return bytes(buf)


def _make_alpha_blob(res: int = 256) -> str:
    """BASE64-encoded float32 array (length = res) with a ramp from 0 -> 1.

    The VIDi serializer expects opacities this way rather than embedded in
    colorControls; without it, all opacities stay 0 and the whole volume
    renders transparent (i.e. all-black frames). Ramp skips the first ~5%
    so the empty background stays empty.
    """
    import base64
    import struct
    out = bytearray()
    for i in range(res):
        t = i / (res - 1)
        a = 0.0 if t < 0.05 else min(1.0, (t - 0.05) / 0.5)
        out += struct.pack("<f", a)
    return base64.b64encode(bytes(out)).decode("ascii")


def _write_json(path: Path, volume_abs: Path, dim: int) -> None:
    alpha_b64 = _make_alpha_blob(256)
    scene = {
        "dataSource": [{
            "dimensions": {"x": dim, "y": dim, "z": dim},
            "endian": "LITTLE_ENDIAN",
            "fileName": str(volume_abs),
            "fileUpperLeft": False,
            "format": "REGULAR_GRID_RAW_BINARY",
            "id": 1,
            "name": "synthetic_volume.raw",
            "offset": 0,
            "type": "FLOAT",
        }],
        "view": {
            "camera": {
                "center": {"x": (dim - 1) / 2, "y": (dim - 1) / 2, "z": (dim - 1) / 2},
                "eye":    {"x": (dim - 1) / 2, "y": (dim - 1) / 2, "z": dim * 3.0},
                "up":     {"x": 0.0, "y": 1.0, "z": 0.0},
                "fovy": 45,
                "projectionMode": "PERSPECTIVE",
                "zNear": 0.1,
                "zFar":  1000.0,
            },
            "volume": {
                "dataId": 1,
                "sampleDistance": 0.5,
                "scalarMappingRange": {"minimum": 0.0, "maximum": 1.0},
                "transferFunction": {
                    "alphaArray": {"data": alpha_b64, "encoding": "BASE64"},
                    "colorControls": [
                        {"color": {"r": 0.0, "g": 0.0, "b": 0.0}, "position": 0.0},
                        {"color": {"r": 1.0, "g": 0.5, "b": 0.0}, "position": 0.5},
                        {"color": {"r": 1.0, "g": 1.0, "b": 1.0}, "position": 1.0},
                    ],
                    "resolution": 256,
                },
                "visible": True,
            },
        },
    }
    path.write_text(json.dumps(scene, indent=2))


def generate(output_dir: Path, dim: int = DIM) -> tuple[Path, Path]:
    """Write volume + scene JSON into ``output_dir``; return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vol_path = output_dir / "synthetic_volume.raw"
    scene_path = output_dir / "synthetic_scene.json"
    vol_path.write_bytes(_gaussian_blob(dim))
    _write_json(scene_path, vol_path, dim)
    return vol_path, scene_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--dim", type=int, default=DIM)
    args = ap.parse_args()
    vol, scene = generate(args.output_dir, args.dim)
    print(f"wrote {vol}")
    print(f"wrote {scene}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
