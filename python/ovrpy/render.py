"""Render an OVR scene JSON to a PNG image.

Exposed as the ``ovrpy-render`` console script via ``pyproject.toml``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import ovrpy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ovrpy-render",
        description="Render a scene file with ovrpy and write a PNG.",
    )
    parser.add_argument("scene", type=Path, help="Path to a scene JSON/USD file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("render.png"),
        help="Output PNG path. Defaults to render.png.",
    )
    parser.add_argument(
        "--backend",
        default="optix7",
        choices=("optix7", "ospray"),
        help="Renderer backend. Defaults to optix7.",
    )
    parser.add_argument("--width", type=int, default=640, help="Framebuffer width.")
    parser.add_argument("--height", type=int, default=480, help="Framebuffer height.")
    parser.add_argument("--spp", type=int, default=4, help="Samples per pixel.")
    parser.add_argument(
        "--density-scale",
        type=float,
        default=50.0,
        help="Volume density multiplier.",
    )
    parser.add_argument(
        "--volume-sampling-rate",
        type=float,
        default=1.0,
        help="Volume sampling rate.",
    )
    parser.add_argument(
        "--path-tracing",
        action="store_true",
        help="Enable path tracing instead of ray marching.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    try:
        from PIL import Image
    except Exception as exc:
        raise SystemExit(
            "Pillow is required to write PNG output. Install ovrpy with the "
            "default dependency set: pip install -e ."
        ) from exc

    scene = ovrpy.create_scene(str(args.scene))

    fbsize = ovrpy.vec2i()
    fbsize.x = args.width
    fbsize.y = args.height

    rgba = ovrpy.render_scene_to_image(
        args.backend,
        scene,
        fbsize,
        sample_per_pixel=args.spp,
        path_tracing=args.path_tracing,
        volume_sampling_rate=args.volume_sampling_rate,
        volume_density_scale=args.density_scale,
        clip=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgba * 255.0 + 0.5).astype(np.uint8), mode="RGBA").save(args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
