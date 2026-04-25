"""Rendering regression: render our synthetic scene and compare against
a committed golden PNG using PSNR + SSIM.

Baselines live under ``test/fixtures/golden/<backend>/<scene>.png``.
On first run (or when baselines are deliberately refreshed) pass
``--update-baselines`` to pytest to write the PNG instead of asserting.

Tolerances are deliberately loose because:

* OSPRay CPU traversal and OptiX 7 GPU traversal take different paths, so
  we store per-backend baselines.
* Blue-noise dithering and any tiny RNG differences (driver version,
  compiler) can shift pixels by a few LSBs.

Backends diverge enough that we **don't** compare optix7 vs ospray output
directly - each has its own golden file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ovrpy


# --------------------------------------------------------------------------
# Tolerances
# --------------------------------------------------------------------------
# These values are intentionally generous. The first-time-run baselines
# are generated on whichever machine calls `--update-baselines`, so small
# numerical drift across driver/library versions is expected.
TOLERANCES = {
    "ospray": {"psnr_min": 30.0, "ssim_min": 0.90},
    "optix7": {"psnr_min": 30.0, "ssim_min": 0.90},
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _render_deterministic(backend: str, scene, fbsize, density_scale: float) -> np.ndarray:
    r = ovrpy.create_renderer(backend)
    r.set_fbsize(fbsize)
    r.init([], scene, scene.camera)
    # Ray marching is dramatically more stable than path tracing for
    # regression purposes and doesn't need a seeded RNG.
    r.set_path_tracing(0)
    # Several samples per pixel both reduces noise and, on optix7, gets us
    # past a one-ray-per-pixel empty-frame corner case.
    r.set_sample_per_pixel(4)
    r.set_frame_accumulation(False)
    r.set_volume_sampling_rate(1.0)
    # See conftest._OVR_TEST_DENSITY_SCALE for why this is >1.0.
    r.set_volume_density_scale(density_scale)
    r.commit()
    r.render()
    r.swap()
    fb = ovrpy.FrameBufferData()
    r.mapframe(fb)
    rgba = np.asarray(fb.rgba(), dtype=np.float32).copy().reshape(fbsize.y, fbsize.x, 4)
    # Scrub any NaN/Inf pixels (see test_framebuffer.py for context) so the
    # downstream PNG cast doesn't blow up.
    rgba = np.nan_to_num(rgba, nan=0.0, posinf=1.0, neginf=0.0)
    # Clip to [0, 1] for 8-bit PNG storage - anything outside is HDR glow
    # we don't care about at this stage.
    rgba = np.clip(rgba, 0.0, 1.0)
    return rgba


def _to_png_bytes(rgba: np.ndarray) -> bytes:
    from PIL import Image
    u8 = (rgba * 255.0 + 0.5).astype(np.uint8)
    img = Image.fromarray(u8, mode="RGBA")
    import io as _io
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _load_png(path: Path) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("RGBA")
    return np.asarray(img, dtype=np.uint8).astype(np.float32) / 255.0


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    # a, b are float arrays in [0, 1]
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0  # treat perfect match as a big number
    return 10.0 * float(np.log10(1.0 / mse))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as sk_ssim
    # Average SSIM over RGB channels; ignore alpha for the comparison.
    rgb_a = a[..., :3]
    rgb_b = b[..., :3]
    return float(sk_ssim(rgb_a, rgb_b, channel_axis=-1, data_range=1.0))


# --------------------------------------------------------------------------
# The actual test
# --------------------------------------------------------------------------
@pytest.mark.golden
def test_render_matches_baseline(backend: str,
                                 scene,
                                 fbsize,
                                 fixtures_dir: Path,
                                 update_baselines: bool,
                                 test_density_scale: float,
                                 tmp_path: Path):
    rgba = _render_deterministic(backend, scene, fbsize, test_density_scale)

    golden_dir = fixtures_dir / "golden" / backend
    golden_path = golden_dir / "synthetic_scene.png"

    # `--update-baselines` is the *only* path that writes a fresh PNG and
    # short-circuits with skip. Everything else - including a missing
    # baseline - is treated as a hard regression so CI can't pass with no
    # actual comparison happening. To bootstrap a new backend, the developer
    # runs once with `--update-baselines` and commits the resulting PNG.
    #
    # The mkdir lives inside the update-baselines branch so a normal test
    # run never modifies the source tree (the directory already exists for
    # the canonical backends thanks to the committed .gitkeep files; we
    # only create it when the user has explicitly opted into writing).
    if update_baselines:
        from PIL import Image
        golden_dir.mkdir(parents=True, exist_ok=True)
        u8 = (rgba * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(u8, mode="RGBA").save(golden_path)
        pytest.skip(
            f"baseline overwritten; wrote {golden_path}. Re-run without "
            "--update-baselines to enforce the regression."
        )

    if not golden_path.exists():
        pytest.fail(
            f"Golden baseline missing: {golden_path}\n"
            "  - Generate it on a reference machine with:\n"
            "      pytest test/python/test_rendering_regression.py --update-baselines\n"
            f"  - Then `git add` the resulting PNG and commit so CI can compare."
        )

    baseline = _load_png(golden_path)
    if baseline.shape != rgba.shape:
        pytest.fail(
            f"baseline shape {baseline.shape} != current render shape "
            f"{rgba.shape}; regenerate with --update-baselines"
        )

    tol = TOLERANCES.get(backend, {"psnr_min": 30.0, "ssim_min": 0.9})
    psnr = _psnr(rgba, baseline)
    ssim = _ssim(rgba, baseline)

    # Write artifacts so CI can upload them on failure.
    from PIL import Image
    Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(tmp_path / "actual.png")
    diff = np.abs(rgba - baseline)
    diff_vis = np.clip(diff * 10.0, 0, 1)
    Image.fromarray((diff_vis * 255).astype(np.uint8), "RGBA").save(tmp_path / "diff.png")

    assert psnr >= tol["psnr_min"], (
        f"{backend}: PSNR={psnr:.2f}dB < {tol['psnr_min']}dB; diff in {tmp_path}"
    )
    assert ssim >= tol["ssim_min"], (
        f"{backend}: SSIM={ssim:.4f} < {tol['ssim_min']}; diff in {tmp_path}"
    )


@pytest.mark.golden
@pytest.mark.slow
def test_render_with_high_spp_path_tracing_is_mostly_finite(backend, scene, fbsize):
    """Sanity: even with path tracing + many samples the output stays
    mostly finite. Not a regression against a baseline - path tracing is
    stochastic. Tolerate <= 1% NaN/Inf pixels (known optix7 quirk)."""
    r = ovrpy.create_renderer(backend)
    r.set_fbsize(fbsize)
    r.init([], scene, scene.camera)
    r.set_path_tracing(1)
    r.set_sample_per_pixel(16)
    r.commit()
    r.render()
    r.swap()
    fb = ovrpy.FrameBufferData()
    r.mapframe(fb)
    rgba = np.asarray(fb.rgba(), dtype=np.float32).copy()
    n_bad = int((~np.isfinite(rgba)).sum())
    assert n_bad / rgba.size < 0.01, f"{n_bad} non-finite pixels"
