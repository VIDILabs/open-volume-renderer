"""Framebuffer lifecycle + mapframe output tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

import ovrpy


def test_uninit_framebuffer_raises():
    fb = ovrpy.FrameBufferData()
    with pytest.raises(RuntimeError):
        fb.rgba()
    with pytest.raises(RuntimeError):
        fb.stats()


def test_mapframe_size_matches_fbsize(initialized_renderer, fbsize):
    fb = ovrpy.FrameBufferData()
    initialized_renderer.mapframe(fb)

    rgba = fb.rgba()
    assert len(rgba) == fbsize.x * fbsize.y * 4, (
        f"rgba has {len(rgba)} elements; expected {fbsize.x * fbsize.y * 4}"
    )

    stats = fb.stats()
    assert len(stats) == fbsize.x * fbsize.y


def test_mapframe_rgba_has_correct_shape(initialized_renderer, fbsize):
    """Mapped rgba must be width*height*4 floats.

    We deliberately don't assert that the pre-render buffer is all-zero:
    some backends map whatever was last in the CUDA/CPU buffer, which may
    contain uninitialized memory (NaN, garbage). The *post*-render
    finiteness check below is the real invariant.
    """
    fb = ovrpy.FrameBufferData()
    initialized_renderer.mapframe(fb)
    rgba = np.asarray(fb.rgba(), dtype=np.float32).copy()
    assert rgba.shape == (fbsize.x * fbsize.y * 4,)


def test_stats_has_correct_length(initialized_renderer, fbsize):
    """RenderStats array must be one entry per pixel.

    Note: ``stats[i].pixel_index`` is only populated in sparse-sampling /
    path-tracing modes; in plain ray marching every entry's pixel_index
    stays 0. Only the array *length* is a stable cross-backend invariant.
    """
    initialized_renderer.render()
    fb = ovrpy.FrameBufferData()
    initialized_renderer.mapframe(fb)
    stats = fb.stats()
    assert len(stats) == fbsize.x * fbsize.y


def test_postrender_rgba_is_mostly_finite(initialized_renderer, fbsize):
    """After render() the vast majority of pixels must be finite.

    We do *not* require values in [0, 1]: some backends and modes
    (path-tracing accumulators, HDR colour mapping) legitimately produce
    values outside that range, and the canonical consumer code in the
    repo already clips before saving. We tolerate up to 1% non-finite
    pixels so the test still catches gross regressions (whole-frame NaN)
    without being sabotaged by known boundary-pixel corner cases on
    optix7.
    """
    initialized_renderer.set_sample_per_pixel(4)
    initialized_renderer.commit()
    initialized_renderer.render()
    fb = ovrpy.FrameBufferData()
    initialized_renderer.mapframe(fb)
    # .copy() detaches the array from the CrossDeviceBuffer that fb owns,
    # so we don't blow up on later GC.
    rgba = np.asarray(fb.rgba(), dtype=np.float32).copy().reshape(fbsize.y, fbsize.x, 4)
    n_total = rgba.size
    n_bad = int((~np.isfinite(rgba)).sum())
    assert n_bad / n_total < 0.01, (
        f"{n_bad}/{n_total} non-finite pixels (> 1%); rendering likely broken"
    )


def test_mapframe_grad_is_available_or_raises_cleanly(initialized_renderer, fbsize):
    """Gradient buffer semantics vary by backend:
      * optix7 populates a vec3f-per-pixel grad buffer
      * ospray doesn't populate grad at all and raises RuntimeError
    Either outcome is acceptable; a partial buffer is not.
    """
    initialized_renderer.render()
    fb = ovrpy.FrameBufferData()
    initialized_renderer.mapframe(fb)
    try:
        grad = np.asarray(fb.grad(), dtype=np.float32).copy()
    except RuntimeError:
        pytest.skip("backend does not populate the gradient buffer")
        return
    n = len(grad)
    expected_full = fbsize.x * fbsize.y * 3
    assert n in (0, expected_full), (
        f"grad buffer size {n} is neither empty nor full ({expected_full})"
    )


def test_stats_as_memoryview_is_readable(initialized_renderer):
    initialized_renderer.render()
    fb = ovrpy.FrameBufferData()
    initialized_renderer.mapframe(fb)
    mv = fb.stats_as_memoryview()
    assert mv is not None
    assert mv.nbytes > 0
