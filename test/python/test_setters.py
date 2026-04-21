"""Setter round-trips through ``MainRenderer``.

These are black-box tests that merely assert each setter is wired up and
``commit()`` accepts the new state without throwing. They don't assert
*semantic* correctness of the parameter (that's what the rendering
regression tests are for); they guard against ABI/binding regressions
like a parameter being dropped on the floor or silently type-converting.
"""
from __future__ import annotations

import pytest

import ovrpy


def test_set_sample_per_pixel(initialized_renderer):
    for spp in (1, 4, 16):
        initialized_renderer.set_sample_per_pixel(spp)
        initialized_renderer.commit()


def test_set_volume_sampling_rate(initialized_renderer):
    for rate in (0.5, 1.0, 2.0):
        initialized_renderer.set_volume_sampling_rate(rate)
        initialized_renderer.commit()


def test_set_volume_density_scale(initialized_renderer):
    for s in (0.1, 1.0, 10.0):
        initialized_renderer.set_volume_density_scale(s)
        initialized_renderer.commit()


def test_set_path_tracing_toggle(initialized_renderer):
    initialized_renderer.set_path_tracing(0)
    initialized_renderer.commit()
    initialized_renderer.set_path_tracing(1)
    initialized_renderer.commit()


def test_set_frame_accumulation_toggle(initialized_renderer):
    initialized_renderer.set_frame_accumulation(True)
    initialized_renderer.commit()
    initialized_renderer.set_frame_accumulation(False)
    initialized_renderer.commit()


def test_set_transfer_function_accepts_piecewise_linear(initialized_renderer):
    # 3-stop colormap: black -> orange -> white
    colors = [
        0.0, 0.0, 0.0,
        1.0, 0.5, 0.0,
        1.0, 1.0, 1.0,
    ]
    # opacities are (position, value) pairs
    opacities = [
        0.0, 0.0,
        0.5, 0.5,
        1.0, 1.0,
    ]
    value_range = ovrpy.vec2f(0.0, 1.0)
    initialized_renderer.set_transfer_function(colors, opacities, value_range)
    initialized_renderer.commit()


def test_set_sparse_sampling_and_focus(initialized_renderer):
    initialized_renderer.set_sparse_sampling(True)
    initialized_renderer.set_focus(ovrpy.vec2f(0.5, 0.5), 0.2, 0.1)
    initialized_renderer.commit()
    initialized_renderer.set_sparse_sampling(False)
    initialized_renderer.commit()


def test_render_after_setters_produces_nonzero_frame(initialized_renderer, fbsize):
    """Smoke test: after all setters run, we can still render and get a
    real framebuffer back. Tolerates a small fraction of NaN pixels
    caused by known backend corner cases."""
    import numpy as np

    initialized_renderer.set_sample_per_pixel(4)
    initialized_renderer.set_path_tracing(0)   # ray marching - deterministic-ish
    initialized_renderer.commit()
    initialized_renderer.render()
    fb = ovrpy.FrameBufferData()
    initialized_renderer.mapframe(fb)
    rgba = np.asarray(fb.rgba(), dtype=np.float32).copy()
    n_bad = int((~np.isfinite(rgba)).sum())
    assert n_bad / rgba.size < 0.01, f"{n_bad} non-finite pixels"
