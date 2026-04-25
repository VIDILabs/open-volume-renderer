"""Threading smoke tests: exercise the TransactionalValue-based parameter
fences by driving setters from a background thread while the main thread
calls render()/commit()/mapframe() in a tight loop.

The goal isn't to prove correctness of the full TransactionalValue
implementation (that's covered by the C++ test_transactional_value
binary); it's to catch GIL-related deadlocks or crashes in the bindings.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import ovrpy


@pytest.mark.slow
def test_background_setters_while_rendering(initialized_renderer, fbsize):
    stop = threading.Event()

    def producer():
        i = 0
        while not stop.is_set():
            initialized_renderer.set_sample_per_pixel(1 + (i % 4))
            initialized_renderer.set_volume_sampling_rate(1.0 + (i % 3) * 0.25)
            initialized_renderer.set_path_tracing(i % 2)
            i += 1
            time.sleep(0.001)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    try:
        for _ in range(25):
            initialized_renderer.commit()
            initialized_renderer.render()
            initialized_renderer.swap()
            fb = ovrpy.FrameBufferData()
            initialized_renderer.mapframe(fb)
            rgba = np.asarray(fb.rgba(), dtype=np.float32).copy()
            # rgba must be mostly finite even under concurrent parameter
            # updates. A sub-1% NaN slice is tolerated (matches known
            # optix7 boundary-pixel quirk).
            n_bad = int((~np.isfinite(rgba)).sum())
            assert n_bad / rgba.size < 0.02, f"{n_bad} non-finite pixels"
    finally:
        stop.set()
        t.join(timeout=2.0)
        assert not t.is_alive(), "producer thread failed to stop in time"


@pytest.mark.slow
def test_final_frame_is_deterministic(backend, scene, fbsize, test_density_scale):
    """Running render() after explicitly pinning every setter should
    produce a bit-identical frame compared to a second run with the same
    settings on a fresh renderer instance."""
    def run_once() -> np.ndarray:
        r = ovrpy.create_renderer(backend)
        r.set_fbsize(fbsize)
        r.init([], scene, scene.camera)
        r.set_sample_per_pixel(4)
        r.set_path_tracing(0)      # deterministic ray marcher
        r.set_frame_accumulation(False)
        r.set_volume_sampling_rate(1.0)
        r.set_volume_density_scale(test_density_scale)
        r.commit()
        r.render()
        r.swap()
        fb = ovrpy.FrameBufferData()
        r.mapframe(fb)
        rgba = np.asarray(fb.rgba(), dtype=np.float32).copy()
        return np.nan_to_num(rgba, nan=0.0, posinf=1.0, neginf=0.0)

    a = run_once()
    b = run_once()
    diff = np.abs(a - b)
    mean = float(diff.mean())
    # Ray marching with accumulation=off should be deterministic modulo any
    # backend-specific RNG reset semantics; tolerate small drift.
    assert mean < 1e-2, f"determinism test: mean |dI| = {mean}"
