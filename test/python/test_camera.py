"""Camera setter parity tests.

These reuse the ``renderer`` fixture so there's only one renderer instance
per test - two Python-level ``ovrpy.create_renderer()`` calls back to back
seem to stress optix7's device-context teardown enough to crash Python's
GC on some hosts. The contract we actually want to verify
(``set_camera(Camera)`` and ``set_camera_vectors(...)`` reach the same
render state) can be exercised by toggling between the two setters on a
single renderer.
"""
from __future__ import annotations

import gc

import numpy as np
import pytest

import ovrpy


def _render(renderer) -> np.ndarray:
    renderer.commit()
    renderer.render()
    fb = ovrpy.FrameBufferData()
    renderer.mapframe(fb)
    # .copy() makes the returned array independent of the buffer owned by
    # the CrossDeviceBuffer that `fb` holds; otherwise the array aliases
    # backend memory and a later renderer teardown corrupts it. nan_to_num
    # scrubs the known optix7 boundary-pixel NaNs.
    rgba = np.asarray(fb.rgba(), dtype=np.float32).copy()
    return np.nan_to_num(rgba, nan=0.0, posinf=1.0, neginf=0.0)


def test_camera_setters_equivalent(renderer, scene, fbsize, test_density_scale):
    """``set_camera(Camera)`` and ``set_camera_vectors(eye, at, up)`` must
    produce equivalent render state when fed matching values."""
    renderer.init([], scene, scene.camera)
    # Several samples per pixel: on optix7, spp=1 happens to render an
    # empty frame for this scene (known bug at volume boundaries). The
    # density bump keeps the synthetic scene visible on both backends.
    renderer.set_sample_per_pixel(4)
    renderer.set_path_tracing(0)
    renderer.set_volume_density_scale(test_density_scale)

    eye = ovrpy.vec3f(scene.camera.eye.x, scene.camera.eye.y, scene.camera.eye.z)
    at  = ovrpy.vec3f(scene.camera.at.x,  scene.camera.at.y,  scene.camera.at.z)
    up  = ovrpy.vec3f(scene.camera.up.x,  scene.camera.up.y,  scene.camera.up.z)

    cam = ovrpy.Camera()
    cam.eye = eye; cam.at = at; cam.up = up
    renderer.set_camera(cam)
    rgba_a = _render(renderer)

    renderer.set_camera_vectors(eye, at, up)
    rgba_b = _render(renderer)

    # Both frames must actually contain content; without this the test
    # passes vacuously when a backend produces all-zero frames. An
    # all-zero result is a renderer regression, not a "skip" condition:
    # the postcondition tests in test_framebuffer.py
    # (test_postrender_frame_has_content) anchor the same invariant, so
    # if those still pass and only the camera setters produce zero, the
    # camera path itself is broken. Either way it is a real failure.
    if rgba_a.sum() == 0.0 and rgba_b.sum() == 0.0:
        pytest.fail(
            "Both set_camera and set_camera_vectors produced all-zero frames; "
            "renderer (or camera setter) is broken. Check "
            "test_postrender_frame_has_content for the global invariant."
        )

    diff = np.abs(rgba_a - rgba_b)
    mean_err = float(diff.mean())
    max_err  = float(diff.max())
    # Allow small drift from blue-noise / accumulation re-seeding, but the
    # two paths must agree to a visible-quality threshold.
    assert mean_err < 1e-2, f"mean |dI| = {mean_err}"
    assert max_err  < 5e-1, f"max  |dI| = {max_err}"


def test_camera_move_changes_image(renderer, scene, fbsize, test_density_scale):
    """Sanity: a camera offset must produce a different image.

    Using the *scene's own* camera as the baseline is important because
    the scene was authored with a camera that the renderer is known to
    handle. Arbitrary synthetic camera poses can trigger backend corner
    cases (e.g. optix7 occasionally renders empty frames for certain
    up-vector / eye combinations); the baseline-plus-offset pattern
    avoids that. The offset is a true sideways translation (right-vector
    in the camera's local frame) so both views see the same volume from
    slightly different angles.
    """
    renderer.init([], scene, scene.camera)
    renderer.set_sample_per_pixel(4)
    renderer.set_path_tracing(0)
    renderer.set_volume_density_scale(test_density_scale)
    renderer.commit()
    renderer.render()
    fb = ovrpy.FrameBufferData()
    renderer.mapframe(fb)
    baseline = np.nan_to_num(
        np.asarray(fb.rgba(), dtype=np.float32).copy(),
        nan=0.0, posinf=1.0, neginf=0.0,
    )

    # True right-vector offset in the camera's local frame:
    #   right = normalize(cross(up, forward)),
    #   eye  += right * (|forward| * 0.25)
    # This guarantees the offset is perpendicular to the view direction
    # regardless of the scene's axis alignment.
    eye = scene.camera.eye
    at  = scene.camera.at
    up  = scene.camera.up
    fwd = np.array([at.x - eye.x, at.y - eye.y, at.z - eye.z], dtype=np.float64)
    fwd_len = float(np.linalg.norm(fwd))
    assert fwd_len > 0.0, "scene camera has zero forward vector"
    up_v = np.array([up.x, up.y, up.z], dtype=np.float64)
    right = np.cross(up_v, fwd / fwd_len)
    right_norm = float(np.linalg.norm(right))
    assert right_norm > 0.0, "scene camera up is parallel to forward"
    right = right / right_norm * (fwd_len * 0.25)
    renderer.set_camera_vectors(
        ovrpy.vec3f(float(eye.x + right[0]),
                    float(eye.y + right[1]),
                    float(eye.z + right[2])),
        at,
        up,
    )
    renderer.commit()
    renderer.render()
    renderer.mapframe(fb)
    moved = np.nan_to_num(
        np.asarray(fb.rgba(), dtype=np.float32).copy(),
        nan=0.0, posinf=1.0, neginf=0.0,
    )

    # The baseline must contain content. An all-zero baseline is a
    # renderer regression, not a "skip" condition:
    # test_postrender_frame_has_content (test_framebuffer.py) anchors
    # this invariant. If that test passes and *this* baseline is zero,
    # something specific to this code path is broken; either way fail.
    if baseline.sum() == 0.0:
        pytest.fail(
            "Backend rendered all-zero with the scene's own camera; "
            "renderer is broken (the synthetic scene is configured to "
            "render visibly via the test_density_scale fixture). See "
            "test_postrender_frame_has_content for the global invariant."
        )

    diff_rms = float(np.sqrt(np.mean((baseline - moved) ** 2)))
    assert diff_rms > 1e-3, (
        f"Camera offset produced RMS-diff = {diff_rms}; "
        "the camera setter may not be propagating to the renderer."
    )
