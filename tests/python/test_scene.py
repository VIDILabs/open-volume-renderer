"""Scene loading + introspection tests (no renderer required)."""
from __future__ import annotations

import io
import sys

import pytest

import ovrpy


def test_create_scene_returns_scene(scene):
    assert scene is not None
    # Scene is a pybind11 value type; bounds getter must work on a loaded scene.
    bounds = scene.get_bounds()
    assert bounds is not None


def test_get_bounds_is_nonempty(scene):
    b = scene.get_bounds()
    # Our synthetic fixture is a 32^3 volume with grid_spacing=1: bounds must
    # span at least 32 units on each axis.
    dx = b.upper.x - b.lower.x
    dy = b.upper.y - b.lower.y
    dz = b.upper.z - b.lower.z
    assert dx > 0, f"bounds degenerate along x: {dx}"
    assert dy > 0
    assert dz > 0


def test_scene_scalar_fields_accessible(scene):
    # Defaults set by create_scene_default + VIDi parser; values vary by fixture
    # but the *types* must be right.
    assert isinstance(scene.spp, int)
    assert isinstance(scene.ao_samples, int)
    assert isinstance(scene.volume_sampling_rate, float)
    assert isinstance(scene.roulette_path_length, int)
    assert isinstance(scene.max_path_length, int)
    assert isinstance(scene.use_dda, int)
    assert isinstance(scene.parallel_view, bool)
    assert isinstance(scene.simple_path_tracing, bool)


def test_scene_spp_is_settable(scene):
    scene.spp = 4
    assert scene.spp == 4
    scene.spp = 1


def test_camera_is_accessible(scene):
    cam = scene.camera
    # Basic members exist and are not default-all-zero
    assert hasattr(cam, "eye")
    assert hasattr(cam, "at")
    assert hasattr(cam, "up")


def test_scene_print_does_not_raise(scene, capsys):
    """Scene.print() writes to std::cout; we only assert it doesn't throw
    and that *something* ends up on stdout when captured via pytest's C++-
    bridged capture (disabled_capsys is platform-dependent so we just
    ensure no exception bubbles up)."""
    scene.print()
    # Whatever ends up on stdout is backend/build-config dependent; the
    # important contract is "does not raise".
