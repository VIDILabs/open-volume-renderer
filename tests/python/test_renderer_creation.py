"""Renderer factory tests."""
from __future__ import annotations

import pytest

import ovrpy


def test_create_renderer_known_backend_returns_object(backend):
    r = ovrpy.create_renderer(backend)
    assert r is not None


def test_create_renderer_unknown_name_raises():
    # Any name that isn't optix7 or ospray should fail. create_renderer
    # tries dynamic loading first, then throws - we don't care which branch
    # fires, just that *something* is raised.
    with pytest.raises(Exception):
        ovrpy.create_renderer("__definitely_not_a_backend__")


def test_available_backends_is_nonempty(available_backends):
    # If neither backend builds, the whole suite is meaningless.
    assert available_backends, (
        "No backends available: build with OVR_BUILD_DEVICE_OPTIX7=ON "
        "and/or OVR_BUILD_DEVICE_OSPRAY=ON"
    )
