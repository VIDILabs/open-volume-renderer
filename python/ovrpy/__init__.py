"""ovrpy - Python bindings for the Open Volume Renderer (OVR).

The native pybind11 module lives at ``ovrpy._core`` and is re-exported here
so that ``import ovrpy`` works exactly as it did when the bindings shipped
as a single bare extension module.

If the native extension fails to load because a *system* shared library is
missing (libcuda, libGL, libX11, libvulkan, ...), the stock dynamic-loader
error is opaque -- it names the file but not how to install it. We catch
the ImportError and translate the "lib<X>.so: cannot open shared object
file" pattern into a message that names the package on the common distros.
"""
from __future__ import annotations

import re

__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# Libs we deliberately *don't* bundle in the wheel because they're owned by
# the host (driver, system OpenGL/X11/Vulkan stacks). Keys are the soname
# strings the dynamic loader prints; values are (human label, install hint).
# ---------------------------------------------------------------------------
_SYSTEM_LIB_HINTS: dict[str, tuple[str, str]] = {
    "libcuda.so.1": (
        "NVIDIA CUDA driver runtime",
        "Install the proprietary NVIDIA driver matching your GPU.\n"
        "    Ubuntu/Debian: sudo apt install nvidia-driver-<version>\n"
        "    RHEL/Fedora:   sudo dnf install nvidia-driver",
    ),
    "libGL.so.1": (
        "system OpenGL loader",
        "    Ubuntu/Debian: sudo apt install libgl1\n"
        "    RHEL/Fedora:   sudo dnf install mesa-libGL",
    ),
    "libOpenGL.so.0": (
        "system OpenGL ABI (GLVND)",
        "    Ubuntu/Debian: sudo apt install libopengl0\n"
        "    RHEL/Fedora:   sudo dnf install libglvnd-opengl",
    ),
    "libX11.so.6": (
        "X11 client library (needed by the interactive viewer; install or "
        "switch to a headless backend like ospray with no GUI)",
        "    Ubuntu/Debian: sudo apt install libx11-6\n"
        "    RHEL/Fedora:   sudo dnf install libX11",
    ),
    "libvulkan.so.1": (
        "Vulkan loader",
        "    Ubuntu/Debian: sudo apt install libvulkan1\n"
        "    RHEL/Fedora:   sudo dnf install vulkan-loader",
    ),
}

_MISSING_LIB_RE = re.compile(
    r"(lib[\w\-+.]+\.so(?:\.\d+)*): cannot open shared object file"
)


def _translate_loader_error(exc: ImportError) -> ImportError:
    """Return a friendlier ImportError when *exc* names a missing shared lib."""
    match = _MISSING_LIB_RE.search(str(exc))
    if not match:
        return exc

    missing = match.group(1)
    if missing in _SYSTEM_LIB_HINTS:
        what, install_hint = _SYSTEM_LIB_HINTS[missing]
        return ImportError(
            f"ovrpy could not load its native extension because the {what} "
            f"({missing}) is not installed on this system.\n\n"
            f"{install_hint}\n\n"
            f"Original loader error: {exc}"
        )

    return ImportError(
        f"ovrpy could not load its native extension: required shared "
        f"library {missing} was not found. ovrpy bundles its own renderer "
        f"and OSPRay closure, so this almost always means a *system* library "
        f"is missing on the host.\n\n"
        f"Original loader error: {exc}"
    )


try:
    from . import _core
    from ._core import *  # noqa: F401,F403
except ImportError as _exc:
    raise _translate_loader_error(_exc) from _exc
