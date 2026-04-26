"""Console-entry shims for the bundled C++ executables.

The wheel ships ``renderapp`` and ``renderbatch`` inside
``<site-packages>/ovrpy/bin/`` (see ``apps/CMakeLists.txt``). Putting the
binaries directly on ``$PATH`` from there would require an
``INSTALL_RPATH`` that crosses the bin/ <-> site-packages/ovrpy/
boundary - brittle because the relative path depends on the Python
version (``python3.11``, ``python3.12``, ...).

Instead, ``[project.scripts]`` in ``pyproject.toml`` maps the console
names to the shim functions below; pip emits a tiny Python launcher in
``<env>/bin/`` that calls our shim, which in turn ``os.execv``s the real
binary. The binary's own ``INSTALL_RPATH=$ORIGIN/..`` then resolves
``librenderlib.so`` + the rest of the closure as siblings inside the
package, with no Python-version-specific paths involved.

We anchor the lookup on ``_core.__file__`` rather than ``__file__``
because under an editable install (``pip install -e .`` /
``uv sync --extra test``), this module's ``__file__`` resolves to the
*source* tree (``<repo>/python/ovrpy/_apps.py``), while the binaries
live next to the *installed* extension. ``_core.__file__`` always points
at the real installed ``.so`` regardless of editable / wheel mode.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from . import _core


def _bin_dir() -> Path:
    return Path(_core.__file__).resolve().parent / "bin"


def _exec(name: str) -> None:
    binary = _bin_dir() / name
    if not binary.is_file():
        raise SystemExit(
            f"ovrpy: bundled binary '{name}' not found at {binary}.\n"
            "This usually means ovrpy was built with `OVR_BUILD_APPS=OFF`. "
            "Reinstall with the default cmake.define block, or set "
            "`OVR_BUILD_APPS=ON` explicitly:\n"
            "    pip install . -C cmake.define.OVR_BUILD_APPS=ON"
        )
    os.execv(str(binary), [str(binary), *sys.argv[1:]])


def renderapp() -> None:
    """Launch the interactive GLFW + ImGui viewer."""
    _exec("renderapp")


def renderbatch() -> None:
    """Launch the offline render-to-PNG driver."""
    _exec("renderbatch")
