# Open Volume Renderer (OVR)

[![CI](https://github.com/wilsonCernWq/open-volume-renderer/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/wilsonCernWq/open-volume-renderer/actions/workflows/main.yml)

![Expected Rendering Result](./data/example.jpg)

OVR is a CUDA + OSPRay volume rendering framework. It ships:

- A C++/CUDA library (`renderlib`) with two pluggable backends:
  - **OptiX 7** (NVIDIA GPU) for hardware-accelerated ray tracing
  - **OSPRay** (Intel CPU) for high-quality CPU rendering, including
    distributed rendering hooks
- Two reference applications:
  - `renderapp` — interactive GLFW + ImGui viewer
  - `renderbatch` — offline render-to-PNG driver
- A Python package (`ovrpy`) built with pybind11, including the
  `ovrpy-render` console script
- A unified test suite (CTest + doctest + pytest) with optional
  PSNR/SSIM rendering regression

For build-system internals, packaging, CI, and contributor notes see
[`DEV.md`](DEV.md). Test infrastructure (markers, golden-image workflow,
coverage) is documented in [`tests/README.md`](tests/README.md).

## Repository layout

```
.
├── apps/                   renderapp + renderbatch executables
├── ovr/                    core library (rendercommon, renderlib, devices/)
│   ├── devices/optix7/     OptiX 7 backend
│   ├── devices/ospray/     OSPRay backend
│   └── serializer/         scene loaders (DiVA/VIDi JSON, optional USDA)
├── python/                 pybind11 binding source + `ovrpy` Python package
│   ├── python.cpp          PYBIND11_MODULE(_core, m)
│   └── ovrpy/              re-exports `_core` as `import ovrpy`
├── tests/                  doctest + pytest test suite (see tests/README.md)
├── pyproject.toml          scikit-build-core driver for `pip install`
├── scripts/build.sh        configure/build/test driver for the cmake flow
├── cmake/                  CMake configure modules
├── extern/                 vendored / FetchContent dependency wrappers
├── data/                   example scenes, configs, transfer functions
├── gdt/                    vendored math header library
└── projects/               sample integrations
```

## Requirements

- CMake **>= 3.18**
- A C++17 compiler — tested on GCC 11/13 (Ubuntu 22.04/24.04) and MSVC 2019/2022
- Threads (POSIX or Win32)

Optional, gated by build-time flags (defaults shown):

| Component | When needed | Notes |
| --- | --- | --- |
| CUDA Toolkit (≥ 11.x) | `OVR_BUILD_CUDA=ON` | CI tests against 11.8 and 12.8 |
| OptiX 7 SDK | `OVR_BUILD_DEVICE_OPTIX7=ON` | `OptiX_INSTALL_DIR=<path>`; download from <https://developer.nvidia.com/optix> |
| OSPRay + TBB | `OVR_BUILD_DEVICE_OSPRAY=ON` | Auto-fetched via FetchContent; override with `-Dospray_DIR=...` |
| OpenGL + GLFW | `OVR_BUILD_OPENGL=ON` | Required for `renderapp`; on Debian/Ubuntu: `sudo apt install libglfw3-dev xorg-dev libtbb-dev` |
| Python 3 + pytest | `OVR_BUILD_TESTS=ON` and/or `OVR_BUILD_PYTHON_BINDINGS=ON` | `pip install -e .[test]` |
| Pixar USD | `OVR_BUILD_USD=ON` (off by default) | For USDA scene loading |

Submodules:

```bash
git submodule update --init --recursive
```

`scripts/build.sh` checks `git submodule status` and refuses to proceed
if any submodule is uninitialised — it never runs `git submodule update`
on your behalf.

## Install

There are two equivalent paths; pick whichever matches what you're after.
Don't combine them on the same checkout — you'd build twice.

### Python tier (`ovrpy`) via pip / uv

```bash
# uv
uv sync --extra test
uv run ovrpy-render --help
uv run python -c "import ovrpy; print(ovrpy.vec2i())"

# or plain pip
pip install -e .[test]
ovrpy-render --help
```

`pip install` / `uv sync` invokes the same top-level `CMakeLists.txt`
through [scikit-build-core](https://scikit-build-core.readthedocs.io/),
producing a self-contained wheel at `<site-packages>/ovrpy/` (renderer +
OSPRay closure + interactive OpenGL tier all bundled). Headless / CPU-only
users can opt out of CUDA/OptiX:

```bash
CMAKE_ARGS="-DOVR_BUILD_CUDA=OFF -DOVR_BUILD_DEVICE_OPTIX7=OFF" uv sync --extra test
```

ovrpy bundles its own renderer + OSPRay/imgui closure but relies on a few
host-provided system libraries (`libcuda.so.1`, `libGL.so.1`,
`libOpenGL.so.0`, `libX11.so.6`, `libvulkan.so.1`). If any are missing,
`import ovrpy` raises a translated `ImportError` that names the missing
lib and prints the exact `apt`/`dnf` install command. See
[`DEV.md`](DEV.md#loader-error-translation) for the full table.

### C++ apps via `scripts/build.sh`

`--configure`, `--build`, `--test` are additive; `--all` ≡ all three.
`--python` toggles the Python bindings on the cmake side (independent of
the pip/uv flow above).

```bash
./scripts/build.sh                # configure + build (C++ only)
./scripts/build.sh --python       # configure + build, with the ovrpy bindings
./scripts/build.sh --all          # configure + build + test
./scripts/build.sh --test         # tests only, against an existing build
./scripts/build.sh --clean        # rm -rf the build dir, exit
```

The full phase/option matrix and manual `cmake ... && cmake --build ...`
recipes are in [`DEV.md`](DEV.md).

## Running

After `pip install -e .` / `uv sync`, three console scripts land on
`$PATH` — `ovrpy-render` is a Python entry point; `renderapp` and
`renderbatch` are the bundled C++ apps wrapped by tiny entry-point
shims (`python/ovrpy/_apps.py`) so they resolve their RPATH cleanly.

```bash
# Python entry point
ovrpy-render data/configs/<scene>.json -o render.png --backend ospray

# C++ apps (same binaries as the cmake-only flow, but on $PATH)
renderapp   data/configs/<scene>.json    # interactive viewer (needs GLFW + display)
renderbatch data/configs/<scene>.json    # offline render to PNG

# Same C++ apps from a cmake-only build (no pip install needed)
./build/renderapp   data/configs/<scene>.json
./build/renderbatch data/configs/<scene>.json
```

Scene JSON files live under `data/configs/`; see
`data/configs/README.md` for the schema. The Python tier exposes the
same renderer through `import ovrpy`; see `tests/python/` for end-to-end
usage examples.

## Testing

```bash
./scripts/build.sh --all     # configure + build + run the full suite
./scripts/build.sh --test    # iterate on tests against an existing build

# pytest only (after `pip install -e .[test]` or `uv sync --extra test`):
uv run pytest tests/python/ -v
```

When `OVR_BUILD_TESTS=ON`, CMake hard-requires `Python3` +
`OVR_BUILD_PYTHON_BINDINGS=ON` + an importable `pytest` and emits a
`FATAL_ERROR` with the install command if anything is missing — there is
no silent skip.

See [`tests/README.md`](tests/README.md) for labels (`cpp` / `gpu` /
`python` / `golden`), the golden-image baseline workflow, coverage, and
troubleshooting.

## License

OVR is distributed under the Apache 2.0 License (see [`LICENSE`](LICENSE)).
Bundled / fetched dependencies retain their own licenses.
