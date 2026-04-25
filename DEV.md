# OVR developer guide

Build-system internals, packaging guts, and contributor-only notes for
[Open Volume Renderer](README.md). End-user install instructions live in
the README.

## Top-level CMake options

| Option | Default | Effect |
| --- | --- | --- |
| `OVR_BUILD_DEVICE_OPTIX7` | ON | OptiX 7 backend (requires CUDA + OptiX SDK) |
| `OVR_BUILD_DEVICE_OSPRAY` | ON | OSPRay backend |
| `OVR_BUILD_CUDA` | ON | Compile CUDA sources / link cudart |
| `OVR_BUILD_OPENGL` | ON | GLFW + ImGui-based `renderapp`; also gates the imgui hooks compiled into the device backends |
| `OVR_BUILD_APPS` | ON | Application executables (`renderapp`, `renderbatch`) |
| `OVR_BUILD_PYTHON_BINDINGS` | OFF | The `ovrpy` pybind11 module |
| `OVR_BUILD_USD` | OFF | USDA scene loading via Pixar USD |
| `OVR_BUILD_TESTS` | OFF | C++/CUDA unit tests + register Python tests |
| `OVR_ENABLE_COVERAGE` | OFF | `--coverage -O0 -g` for GCC/Clang; adds the `coverage` target |

## Manual cmake invocation

The full equivalent of `./scripts/build.sh --all`:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOVR_BUILD_DEVICE_OPTIX7=ON \
  -DOVR_BUILD_DEVICE_OSPRAY=ON \
  -DOVR_BUILD_PYTHON_BINDINGS=ON \
  -DOVR_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Pointing at out-of-tree dependencies:

```bash
cmake -S . -B build \
  -DOptiX_INSTALL_DIR=<path-to-optix7-sdk> \
  -Dospray_DIR=<path-to-ospray>/lib/cmake/ospray-x.x.0 \
  -DTBB_DIR=<path-to-tbb>/lib/cmake/tbb \
  -DCMAKE_PREFIX_PATH=<path-to-libtorch>     # optional
```

## `scripts/build.sh` phase matrix

`--configure`, `--build`, `--test` are additive; with no flags the script
does configure + build. `--all` ≡ `--configure --build --test`. `--python`
is orthogonal and toggles `OVR_BUILD_PYTHON_BINDINGS=ON`.

| Invocation | Configure | Build | Test | `OVR_BUILD_PYTHON_BINDINGS` | `OVR_BUILD_TESTS` |
| --- | :-: | :-: | :-: | :-: | :-: |
| (none) | yes | yes |  | OFF | OFF |
| `--configure` | yes |  |  | OFF | OFF |
| `--build` |  | yes |  | (cache) | (cache) |
| `--test` |  |  | yes | **ON** | **ON** |
| `--build --test` |  | yes | yes | **ON** | **ON** |
| `--configure --test` | yes |  | yes | **ON** | **ON** |
| `--configure --build` | yes | yes |  | OFF | OFF |
| `--configure --build --test` | yes | yes | yes | **ON** | **ON** |
| `--all` | yes | yes | yes | **ON** | **ON** |
| `--python` | yes | yes |  | **ON** | OFF |
| `--build --python` |  | yes |  | **ON** (cache) | (cache) |
| `--clean` | (deletes the build dir, then exits) | | | | |

`--test` (and therefore `--all`) implies `--python` because the test tier
is gated on the bindings. `--python` only takes effect when configure
runs; if you only `--build`, the script reuses whatever was in the cmake
cache from the previous configure.

The script never mutates your environment. If `pytest` isn't importable
when `--test` is requested it prints `pip install -e .[test]` and exits.
If git submodules are missing it prints
`git submodule update --init --recursive` and exits. It does **not** run
pip, install system packages, or fetch submodules silently.

Environment overrides:

| Variable | Effect |
| --- | --- |
| `BUILD_DIR` | Override the build directory (default: `<repo>/build`) |
| `CMAKE_ARGS` | Extra flags appended to the cmake invocation |
| `CTEST_ARGS` | Extra flags passed to ctest (default: `-LE gpu --output-on-failure --no-tests=error`) |

## Python wheel build (scikit-build-core)

`pyproject.toml` drives
[scikit-build-core](https://scikit-build-core.readthedocs.io/), so
`pip install` / `uv sync` invokes the same top-level `CMakeLists.txt`
the cmake-only flow uses, with these scoped defaults:

```toml
[tool.scikit-build.cmake.define]
OVR_BUILD_PYTHON_BINDINGS = "ON"
OVR_BUILD_APPS            = "OFF"
OVR_BUILD_TESTS           = "OFF"
```

`OVR_BUILD_OPENGL` keeps its global default (`ON`) so the interactive
viewer remains reachable from Python; the imgui/glad shared libs end up
in the wheel alongside the renderer + OSPRay closure.

### Wheel layout

```
ovrpy/
├── __init__.py                 re-exports the native module + loader-error helper
├── _core*.so                   pybind11 extension (PYBIND11_MODULE(_core, m))
├── render.py                   `ovrpy-render` console entry point
├── librenderlib.so             renderer + statically-absorbed device backends
├── librendercommon.so          common runtime
├── libimgui.so, libglad.so     interactive (OpenGL) tier
└── lib*.so* (OSPRay closure)   libospray, libtbb, libembree4, libopenvkl,
                                libispcrt, libOpenImageDenoise, plus
                                OpenVKL's 4/8/16-wide CPU-device modules
```

### Component scoping

A custom CMake install component, `ovrwheel`, scopes which `install()`
rules ship in the wheel — only the OVR-owned libs above. Third-party
FetchContent install rules (glfw3, glad's own rules, OSPRay's cmake
configs) would otherwise leak their files into the wheel; we filter them
out via:

```toml
[tool.scikit-build]
install.components = ["ovrwheel"]
```

The matching `python/CMakeLists.txt` rules attach `COMPONENT ovrwheel` to
every `install(TARGETS ...)` and `install(DIRECTORY ...)` call, plus set
`INSTALL_RPATH=$ORIGIN` on `_core`, `renderlib`, `rendercommon`, `imgui`,
and `glad`. OSPRay's own libs already ship with
`RUNPATH=$ORIGIN:$ORIGIN/../lib` so they self-resolve once unpacked.

The OSPRay closure is sourced from `${ospray_DIR}/../..` (i.e. the lib
dir under whatever prefix `ospray_DIR` points at — set either by the
local FetchContent of `ospray_binary` or by an external
`-Dospray_DIR=...` override).

### Loader-error translation

`python/ovrpy/__init__.py` wraps `from . import _core` and translates the
dynamic linker's `lib<X>.so: cannot open shared object file` message
into a friendlier `ImportError` that names the missing lib *and* prints
the matching `apt`/`dnf` install command. Recognised system libs:

| Lib | Used by | Ubuntu/Debian | RHEL/Fedora |
| --- | --- | --- | --- |
| `libcuda.so.1` | OptiX backend | (NVIDIA driver) | (NVIDIA driver) |
| `libGL.so.1` | OpenGL tier | `libgl1` | `mesa-libGL` |
| `libOpenGL.so.0` | OpenGL ABI (GLVND) | `libopengl0` | `libglvnd-opengl` |
| `libX11.so.6` | imgui (interactive viewer) | `libx11-6` | `libX11` |
| `libvulkan.so.1` | imgui (Vulkan loader) | `libvulkan1` | `vulkan-loader` |

To extend, edit `_SYSTEM_LIB_HINTS` in `python/ovrpy/__init__.py`.

### Headless / CPU-only opt-outs

```bash
pip install . -C cmake.define.OVR_BUILD_CUDA=OFF \
              -C cmake.define.OVR_BUILD_DEVICE_OPTIX7=OFF
# uv equivalent (pass through to scikit-build-core):
CMAKE_ARGS="-DOVR_BUILD_CUDA=OFF -DOVR_BUILD_DEVICE_OPTIX7=OFF" uv sync --extra test
```

### Starting clean

When something goes sideways (stale editable install, mismatched RPATHs,
half-built `_core.so`):

```bash
rm -rf .venv build build/skbuild
uv sync --extra test
```

`build/` holds the cmake-only artifacts (`scripts/build.sh`),
`build/skbuild/` holds the scikit-build-core wheel-build tree; both are
safe to delete.

### Open follow-ups

- Manylinux-tag adjustment (`auditwheel repair`) for cross-distro wheels.

## Embedding OVR in a parent CMake project

OVR uses `OVR_INSTALL_INCLUDEDIR` to describe the include root that gets
encoded into installed/exported OVR interface targets:

- Standalone OVR: the top-level `CMakeLists.txt` includes
  `GNUInstallDirs` and defaults `OVR_INSTALL_INCLUDEDIR` to
  `${CMAKE_INSTALL_INCLUDEDIR}` (normally `include`).
- Embedded as a subdirectory: the parent may override
  `OVR_INSTALL_INCLUDEDIR` before `add_subdirectory(...)` if OVR's
  public headers should live under a package-specific subtree such as
  `include/<package>`.
- This variable controls only the *installed/exported* include
  interface. OVR's build interface stays rooted in the OVR source tree
  so the repository doesn't depend on parent-specific path conventions.

Example:

```cmake
include(GNUInstallDirs)
set(OVR_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/instantvnr")
add_subdirectory(open-volume-renderer)
```

## Continuous integration

`.github/workflows/main.yml` runs three jobs per push / PR:

| Job | Runner | What it covers |
| --- | --- | --- |
| `build-linux` | ubuntu-22.04 + ubuntu-24.04 (matrix) | Release build with CUDA 11.8 / 12.8 × OSPRay fetchcontent / external; runs C++ + Python tests, **excluding `gpu`-labelled tests** because GHA runners have no NVIDIA GPU |
| `build-windows` | windows-latest | Release build with MSVC + CUDA 12.8.1; runs C++ tests excluding `gpu` |
| `coverage-linux` | ubuntu-22.04 | Debug build with `OVR_ENABLE_COVERAGE=ON`; produces `coverage.xml` (Cobertura) and an HTML drill-down via `gcovr`, uploaded as an artifact |

No runner has a GPU, so the OptiX 7 backend's runtime tests are local-only.
CI exercises the C++/binding surface and the OSPRay backend end-to-end.

## Roadmap / open items

- Commit golden-image regression baselines for the OSPRay and OptiX
  backends (see `test/README.md` and the `--update-baselines` workflow).
- Per-setter render-side assertions in `test/python/test_setters.py`
  (currently binding-level smoke only).
- A self-hosted GPU CI matrix entry to exercise the OptiX 7 + CUDA tests
  that are skipped on GHA runners.
- USD/USDA loader is functional but off by default; tests cover only the
  JSON path today.
- Manylinux wheel (`auditwheel repair`) for cross-distro distribution.
