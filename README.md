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
- Optional Python bindings (`ovrpy`) built with pybind11
- A unified test suite (CTest + doctest + pytest) with optional
  PSNR/SSIM rendering regression

## Repository layout

```
.
├── apps/                   renderapp + renderbatch executables
├── ovr/                    core library (rendercommon, renderlib, devices/)
│   ├── devices/optix7/     OptiX 7 backend
│   ├── devices/ospray/     OSPRay backend
│   └── serializer/         scene loaders (DiVA/VIDi JSON, optional USDA)
├── python/                 pybind11 bindings → ovrpy module
├── test/                   doctest + pytest test suite (see test/README.md)
├── gdt/                    vendored math header library
├── projects/               sample integrations
├── scripts/build.sh        convenience configure/build/test driver
├── cmake/                  CMake configure modules
├── extern/                 vendored / FetchContent dependency wrappers
├── data/                   example scenes, configs, transfer functions
└── github-actions/         CI helper submodule
```

## Requirements

System (always required):

- CMake **>= 3.18**
- A C++17 compiler — tested on GCC 11/13 (Ubuntu 22.04/24.04) and MSVC 2019/2022
- Threads (POSIX or Win32)

Optional, gated by build-time flags:

| Component | When needed | Notes |
| --- | --- | --- |
| CUDA Toolkit (≥ 11.x) | `OVR_BUILD_CUDA=ON` (default) | CI tests against 11.8 and 12.8 |
| OptiX 7 SDK | `OVR_BUILD_DEVICE_OPTIX7=ON` (default) | Set `OptiX_INSTALL_DIR=<path>`; download from <https://developer.nvidia.com/optix> |
| OSPRay + TBB | `OVR_BUILD_DEVICE_OSPRAY=ON` (default) | Auto-fetched via FetchContent by default; supply your own with `-Dospray_DIR=...` |
| OpenGL + GLFW | `OVR_BUILD_OPENGL=ON` (default) | Required for `renderapp`; on Debian/Ubuntu: `sudo apt install libglfw3-dev xorg-dev libtbb-dev` |
| Python 3 + pytest | `OVR_BUILD_TESTS=ON` and/or `OVR_BUILD_PYTHON_BINDINGS=ON` | `pip install -r test/requirements.txt` |
| Pixar USD | `OVR_BUILD_USD=ON` (off by default) | For USDA scene loading |

Submodules are required:

```bash
git submodule update --init --recursive
```

`scripts/build.sh` checks `git submodule status` before doing anything
and refuses to proceed if any submodule is uninitialised — it never
runs `git submodule update` on your behalf.

## Quick start (recommended): `scripts/build.sh`

The phase flags (`--configure`, `--build`, `--test`) are **additive** —
each one toggles its phase, and you can combine them freely. With no
flags the script does configure + build (the historical default).
`--all` is sugar for `--configure --build --test`. `--python` is
orthogonal and enables `OVR_BUILD_PYTHON_BINDINGS=ON`.

```bash
./scripts/build.sh                            # configure + build (C++ only)
./scripts/build.sh --python                   # configure + build, with the ovrpy bindings
./scripts/build.sh --configure                # configure only
./scripts/build.sh --build                    # build only (skip configure)
./scripts/build.sh --test                     # run tests only (assumes an existing build)
./scripts/build.sh --build --test             # build, then run tests
./scripts/build.sh --configure --build --test # ≡ --all
./scripts/build.sh --all                      # configure + build + run tests
./scripts/build.sh --clean                    # rm -rf the build dir, exit
```

`--test` (and therefore `--all`) implies `--python`, because the test
tier is gated on the bindings.

Phase / option matrix:

| Invocation | Configure | Build | Test | `OVR_BUILD_PYTHON_BINDINGS` | `OVR_BUILD_TESTS` |
| --- | :-: | :-: | :-: | :-: | :-: |
| (none) | yes | yes |  | OFF | OFF |
| `--configure` | yes |  |  | OFF | OFF |
| `--build` |  | yes |  | (uses cache) | (uses cache) |
| `--test` |  |  | yes | **ON** | **ON** |
| `--build --test` |  | yes | yes | **ON** | **ON** |
| `--configure --test` | yes |  | yes | **ON** | **ON** |
| `--configure --build` | yes | yes |  | OFF | OFF |
| `--configure --build --test` | yes | yes | yes | **ON** | **ON** |
| `--all` | yes | yes | yes | **ON** | **ON** |
| `--python` | yes | yes |  | **ON** | OFF |
| `--build --python` |  | yes |  | **ON** (cache) | (cache) |
| `--clean` | (deletes the build dir, then exits) | | | | |

`--python` can be added to any of the above to flip
`OVR_BUILD_PYTHON_BINDINGS=ON` at configure time. Note that the cmake
option only takes effect when configure runs; if you only do `--build`
the script reuses whatever was in the cmake cache from the previous
configure.

The script never mutates your environment for you:

- If `--test`/`--all` is requested but `pytest` isn't importable, it
  prints the exact `pip install -r test/requirements.txt` command and
  exits non-zero. You install, then re-run the script.
- If git submodules are uninitialised, it prints
  `git submodule update --init --recursive` and exits.
- It does **not** install system packages, run pip, or fetch submodules
  silently.

Environment overrides honoured by the script:

| Variable | Effect |
| --- | --- |
| `BUILD_DIR` | Override the build directory (default: `<repo>/build`) |
| `CMAKE_ARGS` | Extra flags appended to the cmake invocation |
| `CTEST_ARGS` | Extra flags passed to ctest (default: `-LE gpu --output-on-failure --no-tests=error`) |

## Manual cmake invocation

If you'd rather drive cmake yourself, the equivalent of `--all` is:

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

Top-level CMake options:

| Option | Default | Effect |
| --- | --- | --- |
| `OVR_BUILD_DEVICE_OPTIX7` | ON | Build the OptiX 7 backend (requires CUDA + OptiX SDK) |
| `OVR_BUILD_DEVICE_OSPRAY` | ON | Build the OSPRay backend |
| `OVR_BUILD_CUDA` | ON | Compile CUDA sources / link cudart |
| `OVR_BUILD_OPENGL` | ON | Build the GLFW + ImGui-based `renderapp` |
| `OVR_BUILD_APPS` | ON | Build the application executables (`renderapp`, `renderbatch`) |
| `OVR_BUILD_PYTHON_BINDINGS` | OFF | Build the `ovrpy` pybind11 module |
| `OVR_BUILD_USD` | OFF | Enable USDA scene loading via Pixar USD |
| `OVR_BUILD_TESTS` | OFF | Build C++/CUDA unit tests + register Python tests |
| `OVR_ENABLE_COVERAGE` | OFF | `--coverage -O0 -g` for GCC/Clang; adds the `coverage` custom target |

External libraries you may want to point at explicitly:

```bash
cmake -S . -B build \
  -DOptiX_INSTALL_DIR=<path-to-optix7-sdk> \
  -Dospray_DIR=<path-to-ospray>/lib/cmake/ospray-x.x.0 \
  -DTBB_DIR=<path-to-tbb>/lib/cmake/tbb \
  -DCMAKE_PREFIX_PATH=<path-to-libtorch>     # optional
```

## Running

After a build, executables live in `build/`:

```bash
# Interactive viewer (requires GLFW + a display)
./build/renderapp data/configs/<scene>.json

# Offline render
./build/renderbatch data/configs/<scene>.json
```

Scene JSON files live under `data/configs/`; see
`data/configs/README.md` for the schema. The Python tier exposes the
same renderer through `import ovrpy`; see `test/python/` for end-to-end
usage examples.

## Testing

OVR ships a unified test suite driven by CTest:

- C++/CUDA unit tests (doctest binaries under `test/cpp/`)
- Python binding tests (pytest under `test/python/`)
- Rendering regression tests with PSNR + SSIM against committed PNG
  baselines

Quickest path:

```bash
./scripts/build.sh --all     # configure + build + test
./scripts/build.sh --test    # iterate on tests against an existing build
```

Or, manually:

```bash
cmake -S . -B build -DOVR_BUILD_TESTS=ON -DOVR_BUILD_PYTHON_BINDINGS=ON
cmake --build build -j
pip install -r test/requirements.txt        # one-time, in a venv if your system Python is managed
ctest --test-dir build --output-on-failure -LE gpu      # CPU-only, like CI
ctest --test-dir build --output-on-failure              # everything (needs a CUDA-visible GPU)
```

When `OVR_BUILD_TESTS=ON`, CMake hard-requires `Python3` +
`OVR_BUILD_PYTHON_BINDINGS=ON` + an importable `pytest` and emits a
`FATAL_ERROR` with the install command if anything is missing — there
is no silent skip.

See [`test/README.md`](test/README.md) for labels (`cpp` / `gpu` /
`python` / `golden`), the golden-image baseline workflow, coverage
reporting, and troubleshooting.

## Continuous integration

`.github/workflows/main.yml` runs three jobs per push / PR:

| Job | Runner | What it covers |
| --- | --- | --- |
| `build-linux` | ubuntu-22.04 + ubuntu-24.04 (matrix) | Release build with CUDA 11.8 / 12.8 × OSPRay fetchcontent / external; runs C++ + Python tests, **excluding `gpu`-labelled tests** because GHA runners have no NVIDIA GPU |
| `build-windows` | windows-latest | Release build with MSVC + CUDA 12.8.1; runs C++ tests excluding `gpu` |
| `coverage-linux` | ubuntu-22.04 | Debug build with `OVR_ENABLE_COVERAGE=ON`; produces `coverage.xml` (Cobertura) and an HTML drill-down via `gcovr`, uploaded as an artifact |

Because no CI runner has a GPU, the OptiX 7 backend's runtime tests run
locally only. CI verifies the C++/binding surface and the OSPRay backend
end-to-end.

## Installing and embedding

OVR uses a generic CMake variable, `OVR_INSTALL_INCLUDEDIR`, to describe
the include root that gets encoded into installed/exported OVR interface
targets.

- When OVR is configured standalone, the top-level `CMakeLists.txt`
  includes `GNUInstallDirs` and defaults `OVR_INSTALL_INCLUDEDIR` to
  `${CMAKE_INSTALL_INCLUDEDIR}` (normally `include`).
- When OVR is embedded as a subdirectory of a parent project, the parent
  may override `OVR_INSTALL_INCLUDEDIR` before `add_subdirectory(...)`
  if OVR's public headers should live under a package-specific subtree
  such as `include/<package>`.
- This variable only controls the installed/exported include interface.
  OVR's build interface intentionally stays rooted in the OVR source
  tree so the repository does not depend on parent-specific path
  conventions.

Example parent-project override:

```cmake
include(GNUInstallDirs)
set(OVR_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/instantvnr")
add_subdirectory(open-volume-renderer)
```

Use plain `${CMAKE_INSTALL_INCLUDEDIR}` when you want OVR headers
installed directly under the global include root, and override
`OVR_INSTALL_INCLUDEDIR` only when the parent package intentionally
nests them under its own prefix.

## Roadmap / open items

- Commit golden-image regression baselines for the OSPRay and OptiX
  backends (see `test/README.md` and the `--update-baselines` workflow).
- Per-setter render-side assertions in `test/python/test_setters.py`
  (currently binding-level smoke only).
- A self-hosted GPU CI matrix entry to exercise the OptiX 7 + CUDA
  tests that are skipped on GHA runners.
- USD/USDA loader is functional but off by default; tests cover only
  the JSON path today.

## License

OVR is distributed under the Apache 2.0 License (see [`LICENSE`](LICENSE)).
Bundled / fetched dependencies retain their own licenses.
