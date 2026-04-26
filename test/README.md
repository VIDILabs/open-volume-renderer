# OVR Test Suite

This directory contains the unified test suite for OVR: C++/CUDA unit
tests (doctest), Python binding tests (pytest), and rendering regression
tests (golden-image PSNR + SSIM). Everything is driven through **CTest**
so a single command runs the lot.

User-facing install/run instructions live in [`../README.md`](../README.md);
build-system internals (CMake options, scikit-build-core wheel layout,
component scoping, loader-error helper) are in [`../DEV.md`](../DEV.md).

## Requirements

`OVR_BUILD_TESTS=ON` enforces these at configure time — any missing piece
is a hard `FATAL_ERROR` (no silent skips):

* a Python 3 interpreter (`find_package(Python3 ... REQUIRED)`)
* `pytest` importable from that interpreter (probed via `python3 -c "import pytest"`)
* `OVR_BUILD_PYTHON_BINDINGS=ON` and the `_core` (a.k.a. `ovrpy._core`) target

Install the Python deps once before configuring. Either path works:

```bash
# (a) editable install of the ovrpy package; also pulls test extras
python3 -m pip install -e .[test]

# (b) bare deps only (skips the scikit-build-core build step; useful in
#     CI where cmake is invoked separately)
python3 -m pip install pytest pytest-xdist pytest-cov numpy pillow scikit-image
```

(Use a virtualenv if your system Python is managed; on Debian/Ubuntu
`pip install` may otherwise refuse with PEP 668.)

## Quick start

```bash
# 1. Configure with tests enabled (requires the deps above)
cmake -S . -B build -DOVR_BUILD_TESTS=ON -DOVR_BUILD_PYTHON_BINDINGS=ON

# 2. Build (doctest is fetched by FetchContent during this step)
cmake --build build -j

# 3. Run the whole suite
ctest --test-dir build --output-on-failure

# 3b. CPU-only run (skip GPU-gated tests)
ctest --test-dir build --output-on-failure -LE gpu
```

Or, from the top-level helper script. The script never installs anything
on your behalf: when `--test` / `--all` is used and `pytest` isn't
importable, it prints the exact `pip install` command and exits so you
can run it yourself (in a venv, conda env, or with whatever package
manager fits your setup).

```bash
./scripts/build.sh --all     # configure + build + test
./scripts/build.sh --test    # test only (assumes build is up to date)
```

Same applies to git submodules: the script checks `git submodule status`
and exits with the manual `git submodule update --init --recursive`
command if anything is uninitialised, rather than fetching submodules
behind your back.

## What's in the box

```
test/
├── CMakeLists.txt               # wires C++ tests + pytest into CTest
├── conftest.py                  # shared pytest fixtures
├── ../pyproject.toml            # pytest config + marker registry under [tool.pytest.ini_options]
├── ../pyproject.toml            # pip deps live under [project.optional-dependencies] test
├── generate_synthetic_volume.cmake  # generates a 32^3 volume fixture
├── fixtures/
│   ├── test_scene.json          # original DiVA scene (needs external data)
│   ├── test_s3d*.json           # additional DiVA fixtures
│   └── golden/
│       ├── optix7/*.png         # rendering baselines (optix7)
│       └── ospray/*.png         # rendering baselines (ospray)
├── cpp/                         # doctest C++/CUDA test binaries
│   ├── CMakeLists.txt
│   ├── test_math.cpp
│   ├── test_scene.cpp
│   ├── test_serializer_json.cpp
│   ├── test_transactional_value.cpp
│   ├── test_cross_device_buffer.cpp
│   ├── test_count_tfn.cpp
│   ├── test_imageio.cpp
│   ├── test_generate_mask.cpp
│   ├── test_cuda_cross_device_buffer.cu   (GPU-gated)
│   └── test_cuda_generate_mask.cu         (GPU-gated)
└── python/                      # pytest test modules
    ├── test_scene.py
    ├── test_renderer_creation.py
    ├── test_framebuffer.py
    ├── test_setters.py
    ├── test_camera.py
    ├── test_threading.py
    └── test_rendering_regression.py
```

## CTest labels

Every test carries one or more labels so you can slice the run:

| Label        | Meaning                                               |
| ------------ | ----------------------------------------------------- |
| `cpp`        | C++ doctest case (CPU-only unless also `gpu`)         |
| `gpu`        | Needs a CUDA device; skipped when `gpu_probe` fails   |
| `python`     | The `pytest` entry that runs every Python test module |
| `gpu_probe`  | The cheap probe that enables/disables the `gpu` label |

Examples:

```bash
ctest -L cpp              # only C++ unit tests
ctest -L gpu              # only GPU tests
ctest -LE gpu             # everything except GPU tests (CI default)
ctest -R python_tests     # just the Python suite
```

## Python-specific markers

The pytest suite is sliced by hardware tier and by cost. Markers are
registered in `pyproject.toml` (`[tool.pytest.ini_options]`) and
`--strict-markers` is on, so typos error
out instead of silently selecting nothing.

| Marker | Applied to | Use |
| --- | --- | --- |
| `gpu` | `[optix7]` parametrize variants (via `pytest.param("optix7", marks=...)` in `conftest.py`) | `pytest -m "not gpu"` to skip GPU-bound tests on a CPU-only host |
| `cpu` | `[ospray]` parametrize variants | `pytest -m "not cpu"` to run only the GPU tier |
| `slow` | path-tracing tests (`@pytest.mark.slow`) | `pytest -m "not slow"` for a fast inner loop |
| `golden` | rendering regression tests | `pytest -m golden` to run only the baseline checks |

```bash
pytest -m "not gpu"        # CPU-only run (matches `ctest -LE gpu`)
pytest -m "not cpu"        # GPU-only run
pytest -m "not slow"       # skip the expensive path-tracing tests
pytest -m golden           # only the rendering regression tests
```

Nothing is skipped automatically — running `pytest test/python/` with
no `-m` filter attempts every variant and any missing/broken hardware
surfaces as a real failure.

## Golden-image regression workflow

The rendering regression tests in `python/test_rendering_regression.py`
compare a freshly-rendered image against a committed baseline.

* **First time on a new backend or after an intentional render change:**

  ```bash
  pytest test/python/test_rendering_regression.py --update-baselines
  git add test/fixtures/golden/<backend>/*.png
  git commit -m "Regenerate rendering baselines"
  ```

* **CI failures** upload an `actual.png` + `diff.png` artifact per
  failing test; download from the run's Artifacts tab to debug.

* **Tolerances** live in `TOLERANCES` at the top of
  `test_rendering_regression.py` and are currently `PSNR >= 30 dB` and
  `SSIM >= 0.90`. Bump them (tighter) once baselines stabilise on your
  reference machine.

## CUDA / GPU gating

The C++/CUDA tests (`test/cpp/`) and the Python tests (`test/python/`)
gate GPU dependence differently:

* **C++ tier (CTest)**: doctest binaries labelled `gpu` declare a
  `FIXTURES_REQUIRED gpu_available` dependency. The `gpu_available`
  fixture is set up by a small `gpu_probe` executable (built when
  `OVR_BUILD_TESTS=ON` and CUDA is enabled) that exits non-zero when
  `cudaGetDeviceCount() <= 0`. CTest then auto-skips dependent tests on
  GPU-less hosts — no manual flag needed. `ctest -LE gpu` is also
  available as an explicit opt-out.
* **Python tier (pytest)**: nothing auto-skips. The `[optix7]`
  parametrize variants are tagged with the `gpu` marker (see the table
  above); the user opts out explicitly with `pytest -m "not gpu"` (or
  `ctest -LE gpu`, which is what the CI uses).

## Coverage

With `-DOVR_ENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug`, two commands
produce a combined report:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DOVR_BUILD_TESTS=ON \
      -DOVR_BUILD_PYTHON_BINDINGS=ON -DOVR_ENABLE_COVERAGE=ON
cmake --build build -j

# Run tests + generate XML/HTML reports
cmake --build build --target coverage
# Output: build/coverage/coverage.xml  (C++, Cobertura XML)
#         build/coverage/index.html    (C++, drill-down HTML)
#         build/coverage/coverage-py.xml (Python, if pytest-cov installed)
```

`gcovr` is the only external tool required; `pip install gcovr`.

## Synthetic volume fixture

Because the committed `test_scene.json` references an absolute data path
that only exists on the original author's machine, we also generate a
self-contained 32³ gaussian blob as part of the build
(`test/generate_synthetic_volume.cmake`). It ends up at
`build/test/generated_fixtures/synthetic_scene.json` and is what the
Python suite picks up by default through the `scene_path` fixture.

## Troubleshooting

* **`ModuleNotFoundError: No module named 'ovrpy'`**  
  Either the build dir is not on `sys.path` or you haven't installed the
  package. Pick one:
    * Run tests via `ctest` (CTest sets `OVR_BUILD_DIR` to the build dir,
      and the staged `<build>/ovrpy/` package lives directly under it).
    * Export `OVR_BUILD_DIR=<path-to-build>` before invoking pytest.
    * Run `pip install -e .` from the repo root so `ovrpy` lands in
      site-packages.

* **`ImportError: libospray...so`**  
  Load the runtime env first:  `source scripts/run.sh`.

* **`no CUDA device`** during `doctest_discover_tests`  
  GPU tests are discovered via `DISCOVERY_MODE PRE_TEST` and will only
  try to enumerate at ctest-time, not at build-time. If you see this at
  build-time, your CMake may be older than 3.18 — bump it.
