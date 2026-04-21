# OVR Test Suite

This directory contains the unified test suite for OVR: C++/CUDA unit
tests (doctest), Python binding tests (pytest), and rendering regression
tests (golden-image PSNR + SSIM). Everything is driven through **CTest**
so a single command runs the lot.

## Quick start

```bash
# 1. Configure with tests enabled
cmake -S . -B build -DOVR_BUILD_TESTS=ON -DOVR_BUILD_PYTHON_BINDINGS=ON

# 2. Build (doctest is fetched by FetchContent during this step)
cmake --build build -j

# 3. Install Python test deps (first time only)
pip install -r test/requirements.txt

# 4. Run the whole suite
ctest --test-dir build --output-on-failure

# 4b. CPU-only run (skip GPU-gated tests)
ctest --test-dir build --output-on-failure -LE gpu
```

Or, from the top-level helper script:

```bash
./scripts/build.sh --all     # configure + build + test
./scripts/build.sh --test    # test only (assumes build is up to date)
```

## What's in the box

```
test/
├── CMakeLists.txt               # wires C++ tests + pytest into CTest
├── conftest.py                  # shared pytest fixtures
├── pytest.ini                   # pytest config + marker registry
├── requirements.txt             # pip deps for python tests
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

## Controlling the `python_tests` CTest entry

CMake probes for `import pytest` at configure time and auto-skips the
`python_tests` registration when it's missing. Override with
`-DOVR_PYTHON_TESTS=<AUTO|ON|OFF>`:

| Value      | Behaviour                                                  |
| ---------- | ---------------------------------------------------------- |
| `AUTO` (default) | probe; register iff pytest is importable             |
| `ON`       | always register; ctest will surface the import error       |
| `OFF`      | never register                                             |

Use `ON` when the test deps are installed *after* configure (e.g. CI
installs `test/requirements.txt` in a later step), or when you want
ctest to fail loudly instead of silently skipping.

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

`pytest` markers for finer control:

```bash
# Skip the expensive path-tracing tests
pytest -m "not slow"

# Only the rendering regression tests
pytest -m golden
```

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

CUDA-labelled tests use a CTest fixture called `gpu_available` that is
set up by a tiny `gpu_probe` executable (generated at build time) which
exits non-zero when `cudaGetDeviceCount() <= 0`. Tests are then skipped
automatically on GPU-less hosts — no manual flag needed.

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
  Your build dir is not on `sys.path`. Either:
    * Run tests via `ctest` (CTest sets `OVR_BUILD_DIR` automatically), or
    * Export `OVR_BUILD_DIR=<path-to-build>` before invoking pytest.

* **`ImportError: libospray...so`**  
  Load the runtime env first:  `source scripts/run.sh`.

* **`no CUDA device`** during `doctest_discover_tests`  
  GPU tests are discovered via `DISCOVERY_MODE PRE_TEST` and will only
  try to enumerate at ctest-time, not at build-time. If you see this at
  build-time, your CMake may be older than 3.18 — bump it.
