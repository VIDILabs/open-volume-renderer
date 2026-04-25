#!/usr/bin/env bash
# Configure, build, and test the OVR C++/CUDA renderer (and, opt-in, the
# Python bindings + test suite). OpenVKL and other external deps are
# fetched and built automatically via FetchContent during the configure
# step - no separate deps build needed.
#
# Phase flags are additive. With no flags the script does configure +
# build (the historical default); pass any combination of --configure,
# --build, --test to opt into specific phases. --all is sugar for
# "--configure --build --test"; --python toggles the Python bindings
# orthogonally.
#
# Usage examples:
#   ./scripts/build.sh                       # configure + build, C++ only
#   ./scripts/build.sh --configure           # configure only
#   ./scripts/build.sh --build               # build only (skip configure)
#   ./scripts/build.sh --test                # run tests only (assumes existing build)
#   ./scripts/build.sh --build --test        # build, then run tests
#   ./scripts/build.sh --configure --build --test   # equivalent to --all
#   ./scripts/build.sh --all                 # configure + build + test
#   ./scripts/build.sh --python              # also enable -DOVR_BUILD_PYTHON_BINDINGS=ON
#   ./scripts/build.sh --build --python      # build with bindings, no tests
#   ./scripts/build.sh --clean               # wipe build dir and exit
#
# --test (and therefore --all) imply --python because the Python tier is
# a hard prerequisite of the test suite.
#
# Environment overrides:
#   BUILD_DIR    override build directory (default: <repo>/build)
#   CMAKE_ARGS   extra flags passed to cmake
#   CTEST_ARGS   extra flags passed to ctest (default: -LE gpu on hosts w/o CUDA)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The script lives under base/scripts/ so the actual repo root is one up.
SOURCE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$SOURCE_DIR/build}"

# ── parse args ───────────────────────────────────────────────────────────────
# Phase flags start off; we apply the historical default (configure+build)
# only if the user didn't pass any phase flag at all.
DO_CONFIGURE=false
DO_BUILD=false
DO_TEST=false
DO_PYTHON=false
ANY_PHASE_FLAG=false

for arg in "$@"; do
  case "$arg" in
    --configure) DO_CONFIGURE=true; ANY_PHASE_FLAG=true ;;
    --build)     DO_BUILD=true;     ANY_PHASE_FLAG=true ;;
    --test)      DO_TEST=true;      ANY_PHASE_FLAG=true; DO_PYTHON=true ;;
    --all)       DO_CONFIGURE=true; DO_BUILD=true; DO_TEST=true; ANY_PHASE_FLAG=true; DO_PYTHON=true ;;
    --python)    DO_PYTHON=true ;;
    --clean)
      echo "[clean] removing $BUILD_DIR"
      rm -rf "$BUILD_DIR"
      exit 0
      ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# No explicit phase requested → fall back to the historical default
# ("configure + build"). --python alone (no phase flag) therefore still
# behaves as "configure + build with bindings on".
if ! $ANY_PHASE_FLAG; then
  DO_CONFIGURE=true
  DO_BUILD=true
fi

# Safety net in case a future arg sets DO_TEST=true without remembering
# to also flip DO_PYTHON: the test tier hard-requires the bindings.
if $DO_TEST; then
  DO_PYTHON=true
fi

# ── helpers ──────────────────────────────────────────────────────────────────
ok()   { echo -e "\033[32m[ok]\033[0m $*"; }
info() { echo -e "\033[34m[info]\033[0m $*"; }
warn() { echo -e "\033[33m[warn]\033[0m $*"; }
die()  { echo -e "\033[31m[error]\033[0m $*" >&2; exit 1; }

# ── find nvcc ─────────────────────────────────────────────────────────────────
find_nvcc() {
  if   command -v nvcc &>/dev/null;       then echo "nvcc"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then echo "/usr/local/cuda/bin/nvcc"
  else die "nvcc not found in PATH or /usr/local/cuda/bin — CUDA toolkit required"
  fi
}

# ── find python ───────────────────────────────────────────────────────────────
find_python() {
  if   command -v python3 &>/dev/null; then echo "python3"
  elif command -v python  &>/dev/null; then echo "python"
  else die "python3 not found in PATH — required when --test/--all is used"
  fi
}

# Verify pytest is importable from $1 (the python interpreter). When
# missing, the script prints the exact install command for the user to
# run themselves and exits. We never invoke pip - mutating the user's
# Python environment (system, conda, venv, PEP 668 etc.) is something
# only the user can decide to do safely.
require_pytest() {
  local PY="$1"
  if "$PY" -c "import pytest" &>/dev/null; then
    info "pytest: importable via $PY"
    return 0
  fi

  warn "pytest is not importable via $PY (required by --test/--all)."
  echo  "       Install the test dependencies manually, then re-run this script:"
  echo  "           $PY -m pip install -e \"$SOURCE_DIR\"[test]"
  echo  "       (use a virtualenv if your system Python is externally managed)"
  die   "pytest unavailable; install manually and retry."
}

# ── prerequisites ─────────────────────────────────────────────────────────────
check_prereqs() {
  info "Checking prerequisites..."
  command -v cmake &>/dev/null || die "cmake not found (need >= 3.18)"

  NVCC_BIN=$(find_nvcc)
  info "nvcc: $NVCC_BIN"

  # If nvcc is not in PATH, add its directory so cmake's FindCUDA can see it
  if [[ "$NVCC_BIN" != "nvcc" ]]; then
    export PATH="$(dirname "$NVCC_BIN"):$PATH"
    info "Added $(dirname "$NVCC_BIN") to PATH"
  fi

  CMAKE_VER=$(cmake --version | head -1 | awk '{print $3}')
  info "cmake $CMAKE_VER"
}

# Test-only prereqs: Python interpreter + pytest. Only invoked when
# DO_TEST=true so plain configure/build flows (e.g. CI image without
# pytest, or a developer who only wants the renderer libraries) don't
# fail on missing test deps. The matching CMake hard-error in
# test/CMakeLists.txt only fires when -DOVR_BUILD_TESTS=ON, which we
# also gate on DO_TEST below.
check_test_prereqs() {
  info "Checking test prerequisites (Python + pytest)..."
  PY_BIN=$(find_python)
  PY_VER=$("$PY_BIN" -c "import sys; print(sys.version.split()[0])")
  info "python: $PY_BIN ($PY_VER)"
  require_pytest "$PY_BIN"
}

# ── submodules ────────────────────────────────────────────────────────────────
# We do NOT run `git submodule update --init --recursive` automatically:
# that performs network I/O and writes to .git/modules/ + extern/<sub>
# without the user opting in. Instead we check the status, skip silently
# when everything is already initialised, and exit with the manual
# command when something is missing.
check_submodules() {
  cd "$SOURCE_DIR"
  if [[ ! -f .gitmodules ]]; then
    return 0   # repo has no submodules; nothing to check
  fi

  # `git submodule status` prefixes uninitialised entries with '-'.
  # Capture so we can also list them on failure.
  local status_out
  status_out=$(git submodule status 2>/dev/null || true)
  local missing
  missing=$(echo "$status_out" | awk '/^-/ {print $2}')

  if [[ -z "$missing" ]]; then
    info "git submodules: initialised"
    return 0
  fi

  warn "Some git submodules are not initialised:"
  while IFS= read -r path; do
    [[ -n "$path" ]] && echo "         $path"
  done <<< "$missing"
  echo  "       Initialise them manually, then re-run this script:"
  echo  "           git submodule update --init --recursive"
  die   "submodules missing; initialise manually and retry."
}

# ── configure ─────────────────────────────────────────────────────────────────
configure() {
  info "Configuring (build dir: $BUILD_DIR)..."
  # FetchContent fetches rkcommon, embree, and openvkl automatically at configure time.
  #
  # Two cmake options are derived from the script's flags:
  #
  #   * OVR_BUILD_PYTHON_BINDINGS — opt-in via --python (or implied by
  #     --test/--all). Off by default so a plain `./scripts/build.sh` is
  #     a C++-only build that doesn't need a Python interpreter at all.
  #
  #   * OVR_BUILD_TESTS — opt-in via --test/--all. Otherwise off, so the
  #     cmake-side hard-require for pytest in test/CMakeLists.txt only
  #     fires when the user actually wants tests. Users who want tests
  #     configured but not run this invocation can always pass
  #     `CMAKE_ARGS="-DOVR_BUILD_TESTS=ON"` explicitly - in that case
  #     cmake itself will surface the missing-pytest error.
  local _python_flag="OFF"
  if $DO_PYTHON; then
    _python_flag="ON"
  fi
  local _tests_flag="OFF"
  if $DO_TEST; then
    _tests_flag="ON"
  fi
  cmake \
    -S "$SOURCE_DIR" \
    -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DOVR_BUILD_PYTHON_BINDINGS="$_python_flag" \
    -DOVR_BUILD_TESTS="$_tests_flag" \
    ${CMAKE_ARGS:-}
  ok "configure done"
}

# ── build ─────────────────────────────────────────────────────────────────────
build() {
  JOBS=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
  info "Building with $JOBS parallel jobs..."
  cmake --build "$BUILD_DIR" --parallel "$JOBS"
  ok "build done"
}

# ── test ──────────────────────────────────────────────────────────────────────
run_tests() {
  info "Running ctest..."
  # Default: exclude GPU-labelled tests so this works on machines without a GPU.
  # Users can set CTEST_ARGS="" to run everything, or pass their own filters.
  local default_args="-LE gpu --output-on-failure --no-tests=error"
  ctest --test-dir "$BUILD_DIR" ${CTEST_ARGS:-$default_args}
  ok "tests done"
}

# ── main ──────────────────────────────────────────────────────────────────────
check_prereqs
$DO_TEST && check_test_prereqs
check_submodules

$DO_CONFIGURE && configure
$DO_BUILD     && build
$DO_TEST      && run_tests
