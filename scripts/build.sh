#!/usr/bin/env bash
# Configure, build, and test the OVR C++/CUDA + Python bindings.
# OpenVKL and other external deps are fetched and built automatically via
# FetchContent during the configure step - no separate deps build needed.
#
# Usage:
#   ./scripts/build.sh              # configure + build (no tests)
#   ./scripts/build.sh --configure  # configure only
#   ./scripts/build.sh --build      # build only (skip configure)
#   ./scripts/build.sh --test       # test only (skip configure/build)
#   ./scripts/build.sh --all        # configure + build + test
#   ./scripts/build.sh --clean      # wipe build dir and exit
#
# Environment overrides:
#   BUILD_DIR    override build directory (default: <repo>/build)
#   CMAKE_ARGS   extra flags passed to cmake

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The script lives under base/scripts/ so the actual repo root is one up.
SOURCE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$SOURCE_DIR/build}"

# ── parse args ───────────────────────────────────────────────────────────────
DO_CONFIGURE=true
DO_BUILD=true

for arg in "$@"; do
  case "$arg" in
    --configure) DO_BUILD=false; DO_TEST=false  ;;
    --build)     DO_CONFIGURE=false; DO_TEST=false  ;;
    --test)      DO_CONFIGURE=false; DO_BUILD=false ;;
    --clean)
      echo "[clean] removing $BUILD_DIR"
      rm -rf "$BUILD_DIR"
      exit 0
      ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

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

# ── submodules ────────────────────────────────────────────────────────────────
init_submodules() {
  info "Initialising git submodules..."
  cd "$SOURCE_DIR"
  git submodule update --init --recursive
  ok "submodules ready"
}

# ── configure ─────────────────────────────────────────────────────────────────
configure() {
  info "Configuring (build dir: $BUILD_DIR)..."
  # FetchContent fetches rkcommon, embree, and openvkl automatically at configure time.
  cmake \
    -S "$SOURCE_DIR" \
    -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DOVR_BUILD_PYTHON_BINDINGS=ON \
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

# ── main ──────────────────────────────────────────────────────────────────────
check_prereqs
init_submodules

$DO_CONFIGURE && configure
$DO_BUILD     && build
