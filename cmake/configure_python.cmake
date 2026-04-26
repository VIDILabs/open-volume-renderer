cmake_minimum_required(VERSION 3.18)
include_guard(GLOBAL)

# ---------------------------------------------------------------------------
# Locate Python (interpreter + Development.Module headers/libs for pybind11).
#
# We delegate to CMake's `FindPython3` rather than shelling out to `which
# python`. The legacy shim returned a Git-Bash-style POSIX path on Windows
# GHA runners (e.g. `/c/hostedtoolcache/.../python` with no `.exe`), which
# pybind11's bundled `FindPythonLibsNew.cmake` then failed to invoke
# (`Python config failure`). FindPython3 uses each platform's native
# discovery (py launcher / registry / venv / PATH) and returns a path the
# rest of the build can actually exec.
#
# Resolution order:
#   1. -DPYTHON_EXECUTABLE=<path> (or a parent project setting it before
#      add_subdirectory) is honoured by seeding Python3_EXECUTABLE.
#   2. Otherwise FindPython3 picks the active interpreter (venv, etc.).
#
# `Development.Module` (CMake 3.18+) is the minimal dev component needed
# to compile a Python extension; we don't link against libpython itself.
# ---------------------------------------------------------------------------
if(PYTHON_EXECUTABLE AND NOT Python3_EXECUTABLE)
  set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH
      "Python interpreter used by OVR (seeded from PYTHON_EXECUTABLE)" FORCE)
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Mirror back into PYTHON_EXECUTABLE so the legacy variable name still
# works for downstream consumers (incl. the torch ABI probe below).
set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH
    "Python interpreter used by OVR" FORCE)
message(STATUS "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")

# _GLIBCXX_USE_CXX11_ABI policy
# -------------------------------------------------------------------------
# libstdc++ has two incompatible ABIs for std::string / std::list (the
# "dual ABI" introduced in GCC 5). We only need to pin a specific value
# when `ovrpy` will be co-loaded with a prebuilt `libtorch.so` that uses
# a different ABI than our compiler's default.
#
# Resolution order (first match wins):
#   1. Explicit   -DOVR_PYTHON_CXX11_ABI=<0|1>
#   2. torch probe (only if importable)
#   3. Compiler default (detected & reported; no macro added)
# -------------------------------------------------------------------------
include(CheckCXXSourceCompiles)

function(_ovr_detect_compiler_cxx11_abi OUT_VAR)
  # Probe whether libstdc++ defines _GLIBCXX_USE_CXX11_ABI at all, then
  # whether its default value is 1 or 0. Returns "" when the macro
  # isn't defined (e.g. libc++ on macOS).
  check_cxx_source_compiles("
#include <string>
#ifndef _GLIBCXX_USE_CXX11_ABI
# error not libstdc++ dual ABI
#endif
int main(){}
" _OVR_HAS_GLIBCXX_DUAL_ABI)
  if(NOT _OVR_HAS_GLIBCXX_DUAL_ABI)
    set(${OUT_VAR} "" PARENT_SCOPE)
    return()
  endif()
  check_cxx_source_compiles("
#include <string>
#if _GLIBCXX_USE_CXX11_ABI != 1
# error not the new ABI
#endif
int main(){}
" _OVR_GLIBCXX_DEFAULT_IS_1)
  if(_OVR_GLIBCXX_DEFAULT_IS_1)
    set(${OUT_VAR} 1 PARENT_SCOPE)
  else()
    set(${OUT_VAR} 0 PARENT_SCOPE)
  endif()
endfunction()

if(DEFINED OVR_PYTHON_CXX11_ABI)
  message(STATUS "Pinning _GLIBCXX_USE_CXX11_ABI=${OVR_PYTHON_CXX11_ABI} (explicit)")
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${OVR_PYTHON_CXX11_ABI})
else()
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
      "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
    OUTPUT_VARIABLE PYTHON_GLIBCXX_USE_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(PYTHON_GLIBCXX_USE_CXX11_ABI STREQUAL "True")
    message(STATUS "Pinning _GLIBCXX_USE_CXX11_ABI=1 (matching detected PyTorch)")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
  elseif(PYTHON_GLIBCXX_USE_CXX11_ABI STREQUAL "False")
    message(STATUS "Pinning _GLIBCXX_USE_CXX11_ABI=0 (matching detected PyTorch)")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
  else()
    _ovr_detect_compiler_cxx11_abi(_abi)
    if(_abi STREQUAL "")
      message(STATUS
        "_GLIBCXX_USE_CXX11_ABI not exposed by this stdlib (likely libc++); "
        "leaving untouched. Override with -DOVR_PYTHON_CXX11_ABI=<0|1> if needed.")
    else()
      message(STATUS
        "Compiler default _GLIBCXX_USE_CXX11_ABI=${_abi} (no torch detected; "
        "leaving at compiler default). Override with -DOVR_PYTHON_CXX11_ABI=<0|1> "
        "if you will co-load ovrpy with a PyTorch built using a different ABI.")
    endif()
  endif()
endif()

# ------------------------------------------------------------------
# import pybind11
# ------------------------------------------------------------------
# Tell pybind11 to use CMake's modern FindPython (which we just ran via
# FindPython3) instead of its bundled `FindPythonLibsNew.cmake`. Without
# this, pybind11 re-discovers Python through the legacy path and on
# Windows trips over `PYTHON_EXECUTABLE` paths that don't have `.exe`.
set(PYBIND11_FINDPYTHON ON CACHE BOOL
    "pybind11: use CMake's FindPython instead of FindPythonLibsNew" FORCE)

include(FetchContent)
FetchContent_Declare(pybind11
    GIT_REPOSITORY  https://github.com/pybind/pybind11.git
    GIT_TAG         v2.13.6
    GIT_SHALLOW     ON
)
FetchContent_MakeAvailable(pybind11)
