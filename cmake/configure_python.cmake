cmake_minimum_required(VERSION 3.10)
include_guard(GLOBAL)

# Reuse PYTHON_EXECUTABLE set by parent project; only detect if not yet set.
if(NOT PYTHON_EXECUTABLE)
execute_process(
  COMMAND "which" "python" OUTPUT_VARIABLE PYTHON_EXECUTABLE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
endif()
if("${PYTHON_EXECUTABLE}" STREQUAL "")
message(FATAL_ERROR "Python not found — pass -DPYTHON_EXECUTABLE=<path> or activate a venv")
else()
message(STATUS "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")
endif()

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
include(FetchContent)
FetchContent_Declare(pybind11
    GIT_REPOSITORY  https://github.com/pybind/pybind11.git
    GIT_TAG         v2.13.6
    GIT_SHALLOW     ON
)
FetchContent_MakeAvailable(pybind11)
