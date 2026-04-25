# ======================================================================== #
# OVR unit-test scaffolding                                                 #
#                                                                           #
# Fetches doctest at configure time (no extra submodule), exposes           #
# `doctest_discover_tests` (from doctest's own cmake/ dir), and provides    #
# the `ovr_add_cpp_test()` helper used by test/cpp/CMakeLists.txt.          #
#                                                                           #
# GPU-labelled tests are gated via CTest fixtures so they auto-skip on      #
# machines without an NVIDIA GPU (see the `gpu_probe` executable below).    #
# ======================================================================== #

include_guard(GLOBAL)

include(FetchContent)

set(DOCTEST_VERSION "v2.4.11")

FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG        ${DOCTEST_VERSION}
  GIT_SHALLOW    ON
)

# Keep doctest quiet: we don't want its own test targets or install rules.
set(DOCTEST_WITH_TESTS      OFF CACHE BOOL "" FORCE)
set(DOCTEST_WITH_MAIN_IN_STATIC_LIB OFF CACHE BOOL "" FORCE)
set(DOCTEST_NO_INSTALL      ON  CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(doctest)

# Expose doctest's CMake helper module (doctest_discover_tests).
list(APPEND CMAKE_MODULE_PATH "${doctest_SOURCE_DIR}/scripts/cmake")
include(doctest)

# --------------------------------------------------------------------------
# ovr_add_cpp_test(<name>
#                  SOURCES <srcs...>
#                  [LINK <libs...>]
#                  [DEFS <defs...>]
#                  [GPU]          # label the test "gpu" and require gpu_probe
#                  [LABELS <l...>])
# --------------------------------------------------------------------------
function(ovr_add_cpp_test NAME)
  set(_options GPU)
  set(_one_value)
  set(_multi SOURCES LINK DEFS LABELS)
  cmake_parse_arguments(_T "${_options}" "${_one_value}" "${_multi}" ${ARGN})

  if(NOT _T_SOURCES)
    message(FATAL_ERROR "ovr_add_cpp_test(${NAME}): SOURCES is required")
  endif()

  add_executable(${NAME} ${_T_SOURCES})
  target_link_libraries(${NAME} PRIVATE doctest::doctest ${_T_LINK})
  target_compile_definitions(${NAME} PRIVATE ${_T_DEFS})
  target_compile_features(${NAME} PRIVATE cxx_std_17)

  # Keep test binaries grouped under <build>/test/ for easy discovery.
  set_target_properties(${NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  # Inherit coverage flags if enabled
  if(OVR_ENABLE_COVERAGE)
    ovr_apply_coverage_flags(${NAME})
  endif()

  set(_labels cpp)
  if(_T_GPU)
    list(APPEND _labels gpu)
  endif()
  if(_T_LABELS)
    list(APPEND _labels ${_T_LABELS})
  endif()

  # PRE_TEST discovery mode defers running the binary until `ctest` time,
  # which matters for GPU-labelled binaries that may otherwise fail to
  # enumerate on a GPU-less build host.
  #
  # TEST_PREFIX is required so each registered CTest name starts with
  # "<binary>." (e.g. "test_serializer_json.create_json_scene: ..."). Other
  # CMake glue (test/cpp/gpu_fixture_attach.cmake.in) attaches per-binary
  # FIXTURES_REQUIRED by matching this prefix; without it the regex never
  # fires and the fixture dependency silently never attaches.
  doctest_discover_tests(${NAME}
    ADD_LABELS 1
    TEST_PREFIX "${NAME}."
    PROPERTIES LABELS "${_labels}"
    DISCOVERY_MODE PRE_TEST
  )

  # Tie GPU tests to the gpu_probe fixture (defined once below) so CTest
  # auto-skips them when no CUDA device is available.
  if(_T_GPU)
    # Read back the discovered tests and add fixture dependency.
    # doctest_discover_tests creates a CTest script that registers tests at
    # ctest-time, so we attach FIXTURES_REQUIRED by label instead.
    set_property(GLOBAL APPEND PROPERTY OVR_GPU_TEST_TARGETS ${NAME})
  endif()
endfunction()

# --------------------------------------------------------------------------
# gpu_probe: a tiny helper executable that exits 0 iff at least one CUDA
# device is visible. Wired as a CTest FIXTURES_SETUP so gpu-labelled tests
# auto-skip on hostless/driverless machines.
# --------------------------------------------------------------------------
function(ovr_setup_gpu_probe)
  if(NOT OVR_BUILD_CUDA)
    return()
  endif()

  set(_probe_src "${CMAKE_BINARY_DIR}/test/gpu_probe.cpp")
  file(WRITE "${_probe_src}" [=[
// Auto-generated: exits 0 if at least one CUDA device is present.
#include <cuda_runtime.h>
#include <cstdio>
int main() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count <= 0) {
    std::fprintf(stderr, "gpu_probe: no CUDA device (err=%d, count=%d)\n",
                 (int)err, count);
    return 1;
  }
  return 0;
}
]=])

  add_executable(gpu_probe "${_probe_src}")
  target_link_libraries(gpu_probe PRIVATE CUDA::cudart_static)
  set_target_properties(gpu_probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(NAME gpu_probe COMMAND gpu_probe)
  set_tests_properties(gpu_probe PROPERTIES
    FIXTURES_SETUP gpu_available
    LABELS "gpu_probe"
  )
endfunction()
