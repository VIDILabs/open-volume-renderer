# ======================================================================== #
# OVR unit-test scaffolding                                                 #
#                                                                           #
# Fetches doctest at configure time (no extra submodule), exposes           #
# `doctest_discover_tests` (from doctest's own cmake/ dir), and provides    #
# the `ovr_add_cpp_test()` helper used by tests/cpp/CMakeLists.txt.         #
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

  # Keep test binaries grouped under <build>/tests/ for easy discovery.
  set_target_properties(${NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
  )

  # ----- Windows DLL staging -----------------------------------------------
  # On POSIX the test exes resolve their dependent shared libs via RPATH /
  # the build's flat <build>/ layout, so this is a no-op there.
  #
  # On Windows the SHARED outputs (renderlib.dll, rendercommon.dll, glad.dll
  # built here + the OSPRay/TBB closure pulled in by find_package) all land
  # in <build>/<config>/ while these exes land in <build>/tests/<config>/.
  # Windows' loader searches the directory of the running exe first; that
  # directory has none of those DLLs, so doctest_discover_tests's spawn of
  # the freshly-built exe at PostBuildEvent time fails with 0xc0000135
  # (STATUS_DLL_NOT_FOUND) before main() runs - empty stdout, build fails.
  # ctest-time invocations would hit the same wall.
  #
  # Stage the runtime closure next to each test exe. CMake's
  # TARGET_RUNTIME_DLLS (3.21+) walks IMPORTED_LOCATION on linked imported /
  # built SHARED targets. Some binary packages, notably OSPRay, also require
  # sibling DLLs that are loaded by name and are not always represented in the
  # imported target graph, so configure_oneapi.cmake records those explicitly
  # in OVR_EXTRA_RUNTIME_DLLS. The $<IF> wraps the empty TARGET_RUNTIME_DLLS
  # case (e.g. gpu_probe links only cudart_static) so cmake -E doesn't trip on
  # a zero-source copy.
  if(WIN32 AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
    add_custom_command(TARGET ${NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E
              "$<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:${NAME}>>,copy_if_different,true>"
              "$<TARGET_RUNTIME_DLLS:${NAME}>"
              "$<TARGET_FILE_DIR:${NAME}>"
      COMMAND_EXPAND_LISTS
      VERBATIM
    )
    get_property(_ovr_extra_runtime_dlls GLOBAL PROPERTY OVR_EXTRA_RUNTIME_DLLS)
    if(_ovr_extra_runtime_dlls)
      list(REMOVE_DUPLICATES _ovr_extra_runtime_dlls)
      add_custom_command(TARGET ${NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_ovr_extra_runtime_dlls}
                "$<TARGET_FILE_DIR:${NAME}>"
        COMMAND_EXPAND_LISTS
        VERBATIM
      )
    endif()
  endif()

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

  # On Windows, do not use doctest_discover_tests. The doctest module bundled
  # with v2.4.11 always executes the test binary as a POST_BUILD step to
  # enumerate test cases; its DISCOVERY_MODE PRE_TEST option is not supported
  # in that version. That makes the build itself fail with MSB3073 when the
  # Windows loader cannot resolve a runtime DLL before main() starts. Register
  # one CTest entry per executable on Windows instead. This preserves coverage
  # and CI signal while keeping DLL-load failures in the test phase, after all
  # runtime staging has completed.
  if(WIN32)
    add_test(NAME ${NAME}
      COMMAND "$<TARGET_FILE:${NAME}>"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    set_tests_properties(${NAME} PROPERTIES
      LABELS "${_labels}"
    )
  else()
    # TEST_PREFIX is required so each registered CTest name starts with
    # "<binary>." (e.g. "test_serializer_json.create_json_scene: ..."). Other
    # CMake glue (tests/cpp/gpu_fixture_attach.cmake.in) attaches per-binary
    # FIXTURES_REQUIRED by matching this prefix; without it the regex never
    # fires and the fixture dependency silently never attaches.
    doctest_discover_tests(${NAME}
      ADD_LABELS 1
      TEST_PREFIX "${NAME}."
      PROPERTIES LABELS "${_labels}"
    )
  endif()

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

  set(_probe_src "${CMAKE_BINARY_DIR}/tests/gpu_probe.cpp")
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
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
  )

  # Mirror the DLL-staging the cpp tests get (see ovr_add_cpp_test). Today
  # gpu_probe only links cudart_static so the closure is empty, but a future
  # tweak that pulls in a SHARED dep would otherwise break the same way.
  if(WIN32 AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
    add_custom_command(TARGET gpu_probe POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E
              "$<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:gpu_probe>>,copy_if_different,true>"
              "$<TARGET_RUNTIME_DLLS:gpu_probe>"
              "$<TARGET_FILE_DIR:gpu_probe>"
      COMMAND_EXPAND_LISTS
      VERBATIM
    )
  endif()

  add_test(NAME gpu_probe COMMAND gpu_probe)
  set_tests_properties(gpu_probe PROPERTIES
    FIXTURES_SETUP gpu_available
    LABELS "gpu_probe"
    # Treat exit code 1 ("no CUDA device on this host") as Skipped, not
    # Failed. The fixture is still marked unavailable so dependent gpu-
    # labelled tests auto-skip - the only difference is that ctest itself
    # no longer returns non-zero on GPU-less hosts (CI runners, coverage
    # jobs that build with OptiX on but have no driver, ...).
    SKIP_RETURN_CODE 1
  )
endfunction()
