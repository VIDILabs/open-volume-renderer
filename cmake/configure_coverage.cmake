# ======================================================================== #
# OVR coverage instrumentation (GCC / Clang only)                           #
#                                                                           #
# Enable with `-DOVR_ENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug`.           #
# Applies `--coverage -O0 -g` to every target that opts in via              #
# `ovr_apply_coverage_flags(<target>)`. A top-level `coverage` custom       #
# target is provided that runs ctest and invokes gcovr to produce an XML    #
# and an HTML report under ${CMAKE_BINARY_DIR}/coverage/.                   #
# ======================================================================== #

include_guard(GLOBAL)

if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  message(WARNING "OVR_ENABLE_COVERAGE requested but only GCC/Clang are supported; disabling.")
  set(OVR_ENABLE_COVERAGE OFF CACHE BOOL "" FORCE)
  return()
endif()

message(STATUS "OVR coverage instrumentation enabled (--coverage)")

# Apply coverage flags to host C/C++ compilation only; skip CUDA (nvcc doesn't
# understand --coverage directly). External dependencies built under extern/
# will also pick these up, but gcovr's --exclude below filters them back out.
add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:--coverage>
                    $<$<COMPILE_LANGUAGE:C,CXX>:-O0>
                    $<$<COMPILE_LANGUAGE:C,CXX>:-g>)
add_link_options(--coverage)

# Back-compat shim so existing calls to ovr_apply_coverage_flags() are no-ops.
function(ovr_apply_coverage_flags TARGET)
endfunction()

find_program(GCOVR_BIN gcovr)

if(GCOVR_BIN)
  add_custom_target(coverage
    COMMENT "Running ctest and collecting C++ coverage with gcovr"
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/coverage"
    COMMAND ${GCOVR_BIN}
      -r ${CMAKE_SOURCE_DIR}
      --exclude ${CMAKE_BINARY_DIR}
      --exclude ${CMAKE_SOURCE_DIR}/extern
      --exclude ${CMAKE_SOURCE_DIR}/test
      # nvcc's device-link step generates an ephemeral
      # /tmp/tmpxft_*_cmake_device_link.reg.c that's deleted before gcovr
      # runs. Without this flag, gcovr aborts on those .gcda files; with
      # it, gcovr just warns and skips them (what we want - the
      # device-link stub has no user-meaningful coverage anyway).
      --gcov-ignore-errors=source_not_found
      --xml ${CMAKE_BINARY_DIR}/coverage/coverage.xml
      --html-details ${CMAKE_BINARY_DIR}/coverage/index.html
    USES_TERMINAL
    VERBATIM
  )
else()
  add_custom_target(coverage
    COMMENT "gcovr not found; install with 'pip install gcovr'"
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E echo
      "gcovr not found; raw .gcda/.gcno remain under ${CMAKE_BINARY_DIR}"
  )
endif()
