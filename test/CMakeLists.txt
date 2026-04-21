# ======================================================================== #
# OVR test-suite top-level CMakeLists.                                      #
# ------------------------------------------------------------------------ #
# Registers:                                                                #
#   * C++ / CUDA doctest binaries (via test/cpp/)                           #
#   * A `python_tests` CTest test that invokes pytest in test/python/       #
#   * A CTest fixture that generates synthetic volume data on demand        #
#                                                                           #
# Tests are sliced by label:                                                #
#   ctest -L cpp      # fast C++ unit tests                                 #
#   ctest -L gpu      # CUDA-dependent tests                                #
#   ctest -L python   # python bindings + rendering regression              #
#   ctest -LE gpu     # everything except GPU tests                         #
# ======================================================================== #

if(COMMAND ovr_setup_gpu_probe)
  ovr_setup_gpu_probe()
endif()

set(OVR_TEST_FIXTURES_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
set(OVR_TEST_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_fixtures")

# ---------------------------------------------------------------------------
# Synthetic fixture generation is a CTest fixture, not a build-time custom
# target: it costs nothing when tests aren't run, and it keeps Python as
# the only generator (no compile-a-helper-then-run dance). Tests that
# consume the fixture declare FIXTURES_REQUIRED=synthetic_fixtures.
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter QUIET)
if(NOT Python3_Interpreter_FOUND)
  find_package(Python COMPONENTS Interpreter QUIET)
  if(Python_Interpreter_FOUND)
    set(Python3_EXECUTABLE "${Python_EXECUTABLE}")
  endif()
endif()

if(Python3_EXECUTABLE)
  add_test(NAME generate_synthetic_fixtures
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/generate_fixtures.py"
            --output-dir "${OVR_TEST_GENERATED_DIR}"
  )
  set_tests_properties(generate_synthetic_fixtures PROPERTIES
    FIXTURES_SETUP synthetic_fixtures
    LABELS "fixtures"
  )
else()
  message(STATUS "OVR tests: no Python interpreter found; "
                 "synthetic fixture generation + pytest will be skipped")
endif()

add_subdirectory(cpp)

# ---------------------------------------------------------------- pytest
# Three-way control over whether `python_tests` gets registered:
#
#   OVR_PYTHON_TESTS=AUTO (default)
#       Probe for `import pytest`. Register if importable, else skip
#       with a clear status message.
#
#   OVR_PYTHON_TESTS=ON
#       Always register, regardless of the probe. Useful when pytest
#       will be installed after configure (e.g. CI installs deps in a
#       later step) or when the user simply wants ctest to report the
#       missing-dependency failure instead of silently skipping.
#
#   OVR_PYTHON_TESTS=OFF
#       Never register.
set(OVR_PYTHON_TESTS "AUTO" CACHE STRING
    "Whether to register the python_tests CTest entry (AUTO, ON, OFF)")
set_property(CACHE OVR_PYTHON_TESTS PROPERTY STRINGS AUTO ON OFF)

if(OVR_BUILD_PYTHON_BINDINGS AND TARGET ovrpy AND Python3_EXECUTABLE
   AND NOT OVR_PYTHON_TESTS STREQUAL "OFF")

  # Probe for pytest only in AUTO mode; ON forces registration even if the
  # probe would fail.
  set(_register_pytest TRUE)
  if(OVR_PYTHON_TESTS STREQUAL "AUTO")
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import pytest"
      RESULT_VARIABLE _pytest_rc
      OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT _pytest_rc EQUAL 0)
      set(_register_pytest FALSE)
      message(STATUS
        "OVR tests: pytest not importable via ${Python3_EXECUTABLE}; "
        "python_tests not registered. "
        "Install with `${Python3_EXECUTABLE} -m pip install -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt` "
        "and re-configure, or force registration with -DOVR_PYTHON_TESTS=ON.")
    endif()
  endif()

  if(_register_pytest)
    set(_pytest_args
      "${CMAKE_CURRENT_SOURCE_DIR}/python"
      --rootdir=${CMAKE_CURRENT_SOURCE_DIR}
      -v
      --color=yes
    )
    if(OVR_ENABLE_COVERAGE)
      list(APPEND _pytest_args
        --cov=ovrpy
        --cov-report=xml:${CMAKE_BINARY_DIR}/coverage/coverage-py.xml
        --cov-report=term
      )
    endif()

    add_test(
      NAME python_tests
      COMMAND ${Python3_EXECUTABLE} -m pytest ${_pytest_args}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(python_tests PROPERTIES
      LABELS "python"
      FIXTURES_REQUIRED synthetic_fixtures
      ENVIRONMENT
        "OVR_BUILD_DIR=$<TARGET_FILE_DIR:ovrpy>;OVR_TEST_FIXTURES=${OVR_TEST_FIXTURES_DIR};OVR_TEST_GENERATED=${OVR_TEST_GENERATED_DIR}"
    )

    if(OVR_PYTHON_TESTS STREQUAL "ON")
      message(STATUS
        "OVR tests: python_tests force-registered (OVR_PYTHON_TESTS=ON); "
        "install test/requirements.txt before running ctest.")
    endif()
  endif()
elseif(NOT Python3_EXECUTABLE)
  message(STATUS "OVR tests: Python interpreter not found; python_tests not registered")
elseif(OVR_PYTHON_TESTS STREQUAL "OFF")
  message(STATUS "OVR tests: python_tests disabled via OVR_PYTHON_TESTS=OFF")
elseif(NOT (OVR_BUILD_PYTHON_BINDINGS AND TARGET ovrpy))
  message(STATUS "OVR tests: ovrpy not built; python_tests not registered")
endif()
