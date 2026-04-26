# ======================================================================== #
# OVR test-suite top-level CMakeLists.                                      #
# ------------------------------------------------------------------------ #
# Registers:                                                                #
#   * C++ / CUDA doctest binaries (via tests/cpp/)                          #
#   * A `python_tests` CTest test that invokes pytest in tests/python/      #
#   * A CTest fixture that generates synthetic volume data on demand        #
#                                                                           #
# Tests are sliced by label:                                                #
#   ctest -L cpp      # fast C++ unit tests                                 #
#   ctest -L gpu      # CUDA-dependent tests                                #
#   ctest -L python   # python bindings + rendering regression              #
#   ctest -LE gpu     # everything except GPU tests                         #
#                                                                           #
# Hard requirements when OVR_BUILD_TESTS=ON:                                #
#   * Python 3 interpreter                                                  #
#   * `pytest` importable from that interpreter                             #
#   * OVR_BUILD_PYTHON_BINDINGS=ON and the `_core` (ovrpy._core) target     #
#                                                                           #
# Any of these missing is a configure-time error. The previous AUTO mode   #
# silently skipped python_tests when pytest wasn't installed, which made   #
# it possible for `ctest` to return green without ever running the Python  #
# tier - that hole is now closed.                                          #
# ======================================================================== #

if(COMMAND ovr_setup_gpu_probe)
  ovr_setup_gpu_probe()
endif()

set(OVR_TEST_FIXTURES_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
set(OVR_TEST_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_fixtures")

# ---------------------------------------------------------------------------
# Hard requirements. Building the test suite without these is an error -
# we never want a half-configured build to silently drop test coverage.
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter REQUIRED)

if(NOT OVR_BUILD_PYTHON_BINDINGS)
  message(FATAL_ERROR
    "OVR_BUILD_TESTS=ON requires OVR_BUILD_PYTHON_BINDINGS=ON: the Python "
    "test tier exercises the `ovrpy` module. Either turn on the bindings "
    "or turn off OVR_BUILD_TESTS.")
endif()

# Probe for pytest at configure time. We deliberately make this a hard
# error rather than auto-installing: configure-time package installs are
# surprising, and the right answer (`pip install -e .[test]`) is something
# the developer's environment manager (venv, conda, system) should handle.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import pytest"
  RESULT_VARIABLE _ovr_pytest_rc
  OUTPUT_QUIET ERROR_QUIET
)
if(NOT _ovr_pytest_rc EQUAL 0)
  message(FATAL_ERROR
    "OVR tests require pytest, but `import pytest` failed with "
    "${Python3_EXECUTABLE}.\n"
    "Install the test dependencies with:\n"
    "    ${Python3_EXECUTABLE} -m pip install -e \"${CMAKE_SOURCE_DIR}\"[test]\n"
    "...then re-run cmake. (Use a virtualenv if your system Python is "
    "managed.)")
endif()

# ---------------------------------------------------------------------------
# Synthetic fixture generation is a CTest fixture, not a build-time custom
# target: it costs nothing when tests aren't run, and it keeps Python as
# the only generator (no compile-a-helper-then-run dance). Tests that
# consume the fixture declare FIXTURES_REQUIRED=synthetic_fixtures.
# ---------------------------------------------------------------------------
add_test(NAME generate_synthetic_fixtures
  COMMAND ${Python3_EXECUTABLE}
          "${CMAKE_CURRENT_SOURCE_DIR}/generate_fixtures.py"
          --output-dir "${OVR_TEST_GENERATED_DIR}"
)
set_tests_properties(generate_synthetic_fixtures PROPERTIES
  FIXTURES_SETUP synthetic_fixtures
  LABELS "fixtures"
)

add_subdirectory(cpp)

# ---------------------------------------------------------------- pytest
# python_tests is unconditionally registered: any of Python3, pytest, or
# the _core target being missing is caught above as a FATAL_ERROR.
if(NOT TARGET _core)
  message(FATAL_ERROR
    "OVR tests expected the `_core` pybind11 target (ovrpy._core) to exist "
    "by the time tests/CMakeLists.txt runs, but it doesn't. Check that "
    "OVR_BUILD_PYTHON_BINDINGS=ON and that python/CMakeLists.txt was "
    "added to the build.")
endif()

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

# `_core` is staged into <build>/ovrpy/ alongside the pure-Python package
# files (see python/CMakeLists.txt), so putting <build> on sys.path makes
# `import ovrpy` resolve. conftest.py treats OVR_BUILD_DIR as the parent of
# the `ovrpy/` package directory.
add_test(
  NAME python_tests
  COMMAND ${Python3_EXECUTABLE} -m pytest ${_pytest_args}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(python_tests PROPERTIES
  LABELS "python"
  FIXTURES_REQUIRED synthetic_fixtures
  ENVIRONMENT
    "OVR_BUILD_DIR=${CMAKE_BINARY_DIR};OVR_TEST_FIXTURES=${OVR_TEST_FIXTURES_DIR};OVR_TEST_GENERATED=${OVR_TEST_GENERATED_DIR}"
)
