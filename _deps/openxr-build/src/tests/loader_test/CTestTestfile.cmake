# CMake generated Testfile for 
# Source directory: C:/Users/joshc/Projects/cpp/ovr-vr/projects/ovr-stereo/openxr/src/tests/loader_test
# Build directory: C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/tests/loader_test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(loader_test "C:/Users/joshc/Projects/cpp/ovr-vr/Debug/loader_test.exe")
  set_tests_properties(loader_test PROPERTIES  WORKING_DIRECTORY "C:/Users/joshc/Projects/cpp/ovr-vr/Debug" _BACKTRACE_TRIPLES "C:/Users/joshc/Projects/cpp/ovr-vr/projects/ovr-stereo/openxr/src/tests/loader_test/CMakeLists.txt;60;add_test;C:/Users/joshc/Projects/cpp/ovr-vr/projects/ovr-stereo/openxr/src/tests/loader_test/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(loader_test "C:/Users/joshc/Projects/cpp/ovr-vr/Release/loader_test.exe")
  set_tests_properties(loader_test PROPERTIES  WORKING_DIRECTORY "C:/Users/joshc/Projects/cpp/ovr-vr/Release" _BACKTRACE_TRIPLES "C:/Users/joshc/Projects/cpp/ovr-vr/projects/ovr-stereo/openxr/src/tests/loader_test/CMakeLists.txt;60;add_test;C:/Users/joshc/Projects/cpp/ovr-vr/projects/ovr-stereo/openxr/src/tests/loader_test/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(loader_test "C:/Users/joshc/Projects/cpp/ovr-vr/MinSizeRel/loader_test.exe")
  set_tests_properties(loader_test PROPERTIES  WORKING_DIRECTORY "C:/Users/joshc/Projects/cpp/ovr-vr/MinSizeRel" _BACKTRACE_TRIPLES "C:/Users/joshc/Projects/cpp/ovr-vr/projects/ovr-stereo/openxr/src/tests/loader_test/CMakeLists.txt;60;add_test;C:/Users/joshc/Projects/cpp/ovr-vr/projects/ovr-stereo/openxr/src/tests/loader_test/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(loader_test "C:/Users/joshc/Projects/cpp/ovr-vr/RelWithDebInfo/loader_test.exe")
  set_tests_properties(loader_test PROPERTIES  WORKING_DIRECTORY "C:/Users/joshc/Projects/cpp/ovr-vr/RelWithDebInfo" _BACKTRACE_TRIPLES "C:/Users/joshc/Projects/cpp/ovr-vr/projects/ovr-stereo/openxr/src/tests/loader_test/CMakeLists.txt;60;add_test;C:/Users/joshc/Projects/cpp/ovr-vr/projects/ovr-stereo/openxr/src/tests/loader_test/CMakeLists.txt;0;")
else()
  add_test(loader_test NOT_AVAILABLE)
endif()
subdirs("test_layers")
subdirs("test_runtimes")
