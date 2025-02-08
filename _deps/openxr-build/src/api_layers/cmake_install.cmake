# Install script for directory: C:/Users/joshc/Projects/cpp/ovr-vr/projects/ovr-stereo/openxr/src/api_layers

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/OpenVisRenderer")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Layers" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/api_layers" TYPE FILE FILES "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/XrApiLayer_api_dump.json")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Layers" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/api_layers" TYPE MODULE FILES "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/XrApiLayer_api_dump.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/api_layers" TYPE MODULE FILES "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/XrApiLayer_api_dump.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/api_layers" TYPE MODULE FILES "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/XrApiLayer_api_dump.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/api_layers" TYPE MODULE FILES "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/XrApiLayer_api_dump.dll")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Layers" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    include("C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/CMakeFiles/XrApiLayer_api_dump.dir/install-cxx-module-bmi-Debug.cmake" OPTIONAL)
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    include("C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/CMakeFiles/XrApiLayer_api_dump.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    include("C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/CMakeFiles/XrApiLayer_api_dump.dir/install-cxx-module-bmi-MinSizeRel.cmake" OPTIONAL)
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    include("C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/CMakeFiles/XrApiLayer_api_dump.dir/install-cxx-module-bmi-RelWithDebInfo.cmake" OPTIONAL)
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Layers" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/api_layers" TYPE FILE FILES "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/XrApiLayer_core_validation.json")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Layers" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/api_layers" TYPE MODULE FILES "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/XrApiLayer_core_validation.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/api_layers" TYPE MODULE FILES "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/XrApiLayer_core_validation.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/api_layers" TYPE MODULE FILES "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/XrApiLayer_core_validation.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/api_layers" TYPE MODULE FILES "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/XrApiLayer_core_validation.dll")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Layers" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    include("C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/CMakeFiles/XrApiLayer_core_validation.dir/install-cxx-module-bmi-Debug.cmake" OPTIONAL)
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    include("C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/CMakeFiles/XrApiLayer_core_validation.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    include("C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/CMakeFiles/XrApiLayer_core_validation.dir/install-cxx-module-bmi-MinSizeRel.cmake" OPTIONAL)
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    include("C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/CMakeFiles/XrApiLayer_core_validation.dir/install-cxx-module-bmi-RelWithDebInfo.cmake" OPTIONAL)
  endif()
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build/src/api_layers/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
