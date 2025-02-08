# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.31.0-rc2)

# Reject any attempt to use a toolchain file. We must not use one because
# we could be downloading it here. If the CMAKE_TOOLCHAIN_FILE environment
# variable is set, the cache variable will have been initialized from it.
unset(CMAKE_TOOLCHAIN_FILE CACHE)
unset(ENV{CMAKE_TOOLCHAIN_FILE})

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(openxr-populate NONE)



include(ExternalProject)
ExternalProject_Add(openxr-populate
                     "UPDATE_DISCONNECTED" "False" "DOWNLOAD_EXTRACT_TIMESTAMP" "TRUE" "EXTERNALPROJECT_INTERNAL_ARGUMENT_SEPARATOR" "URL_HASH" "MD5=924a94a2da0b5ef8e82154c623d88644" "URL" "https://github.com/KhronosGroup/OpenXR-SDK-Source/archive/refs/tags/release-1.0.34.zip"
                    SOURCE_DIR          "C:/Users/joshc/Projects/cpp/ovr-vr/projects/ovr-stereo/openxr"
                    BINARY_DIR          "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/openxr-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


