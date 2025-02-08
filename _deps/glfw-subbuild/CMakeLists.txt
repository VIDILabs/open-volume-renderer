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

project(glfw-populate NONE)



include(ExternalProject)
ExternalProject_Add(glfw-populate
                     "UPDATE_DISCONNECTED" "False" "DOWNLOAD_EXTRACT_TIMESTAMP" "TRUE" "EXTERNALPROJECT_INTERNAL_ARGUMENT_SEPARATOR" "DOWNLOAD_DIR" "glfw" "STAMP_DIR" "glfw/stamp" "URL" "https://github.com/glfw/glfw/archive/refs/tags/3.3.8.zip" "URL_HASH" "SHA256=8106e1a432305a8780b986c24922380df6a009a96b2ca590392cb0859062c8ff"
                    SOURCE_DIR          "C:/Users/joshc/Projects/cpp/ovr-vr/glfw/src"
                    BINARY_DIR          "C:/Users/joshc/Projects/cpp/ovr-vr/glfw/build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


