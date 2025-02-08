# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/Users/joshc/Projects/cpp/ovr-vr/glfw/src")
  file(MAKE_DIRECTORY "C:/Users/joshc/Projects/cpp/ovr-vr/glfw/src")
endif()
file(MAKE_DIRECTORY
  "C:/Users/joshc/Projects/cpp/ovr-vr/glfw/build"
  "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/glfw-subbuild/glfw-populate-prefix"
  "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/glfw-subbuild/glfw-populate-prefix/tmp"
  "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/glfw-subbuild/glfw/stamp"
  "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/glfw-subbuild/glfw"
  "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/glfw-subbuild/glfw/stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/glfw-subbuild/glfw/stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/joshc/Projects/cpp/ovr-vr/_deps/glfw-subbuild/glfw/stamp${cfgdir}") # cfgdir has leading slash
endif()
