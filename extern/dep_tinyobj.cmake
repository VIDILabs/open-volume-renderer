# ======================================================================== #
# Copyright 2019-2026 Qi Wu                                                #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #
#
# tinyobjloader (formerly vendored at extern/tinyobj/). Pinned to v1.0.6
# to match the timeframe of the previously checked-in copy. Upstream:
# https://github.com/tinyobjloader/tinyobjloader
#
# Gated by `OVR_BUILD_TINYOBJ`. Consumers do
# `target_link_libraries(my_target PRIVATE tinyobj)` and
# `#include <tinyobj/tiny_obj_loader.h>`.
#
# tinyobj doesn't bundle any third-party deps (no stb / json), so the
# `tinyobj/` subdir scoping here is purely for symmetry with
# `dep_tinygltf` - a defensive pattern that keeps each FetchContent
# target isolated to its own namespaced include prefix.
#
# Bypasses upstream's CMakeLists.txt (which builds example/test
# executables) by populating the source dir directly and exposing it as
# a header-only INTERFACE target.
option(OVR_BUILD_TINYOBJ "Fetch tinyobjloader via FetchContent and expose it as the `tinyobj` INTERFACE target" ON)

if(OVR_BUILD_TINYOBJ)
  include(FetchContent)

  FetchContent_Declare(tinyobj
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG        v1.0.6
    GIT_SHALLOW    ON
  )
  FetchContent_GetProperties(tinyobj)
  if(NOT tinyobj_POPULATED)
    FetchContent_Populate(tinyobj)
  endif()

  set(_OVR_TINYOBJ_COMPAT_DIR "${CMAKE_BINARY_DIR}/tinyobj_compat")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
      "${tinyobj_SOURCE_DIR}"
      "${_OVR_TINYOBJ_COMPAT_DIR}/tinyobj"
  )

  add_library(tinyobj INTERFACE)
  target_include_directories(tinyobj INTERFACE
    $<BUILD_INTERFACE:${_OVR_TINYOBJ_COMPAT_DIR}>
  )
endif()
