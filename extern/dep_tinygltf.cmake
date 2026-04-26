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
# tinygltf (formerly vendored at extern/tinygltf/). Pinned to v2.5.0 to
# match the version that was previously checked in. Upstream:
# https://github.com/syoyo/tinygltf
#
# Gated by `OVR_BUILD_TINYGLTF`. Consumers do
# `target_link_libraries(my_target PRIVATE tinygltf)` and
# `#include <tinygltf/tiny_gltf.h>`.
#
# Note the namespaced include path: tinygltf bundles its own copies of
# `stb_image.h`, `stb_image_write.h`, and `json.hpp` at the upstream
# repo root. Putting that root on the consumer's include path directly
# would shadow our own `dep_stb` / `dep_json` (or vice versa) depending
# on `target_link_libraries` order - subtle and fragile. We sidestep it
# by mirroring the upstream tree into `build/tinygltf_compat/tinygltf/`
# and only exposing the parent on the include path. tinygltf's internal
# quote-includes (`#include "stb_image.h"`) still resolve to its private
# bundled copies because quote-include search starts from the file's
# own dir.
#
# We bypass tinygltf's own CMakeLists.txt (via the bare
# FetchContent_Populate path rather than _MakeAvailable) because that
# upstream CMakeLists builds a loader_example executable and emits install
# rules - both unwanted noise for our build/wheel.
option(OVR_BUILD_TINYGLTF "Fetch tinygltf via FetchContent and expose it as the `tinygltf` INTERFACE target" ON)

if(OVR_BUILD_TINYGLTF)
  include(FetchContent)

  FetchContent_Declare(tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG        v2.5.0
    GIT_SHALLOW    ON
  )
  FetchContent_GetProperties(tinygltf)
  if(NOT tinygltf_POPULATED)
    FetchContent_Populate(tinygltf)
  endif()

  set(_OVR_TINYGLTF_COMPAT_DIR "${CMAKE_BINARY_DIR}/tinygltf_compat")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
      "${tinygltf_SOURCE_DIR}"
      "${_OVR_TINYGLTF_COMPAT_DIR}/tinygltf"
  )

  add_library(tinygltf INTERFACE)
  target_include_directories(tinygltf INTERFACE
    $<BUILD_INTERFACE:${_OVR_TINYGLTF_COMPAT_DIR}>
  )
endif()
