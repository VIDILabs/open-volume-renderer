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
# Gated by `OVR_BUILD_TINYGLTF` (default OFF) - currently no OVR target
# links against this, so the fetch is opt-in. Flip it ON when you want
# `target_link_libraries(my_target PRIVATE tinygltf)` to make
# `#include "tiny_gltf.h"` available.
#
# We deliberately bypass tinygltf's own CMakeLists.txt (via the bare
# FetchContent_Populate path rather than _MakeAvailable) because that
# upstream CMakeLists builds a loader_example executable and emits install
# rules - both unwanted noise for our build/wheel.
option(OVR_BUILD_TINYGLTF "Fetch tinygltf via FetchContent and expose it as the `tinygltf` INTERFACE target" OFF)

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

  add_library(tinygltf INTERFACE)
  target_include_directories(tinygltf INTERFACE
    $<BUILD_INTERFACE:${tinygltf_SOURCE_DIR}>
  )
endif()
