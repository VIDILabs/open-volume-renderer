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
# nothings/stb (formerly vendored at extern/stbi/). Upstream:
# https://github.com/nothings/stb
#
# Pinned to commit 5736b15f7ea0ffb08dd38af21067c314d6a3aae9 (mid-2023),
# which ships stb_image.h v2.29 + stb_image_write.h v1.16 - newer than
# the vendored copies (v2.25 / v1.14) but API-compatible.
#
# Used by `ovr/common/imageio.cpp`. The repo doesn't tag releases, so
# the pin is a commit hash; bump deliberately when you want a newer
# version. We bypass the upstream CMakeLists.txt (which builds the
# many test/example executables under tests/) by populating the source
# dir directly and exposing it as a header-only INTERFACE target.
include(FetchContent)

FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        5736b15f7ea0ffb08dd38af21067c314d6a3aae9
)
FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
  FetchContent_Populate(stb)
endif()

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE
  $<BUILD_INTERFACE:${stb_SOURCE_DIR}>
)
