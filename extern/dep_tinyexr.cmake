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
# tinyexr (formerly vendored at extern/tinyexr/). Pinned to v1.0.8.
# Upstream: https://github.com/syoyo/tinyexr
#
# Actively used by `ovr/common/imageio.cpp` (EXR read/write through the
# `tinyexr` INTERFACE target wired into rendercommon). Upstream v1.0+
# moved miniz to `deps/miniz/`, so we add both the repo root (for
# `tinyexr.h`) and the miniz subdir (for `miniz.c` which imageio.cpp
# pulls in directly to provide the implementation) to the include path.
#
# We bypass tinyexr's own CMakeLists.txt (FetchContent_Populate, not
# _MakeAvailable) because it builds a CLI test executable + emits install
# rules - both unwanted in our build/wheel.
include(FetchContent)

FetchContent_Declare(tinyexr
  GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
  GIT_TAG        v1.0.8
  GIT_SHALLOW    ON
)
FetchContent_GetProperties(tinyexr)
if(NOT tinyexr_POPULATED)
  FetchContent_Populate(tinyexr)
endif()

add_library(tinyexr INTERFACE)
target_include_directories(tinyexr INTERFACE
  $<BUILD_INTERFACE:${tinyexr_SOURCE_DIR}>
  $<BUILD_INTERFACE:${tinyexr_SOURCE_DIR}/deps/miniz>
)

# Modern tinyexr.h calls into miniz (`mz_compress`, `mz_compressBound`,
# `mz_uncompress`) but only declares them via `<miniz.h>` - the user must
# provide the implementation. We propagate `miniz.c` as an INTERFACE
# source so each consumer compiles it as C (CMake picks the language
# from the .c extension), which keeps tentative-definition handling
# C-style and avoids the C++ redefinition errors that occur when
# `miniz.c` is `#include`d inside a `.cpp` TU.
target_sources(tinyexr INTERFACE
  $<BUILD_INTERFACE:${tinyexr_SOURCE_DIR}/deps/miniz/miniz.c>
)
