# ======================================================================== #
# Copyright 2018 Ingo Wald                                                 #
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
# ======================================================================== #
# Copyright 2019 - 2026 Qi Wu                                              #
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
cmake_minimum_required(VERSION 3.10)
include_guard(GLOBAL)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDAToolkit_INCLUDE_DIRS=${CUDAToolkit_INCLUDE_DIRS}")

if(NOT CMAKE_CUDA_ARCHITECTURES)
  # CMake 3.24+ resolves "native" to the installed GPU(s) at configure time and
  # stores the numeric result in CMAKE_CUDA_ARCHITECTURES_NATIVE.
  set(CMAKE_CUDA_ARCHITECTURES native)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES not set — defaulting to native")
endif()

if(WIN32)
  add_definitions(-DNOMINMAX)
endif()
add_definitions(-D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__=1)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# CUDA compiler setup
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
if (MSVC)
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-bigobj")
else()
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "-U__CUDA_NO_HALF_OPERATORS__")
list(APPEND CUDA_NVCC_FLAGS "-U__CUDA_NO_HALF_CONVERSIONS__")
list(APPEND CUDA_NVCC_FLAGS "-U__CUDA_NO_HALF2_OPERATORS__")

enable_language(CUDA)  # no-op if already enabled by parent
