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

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDAToolkit_INCLUDE_DIRS="${CUDAToolkit_INCLUDE_DIRS})

if((NOT EXISTS ${GDT_CUDA_ARCHITECTURES}) AND (DEFINED ENV{GDT_CUDA_ARCHITECTURES}))
  set(GDT_CUDA_ARCHITECTURES $ENV{GDT_CUDA_ARCHITECTURES})
endif()

if (DEFINED GDT_CUDA_ARCHITECTURES)

  message(STATUS "Obtained target architecture from environment variable GDT_CUDA_ARCHITECTURES=${GDT_CUDA_ARCHITECTURES}")
  set(CMAKE_CUDA_ARCHITECTURES ${GDT_CUDA_ARCHITECTURES})
  if (NOT PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CUDA_ARCHITECTURES ${GDT_CUDA_ARCHITECTURES} PARENT_SCOPE)
  endif()

else()

  # adapted from https://stackoverflow.com/a/69353718
  include(FindCUDA/select_compute_arch)
  CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
  string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
  string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
  string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
  if (NOT PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST} PARENT_SCOPE)
  endif()
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
  message(STATUS "Automatically detected GPU architectures: ${CUDA_ARCH_LIST}")

endif()

# include_directories(${CUDA_TOOLKIT_INCLUDE})

if(WIN32)
  add_definitions(-DNOMINMAX)
endif()
add_definitions(-D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__=1)
