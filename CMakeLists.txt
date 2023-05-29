# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
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

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
if (POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif (POLICY CMP0054)
if (POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif (POLICY CMP0104)

project(OpenVisRenderer LANGUAGES C CXX)

cmake_minimum_required(VERSION 3.18)
# if(NOT WIN32)
#   # visual studio doesn't like these (not need them):
#   set(CMAKE_CXX_FLAGS "--std=c++17")
#   set(CUDA_PROPAGATE_HOST_FLAGS ON)
# endif()
# if(UNIX)
#   set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# endif()

option(OVR_BUILD_OPENGL "Build with OpenGL Apps" ON)
option(OVR_BUILD_CUDA "Build with CUDA" ON)

option(OVR_BUILD_DEVICE_OPTIX7  "Build OptiX7 Device"  ON)
option(OVR_BUILD_DEVICE_OSPRAY  "Build OSPRay Device" OFF)
option(OVR_BUILD_SCENE_USD "Build USDA Scene Reader" OFF)

# ------------------------------------------------------------------
# toggles
# ------------------------------------------------------------------

if(OVR_BUILD_DEVICE_OPTIX7)
  set(OVR_BUILD_OPTIX7 TRUE)
endif()

if(OVR_BUILD_DEVICE_OSPRAY)
  set(OVR_BUILD_OSPRAY TRUE)
endif()

# ------------------------------------------------------------------
# configure
# ------------------------------------------------------------------

if(APPLE) # MacOS is not supported ...
	set(CMAKE_MACOSX_RPATH ON)
endif()
if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
else()
	# if(BUILD_SHARED_LIBS)
	# 	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
	# endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-bigobj")
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	if(BUILD_SHARED_LIBS)
		list(APPEND CUDA_NVCC_FLAGS "-fPIC")
	endif()
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

if(OVR_BUILD_CUDA)
  # adapted from https://stackoverflow.com/a/69353718
  find_package(CUDA REQUIRED)
  # enable_language(CUDA)

  if(DEFINED GDT_CUDA_ARCHITECTURES) 
    message(STATUS "Obtained target architecture from environment variable GDT_CUDA_ARCHITECTURES=${GDT_CUDA_ARCHITECTURES}")
    set(CMAKE_CUDA_ARCHITECTURES ${GDT_CUDA_ARCHITECTURES})
  endif()

  if(NOT CMAKE_CUDA_ARCHITECTURES)
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    message(STATUS "Automatically detected GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  endif()
  
  # we can only enable CUDA if CMAKE_CUDA_ARCHITECTURES is set
  if (CMAKE_CUDA_ARCHITECTURES)
    enable_language(CUDA)
  else()
    message(FATAL_ERROR "Cannot utomatically detected GPU architecture")
  endif()
endif()

include(extern/configure.cmake)

add_subdirectory(ovr)
add_subdirectory(apps)
add_subdirectory(projects)
