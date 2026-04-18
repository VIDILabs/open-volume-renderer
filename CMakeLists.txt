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
if (POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif (POLICY CMP0135)

cmake_minimum_required(VERSION 3.18)
project(OpenVisRenderer LANGUAGES C CXX)

include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/extern")

# `OVR_INSTALL_INCLUDEDIR` is the include root encoded into installed/exported
# OVR interface targets. Standalone OVR defaults to GNUInstallDirs' include
# directory, while parent projects embedding OVR may override it before calling
# `add_subdirectory(base)` if they install OVR headers into a package-specific
# subtree such as `include/<package>`.
if(NOT DEFINED OVR_INSTALL_INCLUDEDIR)
  set(OVR_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}")
endif()

option(OVR_BUILD_DEVICE_OPTIX7 "Build OptiX7 Device" ON)
option(OVR_BUILD_DEVICE_OSPRAY "Build OSPRay Device" ON)
if(OVR_BUILD_DEVICE_OPTIX7)
  set(OVR_BUILD_OPTIX7 TRUE)
endif()
if(OVR_BUILD_DEVICE_OSPRAY)
  set(OVR_BUILD_OSPRAY TRUE)
endif()

###############################################################################
# Setup Repositories
###############################################################################
include(configure_build_type)
include(configure_cxx)

option(OVR_BUILD_CUDA "Build with CUDA" ON)
if(OVR_BUILD_CUDA)
  include(configure_cuda)
endif()
if(OVR_BUILD_OPTIX7)
  include(configure_optix)
endif()

# Core Math Library
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/gdt)

# Other Libraries
include(configure_oneapi)
include(dep_opengl)
include(dep_usd)
find_package(Threads REQUIRED)
# find_package(OpenMP REQUIRED)

###############################################################################
# Python setup
###############################################################################
option(OVR_BUILD_PYTHON_BINDINGS "Build Python Bindings" OFF)
if(OVR_BUILD_PYTHON_BINDINGS)
  include(configure_python)
endif()

###############################################################################
# Application build
###############################################################################
option(OVR_BUILD_APPS "Build OVR application executables" ON)

###############################################################################
# OpenGL Shader Encoder
###############################################################################
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/extern/bin2c EXCLUDE_FROM_ALL)
include(${CMAKE_CURRENT_LIST_DIR}/extern/bin2c/target_add_embeded_shaders.cmake)

###############################################################################
# Transfer Function Module  
###############################################################################
set(TFNMODULE_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/extern/tfn/colormaps)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/extern/tfn/colormaps)
add_library(tfnmodule STATIC ${embedded_colormap})
target_include_directories(tfnmodule PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/extern/tfn/colormaps>
)
if (UNIX)
  set_target_properties(tfnmodule PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_subdirectory(ovr)
if(OVR_BUILD_APPS)
  add_subdirectory(apps)
endif()
add_subdirectory(projects)
if(OVR_BUILD_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()
