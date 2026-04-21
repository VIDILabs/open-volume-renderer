# ======================================================================== #
# Copyright 2020 - 2026 Qi Wu                                              #
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

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/extern")

find_package(TBB REQUIRED)

if(OVR_BUILD_OSPRAY)
  find_package(ospray 2.0)
  if(NOT TARGET ospray::ospray)
    include(FetchContent)
    if(WIN32)
      set(_ospray_url "https://github.com/wilsonCernWq/ospray/releases/download/sparse_sampling_v2.11.0/ospray-2.11.0.x86_64.windows_sparse_sampling.zip")
    else()
      set(_ospray_url "https://github.com/wilsonCernWq/ospray/releases/download/sparse_sampling_v2.11.0/ospray-2.11.0.x86_64.linux_sparse_sampling.tar.gz")
    endif()
    FetchContent_Declare(ospray_binary
      URL "${_ospray_url}"
      SOURCE_DIR "${CMAKE_BINARY_DIR}/ospray_binary"
    )
    FetchContent_GetProperties(ospray_binary)
    if(NOT ospray_binary_POPULATED)
      FetchContent_Populate(ospray_binary)
    endif()
    set(ospray_DIR "${ospray_binary_SOURCE_DIR}/lib/cmake/ospray-2.11.0" CACHE PATH "" FORCE)
    find_package(ospray 2.0 REQUIRED)
  endif()
  if(NOT TARGET ospray::ospray)
    message(FATAL_ERROR "ospray not found")
  endif()
endif()

if(OVR_BUILD_OPENVKL)
  find_package(openvkl)
  if(NOT TARGET openvkl::openvkl)
    include(dep_openvkl)
  endif()
  if(NOT TARGET openvkl::openvkl)
    message(FATAL_ERROR "openvkl not found")
  endif()
endif()
