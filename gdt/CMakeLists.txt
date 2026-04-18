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

project(GPUDevelopmentTools LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)

add_library(gdt INTERFACE gdt/gdt.h)

target_include_directories(gdt INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  # Installed consumers inherit the include root selected by OVR itself.
  $<INSTALL_INTERFACE:${OVR_INSTALL_INCLUDEDIR}>
)
