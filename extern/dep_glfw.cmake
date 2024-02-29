# ======================================================================== #
# Copyright 2019-2024 Qi Wu                                                #
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
include(FetchContent)

set(COMPONENT_NAME glfw)

# set the options
set(GLFW_USE_OSMESA OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL ON CACHE BOOL "" FORCE)

# set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

mark_as_advanced(
  GLFW_INSTALL
  GLFW_BUILD_DOCS 
  GLFW_BUILD_TESTS 
  GLFW_BUILD_EXAMPLES
  GLFW_USE_OSMESA 
  GLFW_USE_WAYLAND 
  GLFW_VULKAN_STATIC
)

# fetch the content
FetchContent_Declare(${COMPONENT_NAME}
  DOWNLOAD_DIR ${COMPONENT_NAME}
  STAMP_DIR ${COMPONENT_NAME}/stamp
  SOURCE_DIR ${COMPONENT_NAME}/src
  BINARY_DIR ${COMPONENT_NAME}/build
  URL "https://github.com/glfw/glfw/archive/refs/tags/3.3.8.zip"
  URL_HASH "SHA256=8106e1a432305a8780b986c24922380df6a009a96b2ca590392cb0859062c8ff"
)
FetchContent_MakeAvailable(${COMPONENT_NAME})

# fix the include directory
target_include_directories(${COMPONENT_NAME} INTERFACE 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/${COMPONENT_NAME}/src/include>
)
message(STATUS ${CMAKE_CURRENT_BINARY_DIR}/${COMPONENT_NAME}/src/include)
