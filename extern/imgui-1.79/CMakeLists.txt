# ======================================================================== #
# Copyright 2018-2021 Qi Wu                                                #
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
project(DearImGui LANGUAGES CXX)
cmake_minimum_required(VERSION 3.5)

add_library(imgui
  imgui.cpp
  imgui_draw.cpp
  imgui_demo.cpp
  imgui_widgets.cpp
  examples/imgui_impl_glfw.cpp
  examples/imgui_impl_opengl2.cpp
  examples/imgui_impl_opengl3.cpp
)

set_target_properties(imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(imgui glad::glad ${GFX_LIBRARIES})

target_include_directories(imgui PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/examples/opengl2_example>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/examples/opengl3_example>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/examples>
)

target_compile_definitions(imgui 
PUBLIC 
  IMGUI_IMPL_OPENGL_LOADER_GLAD
)
