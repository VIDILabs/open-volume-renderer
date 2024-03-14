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
cmake_minimum_required(VERSION 3.7.0)

find_package(Vulkan)

# Now it is important to make imgui a shared library
# TODO: verify the correctness on windows
# TODO: add support for Vulkan
add_library(imgui SHARED
  imgui/imgui.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_demo.cpp
  imgui/imgui_widgets.cpp
  imgui/examples/imgui_impl_glfw.cpp
  imgui/examples/imgui_impl_opengl2.cpp
  imgui/examples/imgui_impl_opengl3.cpp
  implot/implot_demo.cpp
  implot/implot_items.cpp
  implot/implot.cpp
)

target_link_libraries(imgui PRIVATE ${GFX_LIBRARIES})

if(Vulkan_FOUND) 
  target_sources(imgui PRIVATE imgui/examples/imgui_impl_vulkan.cpp)
  target_link_libraries(imgui PRIVATE ${Vulkan_LIBRARY})
endif()

target_include_directories(imgui PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/imgui/examples/opengl2_example>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/imgui/examples/opengl3_example>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/imgui/examples>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/imgui>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/implot>
)

target_compile_definitions(imgui 
PUBLIC 
  IMGUI_IMPL_OPENGL_LOADER_GLAD
)

# if (UNIX)
#   set_target_properties(imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)
# endif()
