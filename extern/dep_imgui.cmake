# ======================================================================== #
# Copyright 2018-2024 Qi Wu                                                #
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
include_guard(GLOBAL)
include(FetchContent)
find_package(Git REQUIRED)

set(IMGUI_VERSION "v1.79")
set(IMPLOT_VERSION "v0.16")
set(IMGUI_PATCH "${CMAKE_CURRENT_LIST_DIR}/imgui.patch")
set(IMGUI_PATCH_HELPER "${CMAKE_CURRENT_LIST_DIR}/apply_imgui_patch.cmake")

FetchContent_Declare(imgui
  DOWNLOAD_DIR imgui
  STAMP_DIR    imgui/stamp
  SOURCE_DIR   imgui/src
  BINARY_DIR   imgui/build
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        ${IMGUI_VERSION}
  GIT_SHALLOW    TRUE
  PATCH_COMMAND
    ${CMAKE_COMMAND}
      "-DGIT_EXECUTABLE:FILEPATH=${GIT_EXECUTABLE}"
      "-DIMGUI_PATCH:FILEPATH=${IMGUI_PATCH}"
      "-DIMGUI_SOURCE_DIR:PATH=<SOURCE_DIR>"
      -P "${IMGUI_PATCH_HELPER}"
)
FetchContent_Declare(implot
  DOWNLOAD_DIR implot
  STAMP_DIR    implot/stamp
  SOURCE_DIR   implot/src
  BINARY_DIR   implot/build
  GIT_REPOSITORY https://github.com/epezent/implot.git
  GIT_TAG        ${IMPLOT_VERSION}
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(imgui implot)

# imgui has no CMakeLists.txt — define the target manually.
if(WIN32)
  add_library(imgui STATIC)
else()
  add_library(imgui SHARED)
endif()

target_sources(imgui PRIVATE
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/examples/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/examples/imgui_impl_opengl2.cpp
  ${imgui_SOURCE_DIR}/examples/imgui_impl_opengl3.cpp
  ${implot_SOURCE_DIR}/implot_demo.cpp
  ${implot_SOURCE_DIR}/implot_items.cpp
  ${implot_SOURCE_DIR}/implot.cpp
)

target_include_directories(imgui PUBLIC
  $<BUILD_INTERFACE:${imgui_SOURCE_DIR}/examples>
  $<BUILD_INTERFACE:${imgui_SOURCE_DIR}>
  $<BUILD_INTERFACE:${implot_SOURCE_DIR}>
)

target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)
target_link_libraries(imgui PRIVATE ${GFX_LIBRARIES})

find_package(Vulkan QUIET)
if(Vulkan_FOUND)
  target_sources(imgui PRIVATE ${imgui_SOURCE_DIR}/examples/imgui_impl_vulkan.cpp)
  target_link_libraries(imgui PRIVATE ${Vulkan_LIBRARY})
  target_include_directories(imgui PUBLIC $<BUILD_INTERFACE:${Vulkan_INCLUDE_DIR}>)
endif()
