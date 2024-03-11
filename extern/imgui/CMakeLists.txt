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

# TODO: copy from GPRT, require some refactoring
# Find Vulkan
if (NOT CMAKE_VERSION VERSION_LESS 3.7.0)
  message(STATUS "Using module to find Vulkan")
  find_package(Vulkan)
endif()

IF(WIN32)
  IF(NOT Vulkan_FOUND)
    find_library(Vulkan_LIBRARY NAMES vulkan-1 vulkan PATHS ${CMAKE_SOURCE_DIR}/libs/vulkan)
    IF (Vulkan_LIBRARY)
      set(Vulkan_FOUND ON)
      MESSAGE("Using bundled Vulkan library version")
    ENDIF()
  ENDIF()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WIN32_KHR")
ELSEIF(LINUX)
  IF (NOT Vulkan_FOUND)
    find_library(Vulkan_LIBRARY NAMES vulkan HINTS "$ENV{VULKAN_SDK}/lib" "${CMAKE_SOURCE_DIR}/libs/vulkan" REQUIRED)
    IF (Vulkan_LIBRARY)
      set(Vulkan_FOUND ON)
      MESSAGE("Using bundled Vulkan library version")
    ENDIF()
  ENDIF()
  find_package(Threads REQUIRED)
  IF(USE_D2D_WSI)
    MESSAGE("Using direct to display extension...")
    add_definitions(-D_DIRECT2DISPLAY)
  ELSEIF(USE_DIRECTFB_WSI)
    find_package(DirectFB REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_DIRECTFB_EXT")
    include_directories(${DIRECTFB_INCLUDE_DIR})
  ELSEIF(USE_WAYLAND_WSI)
    find_program(PKG_CONFIG pkg-config)
    if (NOT PKG_CONFIG)
      message(FATAL_ERROR "pkg-config binary not found")
    endif ()
    find_package(Wayland REQUIRED)
    if (NOT WAYLAND_FOUND)
      message(FATAL_ERROR "Wayland development package not found")
    endif ()
    pkg_check_modules(WAYLAND_PROTOCOLS REQUIRED wayland-protocols)
    if (NOT WAYLAND_PROTOCOLS_FOUND)
      message(FATAL_ERROR "Wayland protocols package not found")
    endif ()
    find_program(WAYLAND_SCANNER wayland-scanner)
    if (NOT WAYLAND_SCANNER)
      message(FATAL_ERROR "wayland-scanner binary not found")
    endif ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WAYLAND_KHR")
    include_directories(${WAYLAND_INCLUDE_DIR})
    pkg_get_variable(protocol_dir wayland-protocols pkgdatadir)
    execute_process(COMMAND ${WAYLAND_SCANNER} client-header ${protocol_dir}/stable/xdg-shell/xdg-shell.xml ${CMAKE_BINARY_DIR}/xdg-shell-client-protocol.h
        COMMAND ${WAYLAND_SCANNER} private-code ${protocol_dir}/stable/xdg-shell/xdg-shell.xml ${CMAKE_BINARY_DIR}/xdg-shell-protocol.c)
    include_directories(${CMAKE_BINARY_DIR})
  ELSEIF(USE_HEADLESS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_HEADLESS_EXT")
  ELSE(USE_D2D_WSI)
    find_package(XCB REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_XCB_KHR")
  ENDIF(USE_D2D_WSI)
ELSEIF(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_MACOS_MVK -DVK_EXAMPLE_XCODE_GENERATED")
  # Todo : android?
ENDIF(WIN32)

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
  imgui/examples/$<$<BOOL:${Vulkan_FOUND}>:imgui_impl_vulkan.cpp>
  implot/implot_demo.cpp
  implot/implot_items.cpp
  implot/implot.cpp
)

target_link_libraries(imgui ${GFX_LIBRARIES} ${Vulkan_LIBRARY})

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
