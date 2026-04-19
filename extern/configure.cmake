# ======================================================================== #
# Copyright 2019-2020 Qi Wu                                                #
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

# ------------------------------------------------------------------
# first, include gdt project to do some general configuration stuff
# (build modes, glut, optix, etc)
# ------------------------------------------------------------------
include_directories(${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/bin2c EXCLUDE_FROM_ALL)
include(${CMAKE_CURRENT_LIST_DIR}/bin2c/target_add_embeded_shaders.cmake)

# ------------------------------------------------------------------
# load external system libraries
# ------------------------------------------------------------------
find_package(Threads REQUIRED)
# find_package(OpenMP  REQUIRED)

# ------------------------------------------------------------------
# import gdt submodule
# ------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/gdt/cmake")
include(configure_build_type)
include_directories(${CMAKE_CURRENT_LIST_DIR}/gdt)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/gdt EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
# find OpenGL
# ------------------------------------------------------------------
if(OVR_BUILD_OPENGL)

  set(OpenGL_GL_PREFERENCE GLVND)
  find_package(OpenGL REQUIRED)
  if(TARGET OpenGL::OpenGL)
    list(APPEND GFX_LIBRARIES OpenGL::OpenGL)
  else()
    list(APPEND GFX_LIBRARIES OpenGL::GL)
  endif()
  if(TARGET OpenGL::GLU)
    list(APPEND GFX_LIBRARIES OpenGL::GLU)
  endif()
  if(TARGET OpenGL::GLX)
    list(APPEND GFX_LIBRARIES OpenGL::GLX)
  endif()

  # build glfw
  include(extern/dep_glfw.cmake)
  list(APPEND GFX_LIBRARIES glfw)

  # build glad
  include(extern/dep_glad.cmake)
  list(APPEND GFX_LIBRARIES glad)

  # import imgui
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/imgui EXCLUDE_FROM_ALL)
  list(APPEND GFX_LIBRARIES imgui)

  # for building render apps
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/glfwapp EXCLUDE_FROM_ALL)
  list(APPEND GFX_LIBRARIES glfwApp)

endif()

# ------------------------------------------------------------------
# import colormap
# ------------------------------------------------------------------
set(TFNMODULE_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/tfn/colormaps)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tfn/colormaps)
add_library(tfnmodule STATIC ${embedded_colormap})
target_include_directories(tfnmodule PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/tfn/colormaps>
)
if (UNIX)
  set_target_properties(tfnmodule PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# ------------------------------------------------------------------
# import CUDA
# ------------------------------------------------------------------
if(OVR_BUILD_CUDA)  
  include(configure_cuda)
  mark_as_advanced(CUDA_SDK_ROOT_DIR)
endif()

# ------------------------------------------------------------------
# import Optix7
# ------------------------------------------------------------------
if(OVR_BUILD_OPTIX7)  
  include(configure_optix)
endif(OVR_BUILD_OPTIX7)

# ------------------------------------------------------------------
# import OneAPI
# ------------------------------------------------------------------
find_package(TBB REQUIRED)

if(OVR_BUILD_OSPRAY)
  find_package(ospray 2.0 REQUIRED)
endif(OVR_BUILD_OSPRAY)

# if(OVR_BUILD_OPENVKL)
#   find_package(rkcommon REQUIRED)
#   find_package(openvkl REQUIRED)
# endif(OVR_BUILD_OPENVKL)

# ------------------------------------------------------------------
# import USD
# ------------------------------------------------------------------
if(OVR_BUILD_SCENE_USD)
  find_package(pxr REQUIRED)
endif()


# ------------------------------------------------------------------
# import pybind11
# ------------------------------------------------------------------
if(OVR_BUILD_PYTHON_BINDINGS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/pybind11)
endif()
