# ======================================================================== #
# Copyright 2018 Ingo Wald                                                 #
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

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(configure_cuda)

# Fetch OptiX headers if no local installation is pointed to
if(NOT DEFINED OptiX_INSTALL_DIR AND NOT DEFINED ENV{OptiX_INSTALL_DIR})
  include(FetchContent)
  FetchContent_Declare(optix_headers
    GIT_REPOSITORY  https://github.com/NVIDIA/optix-dev.git
    GIT_TAG         v7.4.0
    GIT_SHALLOW     ON
  )
  FetchContent_MakeAvailable(optix_headers)
  # Our cmake/FindOptiX.cmake uses OptiX_INSTALL_DIR; owl's FindOptiX uses OptiX_ROOT_DIR.
  set(OptiX_INSTALL_DIR "${optix_headers_SOURCE_DIR}" CACHE PATH "" FORCE)
  set(OptiX_ROOT_DIR    "${optix_headers_SOURCE_DIR}" CACHE PATH "" FORCE)
endif()

find_package(OptiX REQUIRED)

find_program(BIN2C bin2c HINTS "${CUDAToolkit_BIN_DIR}" DOC "Path to the cuda-sdk bin2c executable.")

# Compile a CUDA file to PTX and embed it as a C char array.
# Replaces the legacy cuda_compile_ptx (FindCUDA, removed in CMake 3.28).
#
# Usage:
#   cuda_compile_and_embed(<output_var> <cuda_file>
#     [LINK <target>...]   # targets whose INTERFACE_INCLUDE_DIRECTORIES are forwarded
#     [OPTIONS <flag>...]  # extra nvcc flags
#     [DEPENDS <file>...]  # extra file dependencies
#   )
# After the call, <output_var> holds the path to the generated .c file.
macro(cuda_compile_and_embed output_var cuda_file)

  set(options)
  set(oneArgs)
  set(mulArgs OPTIONS LINK DEPENDS)
  cmake_parse_arguments(in "${options}" "${oneArgs}" "${mulArgs}" ${ARGN})

  # ── build include list ────────────────────────────────────────────────────
  set(_cae_opts ${in_OPTIONS} -Xcudafe="--diag_suppress=20044")
  list(APPEND _cae_opts "-I${OptiX_INCLUDE}")
  if(CUDAToolkit_INCLUDE_DIRS)
    list(APPEND _cae_opts "-I${CUDAToolkit_INCLUDE_DIRS}")
  endif()
  foreach(_cae_tar ${in_LINK})
    get_target_property(_cae_inc ${_cae_tar} INTERFACE_INCLUDE_DIRECTORIES)
    if(_cae_inc)
      foreach(_cae_path ${_cae_inc})
        list(APPEND _cae_opts "-I${_cae_path}")
      endforeach()
    endif()
  endforeach()

  # ── pick PTX architecture: first concrete compute capability ─────────────
  # CMAKE_CUDA_ARCHITECTURES may be a keyword (native/all/all-major); CMake
  # 3.24+ resolves it to actual numeric values in CMAKE_CUDA_ARCHITECTURES_NATIVE.
  if(CMAKE_CUDA_ARCHITECTURES MATCHES "^(native|all|all-major)$")
    set(_cae_archs "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
  else()
    set(_cae_archs "${CMAKE_CUDA_ARCHITECTURES}")
  endif()
  list(GET _cae_archs 0 _cae_arch)
  string(REPLACE "+PTX"     "" _cae_arch "${_cae_arch}")
  string(REPLACE "-virtual" "" _cae_arch "${_cae_arch}")
  string(REPLACE "-real"    "" _cae_arch "${_cae_arch}")

  # ── output paths ──────────────────────────────────────────────────────────
  get_filename_component(_cae_base "${cuda_file}" NAME_WE)
  set(_cae_ptx      "${CMAKE_CURRENT_BINARY_DIR}/${_cae_base}.ptx")
  set(_cae_embedded "${CMAKE_CURRENT_BINARY_DIR}/${_cae_base}_embedded.c")

  # ── step 1: compile to PTX ────────────────────────────────────────────────
  add_custom_command(
    OUTPUT  "${_cae_ptx}"
    COMMAND ${CMAKE_CUDA_COMPILER}
            -ptx -std=c++17
            -arch=compute_${_cae_arch}
            ${_cae_opts} -DENABLE_OPTIX
            "${cuda_file}"
            -o "${_cae_ptx}"
    DEPENDS "${cuda_file}" ${in_DEPENDS}
    COMMENT "Compiling PTX: ${cuda_file}"
  )

  # ── step 2: embed PTX as C char array ────────────────────────────────────
  add_custom_command(
    OUTPUT  "${_cae_embedded}"
    COMMAND ${BIN2C} -c --padd 0 --type char --name ${output_var}
                     "${_cae_ptx}" > "${_cae_embedded}"
    DEPENDS "${_cae_ptx}" ${in_DEPENDS}
    COMMENT "Embedding PTX: ${_cae_base}"
  )

  set(${output_var} "${_cae_embedded}")
endmacro()
