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

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

# ======================================================================== #
# render common
# ======================================================================== #
add_library(rendercommon SHARED
  common/imageio.cpp
  common/generate_mask.cpp
  common/dylink/Library.cpp
  scene.cpp
  scene.h
)
target_link_libraries(rendercommon PUBLIC gdt tfnmodule TBB::tbb Threads::Threads)
target_link_libraries(rendercommon PRIVATE ${CMAKE_DL_LIBS})
target_include_directories(rendercommon PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/..)
target_include_directories(rendercommon PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/common)
target_include_directories(rendercommon PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if(OVR_BUILD_CUDA)
  target_sources(rendercommon PRIVATE 
    common/generate_mask.cu
    common/cuda/texture.cu
  )
  target_link_libraries(rendercommon PUBLIC CUDA::cudart_static) # this is useful for including CUDABuffer in C++ code
  target_compile_options(rendercommon PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
  target_compile_definitions(rendercommon PUBLIC OVR_BUILD_CUDA_DEVICES=1)
  target_include_directories(rendercommon PUBLIC ${CMAKE_CURRENT_LIST_DIR}/common/cuda)
endif()

# embed data into source

# file(GLOB NOISE_FILES "${PROJECT_SOURCE_DIR}/data/noise/*.bin")
# add_custom_command(TARGET rendercommon
#                   POST_BUILD
#                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                   ${NOISE_FILES}
#                   $<TARGET_FILE_DIR:rendercommon>)

target_add_embeded_shaders(rendercommon 
  OUTPUT_NAME ${CMAKE_CURRENT_BINARY_DIR}/noise_files.h 
  SHADERS
  blue_64x64x64=${PROJECT_SOURCE_DIR}/data/noise/blue_64x64x64.bin 
  stbn_128x128x64=${PROJECT_SOURCE_DIR}/data/noise/stbn_128x128x64.bin
)

# Allow device to define its own GUI pannel
if(OVR_BUILD_OPENGL)
  target_link_libraries(rendercommon PUBLIC imgui)
  target_compile_definitions(rendercommon PUBLIC OVR_BUILD_OPENGL=1)
endif()

# ======================================================================== #
# renderer
# ======================================================================== #
add_library(renderlib SHARED
  serializer/serializer.h
  serializer/serializer_json.cpp
  imageops/imageop.cpp
  imageops/$<$<BOOL:${OVR_BUILD_DEVICE_OPTIX7}>:optix7_denoiser.cpp>
  renderer.cpp
  renderer.h
)
target_link_libraries(renderlib 
PUBLIC 
  rendercommon
PRIVATE
  $<$<BOOL:${OVR_BUILD_DEVICE_OPTIX7}>:OptiX::optix7>
)
target_include_directories(renderlib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(renderlib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..)

if(OVR_BUILD_SCENE_USD)
  target_sources(renderlib PRIVATE serializer/serializer_usda.cpp)
  target_link_libraries(renderlib PRIVATE ${PXR_LIBRARIES})
  target_include_directories(renderlib PRIVATE ${PXR_INCLUDE_DIRS})
  target_compile_definitions(renderlib PUBLIC OVR_BUILD_SCENE_USD=1)
endif()

# filesystem
# GUN: https://en.cppreference.com/w/cpp/filesystem 
#   Using this library may require additional compiler/linker options. GNU implementation prior 
#   to 9.1 requires linking with -lstdc++fs and LLVM implementation prior to LLVM 9.0 requires 
#   linking with -lc++fs.
# LLVM: https://releases.llvm.org/9.0.0/projects/libcxx/docs/UsingLibcxx.html#using-filesystem
#   Prior to LLVM 9.0, libc++ provides the implementation of the filesystem library in a separate
#   static library. Users of <filesystem> and <experimental/filesystem> are required to link -lc++fs. 
#   Prior to libc++ 7.0, users of <experimental/filesystem> were required to link libc++experimental.
if (NOT APPLE)
  target_link_libraries(renderlib PUBLIC $<$<CXX_COMPILER_ID:GNU>:stdc++fs>)
  target_link_libraries(renderlib PUBLIC $<$<CXX_COMPILER_ID:Clang>:c++fs>)
endif()

# ---------------------------------------------
# devices  
# ---------------------------------------------
add_subdirectory(devices/optix7)
add_subdirectory(devices/ospray)
