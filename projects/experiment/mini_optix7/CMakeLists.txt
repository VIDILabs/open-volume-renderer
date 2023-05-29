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

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${OptiX_INCLUDE})
include_directories(${TBB_INCLUDE_DIRS})

# ======================================================================== #
# renderer
# ======================================================================== #
include_directories(${TFNMODULE_INCLUDE})
include_directories(${PROJECT_SOURCE_DIR}/common)
cuda_compile_and_embed(embedded_ptx_code renderer.cu)

add_executable(exp_mini_optix7
  ${embedded_ptx_code}
  volume.cpp
  renderer.cu
  renderer.cpp
  main.cpp
)
target_link_libraries(exp_mini_optix7
PRIVATE
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES} cuda
  TBB::tbb
  gdt tfnmodule glfwApp
)
set_property(TARGET exp_mini_optix7 PROPERTY CXX_STANDARD 17)
