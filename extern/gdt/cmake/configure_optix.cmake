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

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(configure_cuda)

find_package(OptiX REQUIRED)

find_program(BIN2C bin2c DOC "Path to the cuda-sdk bin2c executable.")

# this macro defines cmake rules that execute the following four steps:
# 1) compile the given cuda file ${cuda_file} to an intermediary PTX file
# 2) use the 'bin2c' tool (that comes with CUDA) to
#    create a second intermediary (.c-)file which defines a const string variable
#    (named '${c_var_name}') whose (constant) value is the PTX output
#    from the previous step.
# 3) compile the given .c file to an intermediary object file (why thus has
#    that PTX string 'embedded' as a global constant.
# 4) assign the name of the intermediary .o file to the cmake variable
#    'output_var', which can then be added to cmake targets.
macro(cuda_compile_and_embed output_var cuda_file)

  set(options)
  set(oneArgs)
  set(mulArgs OPTIONS LINK DEPENDS)
  cmake_parse_arguments(in "${options}" "${oneArgs}" "${mulArgs}" ${ARGN})

  set(opts ${in_OPTIONS} -Xcudafe="--diag_suppress=20044")

  list(APPEND opts -I"${OptiX_INCLUDE}")
  if (CUDAToolkit_INCLUDE_DIRS)
  	list(APPEND opts -I"${CUDAToolkit_INCLUDE_DIRS}")
  endif()

  foreach(tar ${in_LINK})
    get_target_property(inc ${tar} INTERFACE_INCLUDE_DIRECTORIES)
    foreach(path ${inc})
      list(APPEND opts "-I${path}")
    endforeach()
  endforeach()

  set(c_var_name ${output_var})
  cuda_compile_ptx(ptx_files ${cuda_file} OPTIONS ${opts} -DENABLE_OPTIX)
  list(GET ptx_files 0 ptx_file)
  set(embedded_file ${ptx_file}_embedded.c)

  # message("ptx: ${cuda_file} OPTIONS ${opts}")
  # message("adding rule to compile and embed ${cuda_file} to \"const char ${var_name}[];\"")

  add_custom_command(
    OUTPUT ${embedded_file}
    COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
    DEPENDS ${ptx_file} ${in_DEPENDS}
    COMMENT "compiling (and embedding ptx from) ${cuda_file}")
  set(${output_var} ${embedded_file})

endmacro()

# include_directories(${OptiX_INCLUDE})
