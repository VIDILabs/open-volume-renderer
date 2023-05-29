cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mini_torch C CXX CUDA)

if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif(POLICY CMP0054)

find_package(CUDA REQUIRED)

# https://blog.csdn.net/xzq1207105685/article/details/117400187
find_package(PythonInterp REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(exp_mini_torch example.cpp)
target_link_libraries(exp_mini_torch "${TORCH_LIBRARIES}")
set_property(TARGET exp_mini_torch PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET exp_mini_torch
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:exp_mini_torch>)
endif (MSVC)
