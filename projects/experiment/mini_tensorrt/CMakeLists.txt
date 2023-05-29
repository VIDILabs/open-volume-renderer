cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mini_tensorrt C CXX CUDA)

if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif(POLICY CMP0054)

find_package(PythonInterp REQUIRED)
find_package(Torch REQUIRED)
find_package(TensorRT REQUIRED)
find_package(TorchTensorRT REQUIRED)

add_executable(torch2rt torch2rt.cpp)
set_target_properties(torch2rt PROPERTIES CXX_STANDARD 17)
target_link_libraries(torch2rt PRIVATE ${TORCH_LIBRARIES} ${TENSORRT_LIBRARIES} ${TORCHTRT_LIBRARIES})
