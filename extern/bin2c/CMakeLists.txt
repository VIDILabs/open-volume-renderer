# This CMakeLists.txt compiles bin2c and demonstrates how 'bin2c' can
# be used to embed shader files for other targets
cmake_minimum_required(VERSION 3.8.0)
project(bin2c-test)

# Compile bin2c obviously
add_executable(bin2c bin2c.c)
target_compile_definitions(bin2c PRIVATE _CRT_SECURE_NO_WARNINGS)
# # Create our testing program
# add_executable(test test/main.cpp)

# # The testing program needs to read data from 'test/test.txt'
# add_custom_target(bin2c_run
#   COMMAND bin2c -o test/embeded.h -n target test/test.txt 
#   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#   COMMENT "generate embeded file in ${CMAKE_CURRENT_SOURCE_DIR}"
#   SOURCES test/main.cpp
# )
# add_dependencies(test bin2c_run)
