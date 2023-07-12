project(ImPlot LANGUAGES CXX)
cmake_minimum_required(VERSION 3.5)

add_library(implot
    implot_demo.cpp
    implot_items.cpp
    implot.cpp
)

target_link_libraries(implot imgui)

target_include_directories(implot PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
)

set_target_properties(implot PROPERTIES POSITION_INDEPENDENT_CODE ON)
