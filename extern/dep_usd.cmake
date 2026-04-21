cmake_minimum_required(VERSION 3.24)
include(FetchContent)

option(OVR_BUILD_USD "Build USDA Scene Reader" OFF)
if(OVR_BUILD_USD)
  find_package(pxr REQUIRED)
endif()
