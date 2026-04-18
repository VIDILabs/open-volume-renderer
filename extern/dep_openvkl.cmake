cmake_minimum_required(VERSION 3.24)
include(FetchContent)

option(ENABLE_OPENVKL "Enable OpenVKL CPU sampling backend" OFF)
if(NOT ENABLE_OPENVKL)
  return()
endif()

find_package(TBB CONFIG QUIET)
if(NOT TARGET TBB::tbb)
  message(FATAL_ERROR "ENABLE_OPENVKL requires TBB. Install libtbb-dev or set TBB_DIR.")
endif()

# ── rkcommon ──────────────────────────────────────────────────────────────────
set(RKCOMMON_VERSION "1.13.0" CACHE STRING "rkcommon version")
set(RKCOMMON_TASKING_SYSTEM "TBB" CACHE STRING "" FORCE)
set(BUILD_TESTING           OFF   CACHE BOOL   "" FORCE)
FetchContent_Declare(rkcommon
  GIT_REPOSITORY  https://github.com/ospray/rkcommon.git
  GIT_TAG         v${RKCOMMON_VERSION}
  GIT_SHALLOW     ON
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(rkcommon)
if(TARGET rkcommon AND NOT TARGET rkcommon::rkcommon)
  add_library(rkcommon::rkcommon ALIAS rkcommon)
endif()

# ── embree ────────────────────────────────────────────────────────────────────
set(EMBREE_VERSION           "4.3.3" CACHE STRING "Embree version")
set(EMBREE_ISPC_SUPPORT      OFF CACHE BOOL   "" FORCE)
set(EMBREE_TUTORIALS         OFF CACHE BOOL   "" FORCE)
set(EMBREE_TESTING_INTENSITY 0   CACHE STRING "" FORCE)
set(EMBREE_STATIC_RUNTIME    OFF CACHE BOOL   "" FORCE)
FetchContent_Declare(embree
  GIT_REPOSITORY  https://github.com/RenderKit/embree.git
  GIT_TAG         v${EMBREE_VERSION}
  GIT_SHALLOW     ON
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(embree)

# ── openvkl ───────────────────────────────────────────────────────────────────
set(OPENVKL_VERSION    "2.0.1" CACHE STRING "OpenVKL version")
set(OPENVKL_DEVICE_GPU OFF CACHE BOOL "" FORCE)
set(OPENVKL_DEVICE_CPU ON  CACHE BOOL "" FORCE)
set(BUILD_TESTING      OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS   OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES     OFF CACHE BOOL "" FORCE)
set(BUILD_TOOLS        OFF CACHE BOOL "" FORCE)
FetchContent_Declare(openvkl
  GIT_REPOSITORY  https://github.com/openvkl/openvkl.git
  GIT_TAG         v${OPENVKL_VERSION}
  GIT_SHALLOW     ON
)
FetchContent_MakeAvailable(openvkl)
if(TARGET openvkl AND NOT TARGET openvkl::openvkl)
  add_library(openvkl::openvkl ALIAS openvkl)
endif()
if(TARGET openvkl_module_cpu_device AND NOT TARGET openvkl::openvkl_module_cpu_device)
  add_library(openvkl::openvkl_module_cpu_device ALIAS openvkl_module_cpu_device)
endif()

foreach(_w 4 8 16)
  if(TARGET openvkl_module_cpu_device_${_w})
    target_include_directories(openvkl_module_cpu_device_${_w}
      PUBLIC $<BUILD_INTERFACE:${openvkl_SOURCE_DIR}>)
  endif()
endforeach()
unset(_w)

