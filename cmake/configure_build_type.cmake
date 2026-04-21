# ======================================================================== #
# Copyright 2018-2020 Ingo Wald                                            #
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
# Copyright 2020 - 2026 Qi Wu                                              #
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

# This helper script sets up default build targets for Release/Debug, etc,
# something which each project I worked on seems to need, eventually, so
# having it in one place arguably makes sense.

###############################################################################
# Configure
###############################################################################

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

# set library output path
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(APPLE) # MacOS is not supported ...
	set(CMAKE_MACOSX_RPATH ON)
endif()
if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
else()
	# if(BUILD_SHARED_LIBS)
	# 	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    #   set(CMAKE_POSITION_INDEPENDENT_CODE ON)
	# endif()
endif()
# if(NOT WIN32)
#   # visual studio doesn't like these (not need them):
#   set(CMAKE_CXX_FLAGS "--std=c++17")
#   set(CUDA_PROPAGATE_HOST_FLAGS ON)
# endif()
# if(UNIX)
#   set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# endif()

if(NOT SET_UP_CONFIGURATIONS_DONE)
    set(SET_UP_CONFIGURATIONS_DONE 1)

    # Set a default configuration if none was specified
    if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "No release type specified. Setting to 'Release'.")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
    endif()

    # # No reason to set CMAKE_CONFIGURATION_TYPES if it's not a multiconfig generator
    # # Also no reason mess with CMAKE_BUILD_TYPE if it's a multiconfig generator.
    # if(CMAKE_CONFIGURATION_TYPES) # multiconfig generator?
    #     set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE) 
    # else()
    #     if(NOT CMAKE_BUILD_TYPE)
    #         # message("Defaulting to release build.")
    #         set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
    #     endif()
    #     set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
    #     # set the valid options for cmake-gui drop-down list
    #     set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release")
    # endif()
endif()

# SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
# SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
# SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
