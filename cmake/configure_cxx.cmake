cmake_minimum_required(VERSION 3.10)
include_guard(GLOBAL)

if(APPLE) # MacOS is not supported ...
	set(CMAKE_MACOSX_RPATH ON)
endif()
if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
