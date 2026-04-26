# ======================================================================== #
# Copyright 2018-2021 Qi Wu                                                #
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

# Sources nest under `glfwapp/<name>` so consumers' `#include <glfwapp/X.h>`
# resolves through this target's own dir as the include root (no `..`
# parent traversal required).

# Header-only target that exposes `<glfwapp/X.h>` to consumers. Always
# built (no OpenGL deps) so that offline-render apps like `renderbatch`
# can `#include <glfwapp/camera_frame.h>` (a pure CameraFrame struct)
# without dragging the full GLFW/imgui stack onto their link line.
add_library(glfwapp_headers INTERFACE)
target_include_directories(glfwapp_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
)
target_link_libraries(glfwapp_headers INTERFACE gdt)

# Full GLFW + imgui app shim - only built when OVR_BUILD_OPENGL is ON
# (otherwise the imgui/glfw/GFX_LIBRARIES targets it depends on don't
# exist). Consumers linking glfwapp get the headers transitively.
if(OVR_BUILD_OPENGL)
    add_library(glfwapp STATIC glfwapp/GLFWApp.h glfwapp/GLFWApp.cpp)
    target_link_libraries(glfwapp PUBLIC glfwapp_headers imgui gdt ${GFX_LIBRARIES})
    target_compile_definitions(glfwapp PRIVATE GLFWAPP_USE_OPENGL2)
    if (UNIX)
        set_target_properties(glfwapp PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
endif()
