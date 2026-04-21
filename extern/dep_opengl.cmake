cmake_minimum_required(VERSION 3.24)

option(OVR_BUILD_OPENGL "Build with OpenGL Apps" ON)

# ------------------------------------------------------------------
# find OpenGL
# ------------------------------------------------------------------
if(OVR_BUILD_OPENGL)

  set(OpenGL_GL_PREFERENCE GLVND)
  find_package(OpenGL REQUIRED)
  if(TARGET OpenGL::OpenGL)
    list(APPEND GFX_LIBRARIES OpenGL::OpenGL)
  else()
    list(APPEND GFX_LIBRARIES OpenGL::GL)
  endif()
  if(TARGET OpenGL::GLU)
    list(APPEND GFX_LIBRARIES OpenGL::GLU)
  endif()
  if(TARGET OpenGL::GLX)
    list(APPEND GFX_LIBRARIES OpenGL::GLX)
  endif()

  # build glfw
  include(dep_glfw)
  list(APPEND GFX_LIBRARIES glfw)

  # build glad
  include(dep_glad)
  list(APPEND GFX_LIBRARIES glad)

  # import imgui + implot (fetched via FetchContent)
  include(dep_imgui)
  list(APPEND GFX_LIBRARIES imgui)

  # for building render apps
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/glfwapp EXCLUDE_FROM_ALL)
  list(APPEND GFX_LIBRARIES glfwApp)

endif()
