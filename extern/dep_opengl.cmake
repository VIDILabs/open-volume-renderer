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

  # NOTE: `glfwapp` is intentionally NOT in GFX_LIBRARIES. It's a
  # higher-level wrapper that *consumes* GFX_LIBRARIES; including it
  # here would create a self-referencing list (glfwapp links
  # ${GFX_LIBRARIES} which contains glfwapp) and force `imgui PRIVATE
  # ${GFX_LIBRARIES}` to link its own consumer. Apps that want the
  # full shim should link `glfwapp` directly (see apps/CMakeLists.txt).

endif()
