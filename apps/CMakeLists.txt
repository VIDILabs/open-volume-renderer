# ======================================================================== #
# apps
# ======================================================================== #

add_executable(renderbatch main_batch.cpp)
# `glfwapp_headers` is the header-only INTERFACE target that exposes
# `<glfwapp/camera_frame.h>` (a pure CameraFrame struct, no OpenGL deps).
target_link_libraries(renderbatch PUBLIC renderlib glfwapp_headers)

# the default rendering application
if(OVR_BUILD_OPENGL)
  add_executable(renderapp main_app.cpp)
  target_link_libraries(renderapp PUBLIC renderlib glfwapp)
endif()

# Wheel-bundling: when the Python bindings are built (the pip / uv flow),
# install the apps into <wheel>/ovrpy/bin/ so the entry-point shims in
# python/ovrpy/_apps.py can `os.execv` them. INSTALL_RPATH=$ORIGIN/..
# resolves librenderlib + the rest of the closure that lives in the
# package root one level up.
if(OVR_BUILD_PYTHON_BINDINGS)
  if(TARGET renderbatch)
    set_target_properties(renderbatch PROPERTIES INSTALL_RPATH "$ORIGIN/..")
    install(TARGETS renderbatch
            RUNTIME DESTINATION bin
            COMPONENT ovrwheel)
  endif()
  if(TARGET renderapp)
    set_target_properties(renderapp PROPERTIES INSTALL_RPATH "$ORIGIN/..")
    install(TARGETS renderapp
            RUNTIME DESTINATION bin
            COMPONENT ovrwheel)
  endif()
endif()
