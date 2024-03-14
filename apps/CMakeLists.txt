# ======================================================================== #
# apps
# ======================================================================== #

add_executable(renderbatch main_batch.cpp)
target_link_libraries(renderbatch PUBLIC renderlib)

# the default rendering application
if(OVR_BUILD_OPENGL)
  add_executable(renderapp main_app.cpp)
  target_link_libraries(renderapp 
  PUBLIC 
    renderlib 
    glfwApp
  )
endif()
