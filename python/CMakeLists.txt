if (OVR_BUILD_PYTHON_BINDINGS)
  pybind11_add_module(ovrpy python.cpp)
  target_link_libraries(ovrpy PRIVATE renderlib pybind11::pybind11)
  target_include_directories(ovrpy PUBLIC renderlib)

  find_program(STUBGEN stubgen)
  if (STUBGEN)
      message(STATUS "Found stubgen")
      add_custom_command(TARGET ovrpy POST_BUILD
          COMMAND ${STUBGEN} -m ovrpy -o .
          WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
          COMMENT "Generating .pyi index for python bindings")
  else()
      message(WARNING "Unable to find stubgen. Skipping generation of .pyi index for python bindings")
  endif()

endif()
