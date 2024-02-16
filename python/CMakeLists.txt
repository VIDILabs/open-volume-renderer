if (OVR_BUILD_PYTHON_BINDINGS)
  pybind11_add_module(ovrpy python.cpp)
  target_link_libraries(ovrpy PRIVATE renderlib)
endif()
