if (OVR_BUILD_PYTHON_BINDINGS)
  # The native extension is the underscore-prefixed implementation module
  # `ovrpy._core`. The Python-level `ovrpy` package (sitting alongside this
  # file under python/ovrpy/) re-exports its symbols so user-facing imports
  # remain `import ovrpy`.
  pybind11_add_module(_core python.cpp)
  target_link_libraries(_core PRIVATE renderlib pybind11::pybind11)
  target_include_directories(_core PUBLIC renderlib)

  # Stage <build>/ovrpy/{_core*.so,__init__.py,cli/...} so that the
  # cmake-driven dev flow (scripts/build.sh) can `import ovrpy` simply by
  # putting <build> on PYTHONPATH/sys.path. scikit-build-core's `pip install`
  # path uses the install rule below instead and ignores the staging.
  set(_OVRPY_STAGE_DIR "${CMAKE_BINARY_DIR}/ovrpy")
  set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_OVRPY_STAGE_DIR}"
    # When we eventually ship a redistributable wheel, sibling shared libs
    # (renderlib etc.) will be installed alongside `_core.so`; pre-set the
    # install RPATH so $ORIGIN-relative loading works without auditwheel.
    INSTALL_RPATH "$ORIGIN"
  )
  add_custom_command(TARGET _core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/ovrpy"
            "${_OVRPY_STAGE_DIR}"
    COMMENT "Staging ovrpy package next to _core extension")

  # ---------------------------------------------------------------------- #
  # scikit-build-core install rules (drive `pip install` / `uv sync`).
  #
  # Combined with `wheel.install-dir = "ovrpy"` in pyproject.toml, every
  # DESTINATION "." below resolves to `<wheel>/ovrpy/`. The runtime closure
  # required by `_core.so` is bundled there so site-packages/ovrpy/ is
  # self-contained:
  #
  #   ovrpy/
  #   ├── _core*.so                    pybind11 extension
  #   ├── librenderlib.so              renderer + statically-absorbed device backends
  #   ├── librendercommon.so           common runtime
  #   └── lib*.so* (OSPRay closure)    libospray, libtbb, libembree4,
  #                                    libopenvkl, libispcrt, libOpenImageDenoise,
  #                                    libopenvkl_module_cpu_device_{4,8,16}
  #
  # Each target's INSTALL_RPATH is set to `$ORIGIN` so the loader resolves
  # siblings from inside the wheel without LD_LIBRARY_PATH. OSPRay's own
  # libs already ship with `RUNPATH=$ORIGIN:$ORIGIN/../lib`, so once they
  # land next to each other they self-resolve.
  #
  # System-provided libs we deliberately do NOT bundle:
  #   * libcuda.so.1   - the user's NVIDIA driver
  #   * libstdc++/libc - system C++/C runtime
  # ---------------------------------------------------------------------- #
  # All wheel-bound install rules use COMPONENT "ovrwheel" so
  # scikit-build-core (configured with `install.components = ["ovrwheel"]`
  # in pyproject.toml) installs only what we want. Without this, third-party
  # FetchContent dependencies (glfw3, glad, OSPRay's cmake configs) leak
  # their own install rules into the wheel.
  set_target_properties(renderlib rendercommon PROPERTIES
    INSTALL_RPATH "$ORIGIN")

  install(TARGETS _core renderlib rendercommon
          LIBRARY DESTINATION .
          COMPONENT ovrwheel)

  # Interactive (OpenGL) tier. When OVR_BUILD_OPENGL=ON (the default), the
  # device backends compile with `#ifdef OVR_BUILD_OPENGL` enabled, which
  # makes librenderlib.so dynamically NEED libimgui.so, which itself needs
  # libglad.so. Both are built as shared libs by extern/dep_imgui.cmake +
  # extern/dep_glad.cmake, so we bundle them next to `_core.so`.
  #
  # The remaining transitive deps of libimgui.so (libX11, libGL/libOpenGL,
  # libvulkan) are _system_ libraries we don't ship; ovrpy/__init__.py
  # translates the loader error into an actionable install hint when any
  # of them are missing.
  if(OVR_BUILD_OPENGL AND TARGET imgui AND TARGET glad)
    set_target_properties(imgui glad PROPERTIES INSTALL_RPATH "$ORIGIN")
    install(TARGETS imgui glad
            LIBRARY DESTINATION .
            COMPONENT ovrwheel)
  endif()

  # OSPRay's `.so` closure - including OpenVKL's dlopen-by-name CPU device
  # modules (4/8/16-wide) - has to live next to `_core.so` for librenderlib
  # to load and for OSPRay's own module discovery to work. `ospray_DIR`
  # points at <prefix>/lib/cmake/ospray-X.Y.Z (set either by the local
  # FetchContent of ospray_binary or by an external -Dospray_DIR override),
  # so the actual lib dir is two levels up. Symlinks are preserved by
  # install(DIRECTORY).
  if(OVR_BUILD_DEVICE_OSPRAY AND ospray_DIR)
    get_filename_component(_OVR_OSPRAY_LIB_DIR "${ospray_DIR}/../.." ABSOLUTE)
    if(EXISTS "${_OVR_OSPRAY_LIB_DIR}")
      install(DIRECTORY "${_OVR_OSPRAY_LIB_DIR}/"
              DESTINATION .
              COMPONENT ovrwheel
              USE_SOURCE_PERMISSIONS
              FILES_MATCHING
                PATTERN "*.so"
                PATTERN "*.so.*"
                PATTERN "cmake" EXCLUDE
                PATTERN "pkgconfig" EXCLUDE)
    endif()
  endif()

  find_program(STUBGEN stubgen)
  if (STUBGEN)
      message(STATUS "Found stubgen")
      add_custom_command(TARGET _core POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E env
                  "PYTHONPATH=${CMAKE_BINARY_DIR}"
                  ${STUBGEN} -m ovrpy._core -o "${CMAKE_BINARY_DIR}"
          COMMENT "Generating .pyi stubs for ovrpy._core")
  else()
      message(WARNING "Unable to find stubgen. Skipping generation of .pyi index for python bindings")
  endif()

endif()
