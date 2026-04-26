# ======================================================================== #
# C++ / CUDA doctest binaries                                               #
# ------------------------------------------------------------------------ #
# Each source file becomes its own executable so CTest can parallelize and #
# isolate failures. GPU tests auto-skip when `gpu_probe` fails (no device).#
# ======================================================================== #

# ------------------------------------------------------------- test_math
# gdt is header-only and already an INTERFACE library exported by the
# gdt/ subdirectory. We only need the include path.
ovr_add_cpp_test(test_math
  SOURCES test_math.cpp
  LINK    gdt rendercommon
)

# ----------------------------------------------------------- test_scene
ovr_add_cpp_test(test_scene
  SOURCES test_scene.cpp
  LINK    rendercommon renderlib
)

# ---------------------------------------------------- test_serializer_json
# Depends on the CTest synthetic_fixtures fixture to generate the scene.
ovr_add_cpp_test(test_serializer_json
  SOURCES test_serializer_json.cpp
  LINK    rendercommon renderlib
  DEFS    OVR_TEST_GENERATED_DIR="${OVR_TEST_GENERATED_DIR}"
          OVR_TEST_FIXTURES_DIR="${OVR_TEST_FIXTURES_DIR}"
)
# Every TEST_CASE discovered from test_serializer_json must see the
# synthetic scene on disk before it runs.
set_property(GLOBAL APPEND PROPERTY OVR_FIXTURE_CONSUMERS test_serializer_json)

# ----------------------------------------------------- test_transactional_value
ovr_add_cpp_test(test_transactional_value
  SOURCES test_transactional_value.cpp
  LINK    rendercommon
)

# --------------------------------------------------- test_cross_device_buffer
ovr_add_cpp_test(test_cross_device_buffer
  SOURCES test_cross_device_buffer.cpp
  LINK    rendercommon
)

# ------------------------------------------------------- test_count_tfn
ovr_add_cpp_test(test_count_tfn
  SOURCES test_count_tfn.cpp
  LINK    rendercommon renderlib
)

# -------------------------------------------------------- test_imageio
ovr_add_cpp_test(test_imageio
  SOURCES test_imageio.cpp
  LINK    rendercommon
)

# --------------------------------------------------- test_generate_mask (CPU)
# generate_sparse_sampling_mask_h is only a real function when CUDA is
# enabled (without CUDA it throws "unimplemented"). In both cases we build
# this test to verify either the property-based behaviour (CUDA) or the
# throw (no-CUDA). The binary is CPU-only from a build-system standpoint.
ovr_add_cpp_test(test_generate_mask
  SOURCES test_generate_mask.cpp
  LINK    rendercommon
)

# ----------------------------------------------------------- test_dda
# DDAIter (ovr/devices/optix7/accel/dda.h) is a CUDA `__device__` header
# so it lives under the CUDA-only tests block below when OVR_BUILD_CUDA
# is on. On host-only builds DDA is covered indirectly by the rendering
# regression tests (optix7 backend ends up invoking it).

# ========================================================= CUDA / GPU tests
if(OVR_BUILD_CUDA)
  enable_language(CUDA)

  # CrossDeviceBuffer CPU<->CUDA round-trip
  ovr_add_cpp_test(test_cuda_cross_device_buffer
    SOURCES test_cuda_cross_device_buffer.cu
    LINK    rendercommon CUDA::cudart_static
    GPU
  )
  set_target_properties(test_cuda_cross_device_buffer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

  # generate_sparse_sampling_mask property tests on GPU
  ovr_add_cpp_test(test_cuda_generate_mask
    SOURCES test_cuda_generate_mask.cu
    LINK    rendercommon CUDA::cudart_static
    GPU
  )
  set_target_properties(test_cuda_generate_mask PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

endif()

# --------------------------------------------------------------------------
# Wire up CTest fixture dependencies for tests that doctest discovers
# lazily. Runs at ctest-time via TEST_INCLUDE_FILES, which is the only
# hook that sees the auto-registered TEST_CASEs.
# --------------------------------------------------------------------------
get_property(_fixture_consumers GLOBAL PROPERTY OVR_FIXTURE_CONSUMERS)
list(JOIN _fixture_consumers ";" OVR_FIXTURE_CONSUMERS_STR)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/gpu_fixture_attach.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/gpu_fixture_attach.cmake"
  @ONLY
)
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
  "${CMAKE_CURRENT_BINARY_DIR}/gpu_fixture_attach.cmake"
)
