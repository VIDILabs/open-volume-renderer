# ---------------------------------------------
# optix7 device  
# ---------------------------------------------
if(OVR_BUILD_DEVICE_OPTIX7)

  message(STATUS "Enable OptiX7")

  # TODO: why cuda_compile_and_embed does not transitively link to gdt? we have to explicitly link gdt here.
  cuda_compile_and_embed(embedded_ptx_code__raymarching ${CMAKE_CURRENT_LIST_DIR}/shaders_raymarching.cu LINK rendercommon gdt)
  cuda_compile_and_embed(embedded_ptx_code__pathtracing ${CMAKE_CURRENT_LIST_DIR}/shaders_pathtracing.cu LINK rendercommon gdt)

  add_library(device_optix7 STATIC
    ${embedded_ptx_code__raymarching}
    ${embedded_ptx_code__pathtracing}
    accel/sp_singlemc.cu
    accel/spatial_partition.h
    array.cpp
    array.h
    compute.cu
    device_impl.cpp
    device_impl.h
    device.cpp
    device.h
    optix7_common.h
    params.h
    volume.cpp
    volume.h
  )
  set_target_properties(device_optix7 
  PROPERTIES 
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )
  target_link_libraries(device_optix7 PRIVATE
    rendercommon
    OptiX::optix7
    TBB::tbb
    $<$<AND:$<BOOL:${OVR_BUILD_OPENGL}>,$<TARGET_EXISTS:imgui>>:imgui>
  )
  target_link_libraries     (device_optix7 PRIVATE   cuda) # LNK2019: unresolved external symbol cuCtxGetCurrent referenced in function
  target_compile_options    (device_optix7 PRIVATE   $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
  target_compile_definitions(device_optix7 INTERFACE OVR_BUILD_OPTIX7=1)

  target_link_libraries(renderlib PUBLIC device_optix7)

  # ----------------------------------------------------------------------
  # Windows-only: lazy-load nvcuda.dll
  # ----------------------------------------------------------------------
  # device_optix7 PRIVATE-links `cuda` (cuda.lib -> nvcuda.dll on Windows)
  # to resolve cuCtxGetCurrent. Because device_optix7 is STATIC and
  # renderlib is SHARED, that import gets baked into renderlib.dll's
  # import table.
  #
  # nvcuda.dll only ships with the NVIDIA *driver*, not the CUDA
  # toolkit. GHA windows-latest runners install the toolkit (via
  # Jimver/cuda-toolkit) but have no GPU and no driver, so the loader
  # can't resolve nvcuda.dll -> renderlib.dll fails to map ->
  # STATUS_DLL_NOT_FOUND (0xc0000135) on any process that imports a
  # renderlib symbol. doctest's post-build --list-test-cases probe of
  # test_serializer_json (the only test that calls a renderlib-defined
  # symbol, scene::create_json_scene) is what surfaces this.
  #
  # Delay-loading defers the nvcuda.dll resolution until the first cu*
  # call. Tests that never exercise an OptiX code path (json
  # serializer round-trip, scene helpers, ...) load fine. Tests that
  # do hit cu* are gpu-labelled and already auto-skip on no-GPU hosts
  # via the gpu_probe fixture, so we don't regress those either.
  if(WIN32)
    target_link_options(renderlib PRIVATE /DELAYLOAD:nvcuda.dll)
    target_link_libraries(renderlib PRIVATE delayimp)
  endif()

endif()
