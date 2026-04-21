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

endif()
