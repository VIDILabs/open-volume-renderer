if(OVR_BUILD_DEVICE_OSPRAY)

  message(STATUS "Enable OSPRay")

  add_library(device_ospray STATIC
    device.cpp
    device.h
    device_impl.cpp
    device_impl.h
  )
  target_link_libraries     (device_ospray PRIVATE    rendercommon ospray::ospray)
  target_compile_definitions(device_ospray INTERFACE  OVR_BUILD_OSPRAY=1)
  set_target_properties(device_ospray 
  PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
  )
  target_link_libraries(renderlib PUBLIC device_ospray)

endif()
