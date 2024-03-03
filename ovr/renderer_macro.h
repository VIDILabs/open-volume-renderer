#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
# if defined(__GNUC__)
#  define OVR_DEVICE_API __attribute__ ((dllexport)) 
# else
#  define OVR_DEVICE_API __declspec(dllexport) 
# endif
#elif defined(__GNUC__)
# define OVR_DEVICE_API __attribute__ ((visibility ("default"))) 
#endif

// clang-format off

#define OVR_REGISTER_OBJECT(Object, Name, InternalClass, Device)     \
extern "C" OVR_DEVICE_API Object *                                   \
  ovr_create_##Name##__##Device()                                    \
  {                                                                  \
    auto *instance = new InternalClass;                              \
    return instance;                                                 \
  }                                                                  \
  /* additional declaration to avoid "extra ;" -Wpedantic warnings */\
  Object *ovr_create_##Name##__##Device();


#define OVR_REGISTER_SCENE_LOADER(Device, Filename) \
extern "C" OVR_DEVICE_API ovr::Scene ovr_create_scene__##Device(const char* Filename)

// clang-format on
