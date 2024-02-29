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
#define OVR_REGISTER_OBJECT(Object, Name, InternalClass, ExtName)    \
extern "C" OVR_DEVICE_API Object *                                   \
  ovr_create_##Name##__##ExtName()                                   \
  {                                                                  \
    auto *instance = new InternalClass;                              \
    return instance;                                                 \
  }                                                                  \
  /* additional declaration to avoid "extra ;" -Wpedantic warnings */\
  Object *ovr_create_##Name##__##ExtName();
// clang-format on
