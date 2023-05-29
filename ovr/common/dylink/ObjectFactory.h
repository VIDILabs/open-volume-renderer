//===========================================================================//
//                                                                           //
// LibViDi3D                                                                 //
// Copyright(c) Qi Wu (Wilson)                                               //
// University of California, Davis                                           //
// MIT Licensed                                                              //
//                                                                           //
//===========================================================================//

#ifndef OVR_COMMON_DYNAMIC_OBJECTFACTORY_H
#define OVR_COMMON_DYNAMIC_OBJECTFACTORY_H

#include "Library.h"
#include <memory>
// #include <vidiBase.h>

#include <string>
#include <algorithm>
#include <map>

namespace ovr {
/**
 * @brief The ViDi tools for implementing dynamically loadable modules
 */
namespace dynamic {

namespace details {

/**
 * The function to create an object from a string
 * @tparam T The base type
 * @param type The external type name
 * @return A new instance of the object
 */
template<typename T>
std::shared_ptr<T>
objectFactory(const std::string& tstr, const std::string& type)
{
  // Function pointer type for creating a concrete instance of a subtype of
  // this class.
  using creationFunctionPointer = T* (*)();

  // Function pointers corresponding to each subtype.
  creationFunctionPointer symbol;

  // // type of the abstract class
  // const auto tstr = dict.at(typeid(T).hash_code()); // getTypeName<T>();
  // const auto tstr = std::to_string(typeid(T).hash_code());

  // Construct the name of the creation function to look for.
  std::string creationFunctionName = "ovr_create_" + tstr + "__" + type;

  // Load library from the disk
  auto& repo = *LibraryRepository::GetInstance();
  repo.addDefaultLibrary();

  // Look for the named function.
  symbol = (creationFunctionPointer)repo.getSymbol(creationFunctionName);

  // Create a concrete instance of the requested subtype.
  if (symbol) {
    auto* object = (*symbol)();
    std::shared_ptr<T> ret(object);
    return ret;
  }
  else {
    throw UnknownDynamicModuleError(tstr, type);
  }
}

} // namespace details
} // namespace dynamic
} // namespace vidi


// clang-format off
#define OVR_REGISTER_OBJECT(Object, object_name, InternalClass, external_name)\
  extern "C" /*VIDI_EXPORT*/                                                   \
      Object *ovr_create_##object_name##__##external_name()                   \
  {                                                                            \
    auto *instance = new InternalClass;                                        \
    return instance;                                                           \
  }                                                                            \
  /* additional declaration to avoid "extra ;" -Wpedantic warnings */          \
  Object *ovr_create_##object_name##__##external_name();
// clang-format on

#endif // OVR_COMMON_DYNAMIC_OBJECTFACTORY_H
