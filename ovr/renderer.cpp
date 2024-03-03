//. ======================================================================== //
//. Copyright 2019-2020 Qi Wu                                                //
//.                                                                          //
//. Licensed under the Apache License, Version 2.0 (the "License");          //
//. you may not use this file except in compliance with the License.         //
//. You may obtain a copy of the License at                                  //
//.                                                                          //
//.     http://www.apache.org/licenses/LICENSE-2.0                           //
//.                                                                          //
//. Unless required by applicable law or agreed to in writing, software      //
//. distributed under the License is distributed on an "AS IS" BASIS,        //
//. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
//. See the License for the specific language governing permissions and      //
//. limitations under the License.                                           //
//. ======================================================================== //

#include "renderer.h"
#include "serializer/serializer.h"

#ifdef OVR_BUILD_OPTIX7
#include "devices/optix7/device.h"
#endif

#ifdef OVR_BUILD_OSPRAY
#include "devices/ospray/device.h"
#endif

#include <ovr/common/dylink/ObjectFactory.h>
#include <ovr/common/dylink/Library.h>

#include <colormap.h>

#include <fstream>
#include <random>

using namespace ovr::math;
using namespace ovr;

namespace ovr {
} // namespace ovr

std::shared_ptr<MainRenderer>
create_renderer(std::string name)
{
#ifdef OVR_BUILD_OPTIX7
  if (name == "optix7")
    return std::make_shared<ovr::optix7::DeviceOptix7>();
#endif

#ifdef OVR_BUILD_OSPRAY
  if (name == "ospray")
    return std::make_shared<ovr::ospray::DeviceOSPRay>();
#endif

  ovr::dynamic::LibraryRepository::GetInstance()->addDefaultLibrary();
  ovr::dynamic::LibraryRepository::GetInstance()->add("device_" + name, true);

  return ovr::dynamic::details::objectFactory<MainRenderer>("renderer", name);

  throw std::runtime_error("unknown device name: " + name);
}
