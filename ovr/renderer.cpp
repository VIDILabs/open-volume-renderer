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

// Scene
// create_example_scene()
// {
//   static std::random_device rng;        // Will be used to obtain a seed for the random number engine
//   static std::mt19937 generator(rng()); // Standard mersenne_twister_engine seeded with rng()
//   static std::uniform_int_distribution<> index(0, (int)(colormap::name.size() - 1ULL));
// 
//   scene::TransferFunction tfn;
// 
//   // /* S3D data, range = (-7.6862547926737625e-19,  0.00099257915280759335) */
//   // scene::Volume volume;
//   // volume.type = scene::Volume::STRUCTURED_REGULAR_VOLUME;
//   // volume.structured_regular.data =
//   //   CreateArray3DScalarFromFile("data/3.900E-04_H2O2.raw", vec3i(600, 750, 500), VALUE_TYPE_FLOAT, 0, false);
//   // volume.structured_regular.grid_origin = -vec3i(600, 750, 500) / 2.f;
//   // tfn.value_range = vec2f(0.f, 0.0002f);
//   // tfn.color = CreateColorMap(colormap::name[index(generator)]);
//   // tfn.opacity = CreateArray1DScalar(std::vector<float>{ 0.f, 1.f });
// 
//   /* 0.0070086f, 12.1394f */
//   scene::Volume volume;
//   volume.type = scene::Volume::STRUCTURED_REGULAR_VOLUME;
//   volume.structured_regular.data =
//     CreateArray3DScalarFromFile(std::string("data/vorts1.data"), vec3i(128, 128, 128), VALUE_TYPE_FLOAT, 0, false);
//   volume.structured_regular.grid_origin = -vec3i(128, 128, 128) * 2.f;
//   volume.structured_regular.grid_spacing = 4.f;
//   tfn.value_range = vec2f(0.0, 12.0);
//   tfn.color = CreateColorMap("sequential2/summer");
//   tfn.opacity = CreateArray1DScalar(std::vector<float>{ 0.f, 0.f, 0.01f, 0.2f, 0.01f, 0.f, 0.f });
// 
//   scene::Model model;
//   model.type = scene::Model::VOLUMETRIC_MODEL;
//   model.volume_model.volume = volume;
//   model.volume_model.transfer_function = tfn;
// 
//   scene::Instance instance;
//   instance.models.push_back(model);
//   instance.transform = affine3f::translate(vec3f(0));
// 
//   scene::Scene scene;
//   scene.instances.push_back(instance);
// 
//   return scene;
// }
