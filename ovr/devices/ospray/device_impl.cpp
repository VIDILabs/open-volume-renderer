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

#include "device_impl.h"

#include <generate_mask.h>

#include <ospray/ospray_cpp.h>
#include <ospray/ospray_util.h>
#include <ospray/OSPEnums.h>

#include <tbb/parallel_for.h>

namespace ospray {
OSPTYPEFOR_SPECIALIZATION(gdt::vec2uc, OSP_VEC2UC);
OSPTYPEFOR_SPECIALIZATION(gdt::vec3uc, OSP_VEC3UC);
OSPTYPEFOR_SPECIALIZATION(gdt::vec4uc, OSP_VEC4UC);
OSPTYPEFOR_SPECIALIZATION(gdt::vec2c, OSP_VEC2C);
OSPTYPEFOR_SPECIALIZATION(gdt::vec3c, OSP_VEC3C);
OSPTYPEFOR_SPECIALIZATION(gdt::vec4c, OSP_VEC4C);
OSPTYPEFOR_SPECIALIZATION(gdt::vec2us, OSP_VEC2US);
OSPTYPEFOR_SPECIALIZATION(gdt::vec3us, OSP_VEC3US);
OSPTYPEFOR_SPECIALIZATION(gdt::vec4us, OSP_VEC4US);
OSPTYPEFOR_SPECIALIZATION(gdt::vec2s, OSP_VEC2S);
OSPTYPEFOR_SPECIALIZATION(gdt::vec3s, OSP_VEC3S);
OSPTYPEFOR_SPECIALIZATION(gdt::vec4s, OSP_VEC4S);
OSPTYPEFOR_SPECIALIZATION(gdt::vec2i, OSP_VEC2I);
OSPTYPEFOR_SPECIALIZATION(gdt::vec3i, OSP_VEC3I);
OSPTYPEFOR_SPECIALIZATION(gdt::vec4i, OSP_VEC4I);
OSPTYPEFOR_SPECIALIZATION(gdt::vec2ui, OSP_VEC2UI);
OSPTYPEFOR_SPECIALIZATION(gdt::vec3ui, OSP_VEC3UI);
OSPTYPEFOR_SPECIALIZATION(gdt::vec4ui, OSP_VEC4UI);
OSPTYPEFOR_SPECIALIZATION(gdt::vec2l, OSP_VEC2L);
OSPTYPEFOR_SPECIALIZATION(gdt::vec3l, OSP_VEC3L);
OSPTYPEFOR_SPECIALIZATION(gdt::vec4l, OSP_VEC4L);
OSPTYPEFOR_SPECIALIZATION(gdt::vec2ul, OSP_VEC2UL);
OSPTYPEFOR_SPECIALIZATION(gdt::vec3ul, OSP_VEC3UL);
OSPTYPEFOR_SPECIALIZATION(gdt::vec4ul, OSP_VEC4UL);
OSPTYPEFOR_SPECIALIZATION(gdt::vec2f, OSP_VEC2F);
OSPTYPEFOR_SPECIALIZATION(gdt::vec3f, OSP_VEC3F);
OSPTYPEFOR_SPECIALIZATION(gdt::vec4f, OSP_VEC4F);
OSPTYPEFOR_SPECIALIZATION(gdt::vec2d, OSP_VEC2D);
OSPTYPEFOR_SPECIALIZATION(gdt::vec3d, OSP_VEC3D);
OSPTYPEFOR_SPECIALIZATION(gdt::vec4d, OSP_VEC4D);
OSPTYPEFOR_SPECIALIZATION(gdt::linear2f, OSP_LINEAR2F);
OSPTYPEFOR_SPECIALIZATION(gdt::linear3f, OSP_LINEAR3F);
OSPTYPEFOR_SPECIALIZATION(gdt::affine2f, OSP_AFFINE2F);
OSPTYPEFOR_SPECIALIZATION(gdt::affine3f, OSP_AFFINE3F);
} // namespace ospray

namespace ovr::ospray {

// ------------------------------------------------------------------
// ------------------------------------------------------------------

namespace {

template<typename Type>
void
ospSetVectorAsData(OSPObject o, const char* n, OSPDataType type, const std::vector<Type>& array)
{
  if (!array.empty()) {
    OSPData shared = ospNewSharedData(array.data(), type, array.size());
    OSPData data = ospNewData(type, array.size());
    ospCopyData(shared, data);
    ospSetParam(o, n, OSP_DATA, &data);
    ospRelease(shared);
    ospRelease(data);
  }
}

template<typename Type>
OSPData
ospNewData2D(OSPDataType type, const std::vector<Type>& array)
{
  OSPData shared = ospNewSharedData(array.data(), type, array.size());
  OSPData data = ospNewData(type, array.size());
  ospCopyData(shared, data);
  return data;
}

inline OSPDataType
to_ospray_data_type(ValueType type)
{
  switch (type) {
  case VALUE_TYPE_UINT8: return OSP_UCHAR;
  case VALUE_TYPE_INT8: return OSP_CHAR;
  case VALUE_TYPE_UINT16: return OSP_USHORT;
  case VALUE_TYPE_INT16: return OSP_SHORT;
  case VALUE_TYPE_UINT32: return OSP_UINT;
  case VALUE_TYPE_INT32: return OSP_INT;
  case VALUE_TYPE_FLOAT: return OSP_FLOAT;
  case VALUE_TYPE_DOUBLE: return OSP_DOUBLE;

  case VALUE_TYPE_FLOAT2: return OSP_VEC2F;
  case VALUE_TYPE_FLOAT3: return OSP_VEC3F;
  case VALUE_TYPE_FLOAT4: return OSP_VEC4F;

  default: throw std::runtime_error("unknown type encountered");
  }
}

inline OSPData
create_ospray_array1d_scalar(array_1d_scalar_t input)
{
  return ospNewSharedData1D(input->data(), to_ospray_data_type(input->type), input->dims.v);
}

inline OSPData
create_ospray_array1d_float3(array_1d_float4_t input)
{
  if (input->type != value_type<vec4f>())
    throw std::runtime_error("type mismatch");

  return ospNewSharedData1DStride(input->data(), OSP_VEC3F, input->dims.v, sizeof(vec4f));
}

inline OSPData
create_ospray_array3d_scalar(array_3d_scalar_t input)
{
  return ospNewSharedData3D(input->data(), to_ospray_data_type(input->type), /**/
                            input->dims.x, input->dims.y, input->dims.z);
}

} // namespace

// ------------------------------------------------------------------
// ------------------------------------------------------------------

OSPVolume
DeviceOSPRay::Impl::create_ospray_volume(scene::Volume::VolumeStructuredRegular handler)
{
  OSPVolume volume = ospNewVolume("structuredRegular");
  ospSetParam(volume, "gridOrigin", OSP_VEC3F, &handler.grid_origin);
  ospSetParam(volume, "gridSpacing", OSP_VEC3F, &handler.grid_spacing);
  ospSetObject(volume, "data", create_ospray_array3d_scalar(handler.data));
  ospSetInt(volume, "cellCentered", false);
  ospCommit(volume);
  return volume;
}

OSPVolume
DeviceOSPRay::Impl::create_ospray_volume(scene::Volume handler)
{
  switch (handler.type) {
  case scene::Volume::STRUCTURED_REGULAR_VOLUME: return create_ospray_volume(handler.structured_regular);
  default: throw std::runtime_error("unknown volume type");
  }
}

OSPTransferFunction
DeviceOSPRay::Impl::create_ospray_transfer_function(scene::TransferFunction handler)
{
  OSPTransferFunction tfn = ospNewTransferFunction("piecewiseLinear");
  ospSetParam(tfn, "valueRange", OSP_VEC2F, &handler.value_range);
  ospSetObject(tfn, "color", create_ospray_array1d_float3(handler.color));
  ospSetObject(tfn, "opacity", create_ospray_array1d_scalar(handler.opacity));
  ospCommit(tfn);
  return tfn;
}

OSPGeometry
DeviceOSPRay::Impl::create_ospray_geometry(scene::Geometry::GeometryTriangles handler)
{
  OSPGeometry mesh = ospNewGeometry("mesh");
  // TODO finish the implementation //
  ospCommit(mesh);
  return mesh;
}

OSPGeometry
DeviceOSPRay::Impl::create_ospray_geometry(scene::Geometry handler)
{
  switch (handler.type) {
  case scene::Geometry::TRIANGLES_GEOMETRY: return create_ospray_geometry(handler.triangles);
  default: throw std::runtime_error("unknown geometry type");
  }
}

OSPVolumetricModel
DeviceOSPRay::Impl::create_ospray_volumetric_model(scene::Model::VolumetricModel handler)
{
  auto volume = create_ospray_volume(handler.volume);
  auto tfn = create_ospray_transfer_function(handler.transfer_function);

  OSPVolumetricModel model = ospNewVolumetricModel(volume);
  ospSetObject(model, "transferFunction", tfn);
  ospSetFloat(model, "gradientShadingScale", 1.f);
  ospCommit(model);
  ospRelease(volume);

  ospray.tfns.push_back(tfn);

  return model;
}

OSPGeometricModel
DeviceOSPRay::Impl::create_ospray_geometric_model(scene::Model::GeometricModel handler)
{
  auto geometry = create_ospray_geometry(handler.geometry);

  OSPGeometricModel model = ospNewGeometricModel(geometry);
  ospCommit(model);
  ospRelease(geometry);
  return model;
}

OSPInstance
DeviceOSPRay::Impl::create_ospray_instance(scene::Instance handler)
{
  std::vector<OSPVolumetricModel> volume_group;
  std::vector<OSPGeometricModel> geometry_group;
  for (auto m : handler.models) {
    switch (m.type) {
    case scene::Model::GEOMETRIC_MODEL:
      geometry_group.push_back(create_ospray_geometric_model(m.geometry_model));
      break;
    case scene::Model::VOLUMETRIC_MODEL: 
      volume_group.push_back(create_ospray_volumetric_model(m.volume_model)); 
      break;
    default: throw std::runtime_error("unknown volume type");
    }
  }

  OSPGroup group = ospNewGroup();
  ospSetVectorAsData(group, "geometry", OSP_GEOMETRIC_MODEL, geometry_group);
  ospSetVectorAsData(group, "volume", OSP_VOLUMETRIC_MODEL, volume_group);
  ospCommit(group);

  for (auto m : geometry_group) {
    ospRelease(m);
  }
  for (auto m : volume_group) {
    ospRelease(m);
  }

  // put the group into an instance (give the group a world transform)
  OSPInstance instance = ospNewInstance(group);
  ospSetParam(instance, "transform", OSP_AFFINE3F, &handler.transform);
  ospSetParam(instance, "xfm", OSP_AFFINE3F, &handler.transform);
  ospCommit(instance);
  ospRelease(group);

  return instance;
}

DeviceOSPRay::Impl::~Impl()
{
  for (auto tfn : ospray.tfns) {
    ospRelease(tfn);
  }

  if (ospray.camera) ospRelease(ospray.camera);
  ospRelease(ospray.renderer);
  ospRelease(ospray.world);

  if (ospray.framebuffer) {
    ospUnmapFrameBuffer(framebuffer_rgba_ptr, ospray.framebuffer);
    ospRelease(ospray.framebuffer);
  }

  ospShutdown();
}

void
DeviceOSPRay::Impl::init(int argc, const char** argv, DeviceOSPRay* p)
{
  if (parent) { 
    if (parent != p) throw std::runtime_error("[ospray] a different parent is provided");
  } 
  else {
    parent = p;

    OSPError init_error = ospInit(&argc, argv);
    if (init_error != OSP_NO_ERROR)
      throw std::runtime_error("OSPRay not initialized correctly!");

#if 1
    OSPDevice device = ospGetCurrentDevice();
    if (!device)
      throw std::runtime_error("OSPRay device could not be fetched!");

    ospDeviceSetErrorCallback(
      device,
      [](void*, OSPError error, const char* what) {
        std::cerr << "OSPRay error: " << what << std::endl;
        std::runtime_error(std::string("OSPRay error: ") + what);
      },
      nullptr);
    ospDeviceSetStatusCallback(
      device, [](void*, const char* msg) { std::cout << msg; }, nullptr);

    bool warnAsErrors = true;
    auto logLevel = OSP_LOG_WARNING;
    ospDeviceSetParam(device, "warnAsError", OSP_BOOL, &warnAsErrors);
    ospDeviceSetParam(device, "logLevel", OSP_INT, &logLevel);
    ospDeviceCommit(device);
    ospDeviceRelease(device);
#endif

    ospray.world = ospNewWorld();
  }

  build_scene();

  commit_framebuffer();
  commit_renderer();
  commit_camera();
  commit_framebuffer();
}

void
DeviceOSPRay::Impl::commit_renderer()
{
  auto& scene = parent->current_scene;

  if (parent->params.path_tracing.update() || !ospray.renderer) {
    if (ospray.renderer) ospRelease(ospray.renderer);

    if (parent->params.path_tracing.get()) {
      ospray.renderer = ospNewRenderer("pathtracer");
      ospSetInt(ospray.renderer, "roulettePathLength", scene.roulette_path_length);
      ospSetInt(ospray.renderer, "maxPathLength", scene.max_path_length);
    }
    else {
      ospray.renderer = ospNewRenderer("scivis");
      ospSetFloat(ospray.renderer, "volumeSamplingRate", scene.volume_sampling_rate);
      ospSetInt(ospray.renderer, "aoSamples", scene.ao_samples);
      // ospSetInt(ospray.renderer, "aoSamples", 1);
      // ospSetBool(ospray.renderer, "shadows", false);
    }
    ospSetInt(ospray.renderer, "pixelSamples", scene.spp);
    ospSetFloat(ospray.renderer, "backgroundColor", 0.0f); // white, transparent
    ospCommit(ospray.renderer);

    framebuffer_should_reset_accum = true;
  }

  if (parent->params.sample_per_pixel.update()) {
    scene.spp = parent->params.sample_per_pixel.get();

    ospSetInt(ospray.renderer, "pixelSamples", scene.spp);
    ospCommit(ospray.renderer);

    framebuffer_should_reset_accum = true;
  }

  if (parent->params.volume_sampling_rate.update()) {
    scene.volume_sampling_rate = parent->params.volume_sampling_rate.get();

    ospSetFloat(ospray.renderer, "volumeSamplingRate", scene.volume_sampling_rate);
    ospCommit(ospray.renderer);

    framebuffer_should_reset_accum = true;
  }

}

void
DeviceOSPRay::Impl::commit_framebuffer()
{
  bool recreate = !ospray.framebuffer;

  if (parent->params.fbsize.update()) {
    framebuffer_size_latest = parent->params.fbsize.ref();
    camera_should_update_aspect_ratio = true;
    recreate = true;
  }

  if (parent->params.frame_accumulation.update()) {
    if (parent->params.frame_accumulation.ref()) framebuffer_channels |= OSP_FB_ACCUM;
    else framebuffer_channels &= ~OSP_FB_ACCUM;
    recreate = true;
  }

  if (recreate) {
    if (ospray.framebuffer) {
      ospUnmapFrameBuffer(framebuffer_rgba_ptr, ospray.framebuffer);
      ospRelease(ospray.framebuffer);
    }

    ospray.framebuffer = ospNewFrameBuffer(framebuffer_size_latest.x, framebuffer_size_latest.y, OSP_FB_RGBA32F, framebuffer_channels);

    framebuffer_rgba_ptr = ospMapFrameBuffer(ospray.framebuffer, OSP_FB_COLOR);
    framebuffer_should_reset_accum = true;

    // update sparse sampling buffer
    sparse_sampling_xs_ys.resize(framebuffer_size_latest.long_product() * 2ULL);
    ospray.sparse_samples = ospNewSharedData(sparse_sampling_xs_ys.data(), OSP_VEC2I, sparse_sampling_xs_ys.size()/2ULL);
  }

  if (framebuffer_should_reset_accum && ospray.framebuffer) {
    ospResetAccumulation(ospray.framebuffer);
  }
  framebuffer_should_reset_accum = false;
}

void
DeviceOSPRay::Impl::commit_camera()
{
  if (parent->params.camera.update() || !ospray.camera) {

    const Camera& camera = parent->params.camera.ref();
    // std::cout << "camera update" << std::endl;
    // std::cout << "  from: " << camera.from << std::endl;
    // std::cout << "  at:   " << camera.at << std::endl;
    // std::cout << "  up:   " << camera.up << std::endl;

    if (ospray.camera) ospRelease(ospray.camera);

    if (camera.type == Camera::PERSPECTIVE) {
      ospray.camera = ospNewCamera("perspective");
      ospSetFloat(ospray.camera, "fovy", camera.perspective.fovy);
    }
    else {
      ospray.camera = ospNewCamera("orthographic");
      ospSetFloat(ospray.camera, "height", camera.orthographic.height);
    }

    const vec3f dir = camera.at - camera.from;
    ospSetFloat(ospray.camera, "aspect", framebuffer_size_latest.x / (float)framebuffer_size_latest.y);
    ospSetParam(ospray.camera, "position", OSP_VEC3F, &camera.from);
    ospSetParam(ospray.camera, "direction", OSP_VEC3F, &dir);
    ospSetParam(ospray.camera, "up", OSP_VEC3F, &camera.up);
    ospCommit(ospray.camera); // commit each object to indicate modifications are done

    framebuffer_should_reset_accum = true;
  }

  if (camera_should_update_aspect_ratio) {
    ospSetFloat(ospray.camera, "aspect", framebuffer_size_latest.x / (float)framebuffer_size_latest.y);
    ospCommit(ospray.camera);

    framebuffer_should_reset_accum = true;
  }

  camera_should_update_aspect_ratio = false;
}

void
DeviceOSPRay::Impl::commit_transfer_function()
{
  if (parent->params.tfn.update()) {
    framebuffer_should_reset_accum = true;

    const auto& tfn = parent->params.tfn.ref();
    std::vector<vec3f> tfn_color(tfn.tfn_colors.size() / 3);
    for (int i = 0; i < tfn_color.size(); ++i) {
      tfn_color[i].x = tfn.tfn_colors[3 * i + 0];
      tfn_color[i].y = tfn.tfn_colors[3 * i + 1];
      tfn_color[i].z = tfn.tfn_colors[3 * i + 2];
    }
    std::vector<float> tfn_opacity(tfn.tfn_alphas.size() / 2);
    for (int i = 0; i < tfn_opacity.size(); ++i) {
      tfn_opacity[i] = tfn.tfn_alphas[2 * i + 1];
    }
    vec2f tfn_range = tfn.tfn_value_range;

    if (tfn_color.empty())
      return;
    if (tfn_opacity.empty())
      return;

    OSPData color_data = ospNewData2D(OSP_VEC3F, tfn_color);
    OSPData opacity_data = ospNewData2D(OSP_FLOAT, tfn_opacity);

    for (auto& tfn : ospray.tfns) {
      ospSetParam(tfn, "valueRange", OSP_VEC2F, &tfn_range);
      ospSetObject(tfn, "color", color_data);
      ospSetObject(tfn, "opacity", opacity_data);
      ospCommit(tfn);
    }
  }
}

void
DeviceOSPRay::Impl::build_scene()
{
  // put the instance in the world
  const auto& scene = parent->current_scene;

  // create all ospray instances
  std::vector<OSPInstance> instances;
  for (auto i : scene.instances) {
    instances.push_back(create_ospray_instance(i));
  }
  ospSetVectorAsData(ospray.world, "instance", OSP_INSTANCE, instances);

  // create ospray light
  std::vector<OSPLight> lights;
  for (auto& l : scene.lights) {
    OSPLight light;

    if (l.type == scene::Light::AMBIENT) {
      light = ospNewLight("ambient");
    } 
    else if (l.type == scene::Light::DIRECTIONAL) {
      light = ospNewLight("distant");
      ospSetVec3f(light, "direction", l.directional.direction.x, l.directional.direction.y, l.directional.direction.z);
    }
    else {
      throw std::runtime_error("OSPray: unknown light type");
    }

    ospSetFloat(light, "intensity", l.intensity);
    ospSetVec3f(light, "color", l.color.x, l.color.y, l.color.z);

    ospCommit(light);
    lights.push_back(light);
  }

  // create some default lights if there is no scene light
  auto sun1 = ospNewLight("sunSky");  
  ospSetFloat(sun1, "intensity", 2.0f);
  ospSetVec3f(sun1, "color", 2.6f, 2.5f, 2.3f);
  ospSetVec3f(sun1, "direction", 0, -1, 0);
  ospCommit(sun1);
  {
    lights.push_back(sun1);
  }

  auto sun2 = ospNewLight("sunSky");  
  ospSetFloat(sun2, "intensity", 2.0f);
  ospSetVec3f(sun2, "color", 2.6f, 2.5f, 2.3f);
  ospSetVec3f(sun2, "direction", 0, 1, 0);
  ospCommit(sun2);
  {
    lights.push_back(sun2);
  }

  auto ambLight = ospNewLight("ambient");
  ospSetFloat(ambLight, "intensity", 1.0f);
  ospSetVec3f(ambLight, "color", 1.f, 1.f, 1.f);
  ospCommit(ambLight);
  {
    lights.push_back(ambLight);
  }

  ospSetVectorAsData(ospray.world, "light", OSP_LIGHT, lights);
  ospCommit(ospray.world);

  // // release instances
  // for (auto& i : instances) ospRelease(i);
  // // release lights!
  // for (auto& l : lights) ospRelease(l);
}

void
DeviceOSPRay::Impl::swap()
{
  framebuffer_index = (framebuffer_index + 1) % 2;
}

void
DeviceOSPRay::Impl::commit()
{
  if (parent->params.focus_center.update()) {
    framebuffer_should_reset_accum = true;
  }

  if (parent->params.focus_scale.update()) {
    framebuffer_should_reset_accum = true;
  }

  if (parent->params.base_noise.update()) {
    framebuffer_should_reset_accum = true;
  }

  const auto& scene = parent->current_scene;
  commit_transfer_function();

  commit_framebuffer();

  commit_renderer();
  
  if (parent->params.sparse_sampling.update()) {
    ospSetBool(ospray.renderer, "sparseSampling", parent->params.sparse_sampling.ref());
    ospSetVec2i(ospray.renderer, "sparseSamplingSize", framebuffer_size_latest.x, framebuffer_size_latest.y);
    ospSetObject(ospray.renderer, "sparseSamplingBuffer", ospray.sparse_samples);
    ospCommit(ospray.renderer);
    framebuffer_should_reset_accum = true;
  }

  commit_camera();
  commit_framebuffer();
}

void
DeviceOSPRay::Impl::render()
{
  frame_index++;

  if (parent->params.sparse_sampling.ref()) {
    const int64_t launch_size = generate_sparse_sampling_mask_h(
      sparse_sampling_xs_ys.data(), 
      frame_index, framebuffer_size_latest,
      parent->params.focus_center.ref(), 
      parent->params.focus_scale.ref(),
      parent->params.base_noise.ref()
    ) / 2ULL;

    // TODO throw if launch_size is too large.

    OSPFrameBuffer fb = ospNewFrameBuffer((int)launch_size, 1, OSP_FB_RGBA32F, OSP_FB_COLOR);

    parent->variance = ospRenderFrameBlocking(fb, ospray.renderer, ospray.camera, ospray.world);

    // split rendered data to the actual framebuffer
    memset((void*)framebuffer_rgba_ptr, 0, sizeof(vec4f) * framebuffer_size_latest.long_product());
    const vec4f* data = (vec4f*)ospMapFrameBuffer(fb, OSP_FB_COLOR);
    tbb::parallel_for(int64_t(0), launch_size, [&] (int64_t i) {
      int x = sparse_sampling_xs_ys[2 * i + 0];
      int y = sparse_sampling_xs_ys[2 * i + 1];
      ((vec4f*)framebuffer_rgba_ptr)[y * framebuffer_size_latest.x + x] = data[i];
    });
    ospUnmapFrameBuffer(data, fb);

    ospRelease(fb);
  }
  else {
    parent->variance = ospRenderFrameBlocking(ospray.framebuffer, ospray.renderer, ospray.camera, ospray.world);
  }
}

void
DeviceOSPRay::Impl::mapframe(FrameBufferData* fb)
{
  const size_t num_bytes = framebuffer_size_latest.long_product();
  fb->rgba->set_data((void*)framebuffer_rgba_ptr, num_bytes * sizeof(vec4f), CrossDeviceBuffer::DEVICE_CPU);
}

} // namespace ovr::ospray
