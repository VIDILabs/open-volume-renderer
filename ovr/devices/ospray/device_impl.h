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

#pragma once
#ifndef OVR_OSPRAY_DEVICE_IMPL_H
#define OVR_OSPRAY_DEVICE_IMPL_H

#include "device.h"

#include <ospray/ospray.h>

#include <limits>
#include <mutex>

namespace ovr::ospray {

// In OVR, we consider volumes as special textures
struct OSPTexOrVol {
  OSPVolume vol = nullptr;
  OSPTexture tex = nullptr;
};

struct DeviceOSPRay::Impl {
  DeviceOSPRay* parent{ nullptr };

public:
  ~Impl();

  void init(int argc, const char** argv, DeviceOSPRay* parent);
  void swap();
  void commit();
  void render();
  void mapframe(FrameBufferData*);

  // ------------------------------------------------------------------
  // internal helper functions
  // ------------------------------------------------------------------
protected:
  void commit_camera();
  void commit_transfer_function();
  void commit_renderer();
  void commit_framebuffer();

  void build_scene();

  // scene building functions
  OSPVolume create_ospray_volume(scene::Volume::VolumeStructuredRegular handler);
  OSPVolume create_ospray_volume(scene::Volume handler);

  OSPGeometry create_ospray_geometry(scene::Geometry::GeometryTriangles handler);
  OSPGeometry create_ospray_geometry(scene::Geometry::GeometryIsosurfaces handler);
  OSPGeometry create_ospray_geometry(scene::Geometry handler);
  
  OSPTexture create_ospray_texture(scene::Texture::TransferFunctionTexture handler);
  
  OSPMaterial create_ospray_material(scene::Material::ObjMaterial handler);
  OSPMaterial create_ospray_material(scene::Material handler);

  OSPTransferFunction create_ospray_transfer_function(scene::TransferFunction handler);
  OSPVolumetricModel create_ospray_volumetric_model(scene::Model::VolumetricModel handler);
  OSPGeometricModel create_ospray_geometric_model(scene::Model::GeometricModel handler);
  OSPInstance create_ospray_instance(scene::Instance handler);

  // our extension
  OSPTexOrVol create_ospray_texture(scene::Texture handler);

protected:
  struct {
    OSPCamera camera{ 0 };
    OSPFrameBuffer framebuffer{ 0 };
    OSPWorld world{ 0 };
    OSPRenderer renderer{ 0 };
    OSPImageOperation tonemapper { 0 };
    std::vector<OSPTransferFunction> tfns;
    std::vector<OSPTexOrVol> texorvols;
    std::vector<OSPMaterial> materials;
    OSPData sparse_samples;

    OSPVolume get_volume(int32_t idx) { 
      if (idx < 0 || idx >= texorvols.size()) {
        throw std::runtime_error("invalid texture index: " + std::to_string(idx) + " / " + std::to_string(texorvols.size()));
      }
      auto ret = texorvols[idx];
      if (ret.vol == nullptr) {
        throw std::runtime_error("texture " + std::to_string(idx) + " is not a volume texture");
      }
      return ret.vol;
    }

    OSPTexture get_texture(int32_t idx) { 
      if (idx < 0 || idx >= texorvols.size()) {
        throw std::runtime_error("invalid texture index: " + std::to_string(idx) + " / " + std::to_string(texorvols.size()));
      }
      auto ret = texorvols[idx];
      if (ret.tex == nullptr) {
        throw std::runtime_error("texture " + std::to_string(idx) + " is not a texture");
      }
      return ret.tex;
    }
  } ospray;

  uint32_t framebuffer_channels = OSP_FB_COLOR | OSP_FB_VARIANCE /*| OSP_FB_NORMAL*/ /*| OSP_FB_ACCUM*/;
  vec2i framebuffer_size_latest{ 1 /* give a non zero initial size */};
  // int  framebuffer_index = 0;
  bool framebuffer_should_reset_accum{ true };
  const void* framebuffer_rgba_ptr{ 0 };
  const void* framebuffer_grad_ptr{ 0 }; // TODO gradient layer is not calculated //

  bool camera_should_update_aspect_ratio{ 0 };

  std::vector<int32_t> sparse_sampling_xs_ys;
  int frame_index{ 0 };
};

} // namespace ovr::ospray
#endif // OVR_OSPRAY_DEVICE_IMPL_H
