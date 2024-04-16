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

//. ======================================================================== //
//. Copyright 2018-2019 Ingo Wald                                            //
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

#include "renderer_macro.h"
#include "scene.h"

#include <cross_device_buffer.h>
#include <vidi_transactional_value.h>

#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#define OVR_FRAMEBUFFERDATA_REQUIRE_SIZE 1

namespace ovr {

using vidi::TransactionalValue;
using scene::Camera;

inline int
count_tfn(const scene::Scene& scene, scene::TransferFunction& scene_tfn) 
{
  int count = 0;
  // There are two places transfer function can be stored:
  // 1) in a volume model
  for (const auto& instance : scene.instances) {
    for (const auto& model : instance.models) {
      if (model.type == scene::Model::VOLUMETRIC_MODEL) {
        scene_tfn = model.volume_model.transfer_function;
        count++;
      }
    }
  }
  // 2) in a texture
  for (const auto& texture : scene.textures) {
    if (texture.type == scene::Texture::TRANSFER_FUNCTION_TEXTURE) {
      scene_tfn = texture.transfer_function.transfer_function;
      count++;
    }
  }
  return count;
}

/*! a sample OptiX-7 renderer that demonstrates how to set up
    context, module, programs, pipeline, SBT, etc, and perform a
    valid launch that renders some pixel (using a simple test
    pattern, in this case */
struct MainRenderer {
  // ------------------------------------------------------------------
  // publicly accessible interface
  // ------------------------------------------------------------------
public:
  double render_time;

  struct FrameBufferData {
    vec2i size{ 0 };
    std::shared_ptr<CrossDeviceBuffer> rgba;
    std::shared_ptr<CrossDeviceBuffer> grad;

    FrameBufferData() {
      rgba = std::make_shared<CrossDeviceBuffer>();
      grad = std::make_shared<CrossDeviceBuffer>();
    }
  };

  struct TransferFunctionData {
    std::vector<float> tfn_colors;
    std::vector<float> tfn_alphas;
    vec2f tfn_value_range = { 1, -1 };
  };

  Scene current_scene;

  virtual ~MainRenderer() {}

  /*! unsafe: constructor - performs all setup */
  void init(int argc, const char** argv, Scene scene, Camera camera);

  /*! unsafe: called in worker thread */
  virtual void swap() = 0;
  virtual void commit() = 0;
  virtual void render() = 0;
  virtual void mapframe(FrameBufferData*) = 0;

  /*! unsafe: getters */
  math::vec2i unsafe_get_fbsize() const { return params.fbsize.get(); }
  float unsafe_get_variance() const { return variance; }
  const TransferFunctionData& unsafe_get_tfn() const { return params.tfn.ref(); }

  /*! thread safe: called in GUI thread */
  virtual void ui() {}

  /*! thread safe: setters */
  void set_fbsize(const math::vec2i& fbsize) { params.fbsize = fbsize; }
  void set_camera(const Camera& camera) { params.camera = camera; }
  void set_camera(math::vec3f from, math::vec3f at, math::vec3f up) { set_camera(Camera{ from, at, up }); }
  void set_transfer_function(const std::vector<float>& c, const std::vector<float>& o, const math::vec2f& r) {
    params.tfn.assign([&](TransferFunctionData& d) {
      d.tfn_colors = c; d.tfn_alphas = o; d.tfn_value_range = r;
    });
  }
  void set_sample_per_pixel(int spp) { params.sample_per_pixel = spp; }
  void set_path_tracing(bool path_tracing) { params.path_tracing = path_tracing; }
  void set_frame_accumulation(bool frame_accumulation) { params.frame_accumulation = frame_accumulation; }
  void set_volume_sampling_rate(float volume_sampling_rate) { params.volume_sampling_rate = volume_sampling_rate; }
  void set_volume_density_scale(float volume_density_scale) { params.volume_density_scale = volume_density_scale; }
  void set_tonemapping(bool tonemapping) { params.tonemapping = tonemapping; }

  // setters for sparse sampling //
  void set_sparse_sampling(bool sparse_sampling) { params.sparse_sampling = sparse_sampling; }
  void set_focus(const math::vec2f& center, float scale, float base_noise) {
    params.focus_center = center; params.focus_scale = scale; params.base_noise = base_noise;
  }

protected:
  void set_scene(const Scene& scene);
  virtual void init(int argc, const char** argv) = 0;

protected:
  struct {
    TransactionalValue<Camera> camera;
    TransactionalValue<TransferFunctionData> tfn;
    TransactionalValue<vec2i> fbsize;
    TransactionalValue<int> sample_per_pixel;
    TransactionalValue<float> volume_sampling_rate;
    TransactionalValue<float> volume_density_scale;
    TransactionalValue<bool> path_tracing;
    TransactionalValue<bool> frame_accumulation;
    TransactionalValue<bool> tonemapping;
    // options for sparse sampling 
    TransactionalValue<vec2f> focus_center;
    TransactionalValue<float> focus_scale;
    TransactionalValue<float> base_noise;
    TransactionalValue<bool> sparse_sampling;
  } params;

  float variance{ std::numeric_limits<float>::infinity() };
};

inline void
MainRenderer::init(int argc, const char** argv, Scene scene, Camera camera)
{
  render_time = 0.0;
  set_scene(scene);
  set_camera(camera);
  init(argc, argv);
}

inline void
MainRenderer::set_scene(const Scene& scene)
{
  // TODO generalize to support multiple transfer functions //
  scene::TransferFunction scene_tfn;
  int count = count_tfn(scene, scene_tfn);
  if (count > 1) {
    std::cerr << "ERROR: found multiple transfer functions, they will be treated as one" << std::endl;
  }

  // TODO: find a better way to set transfer function //
  if (count > 0) {
    const float* data_o = scene_tfn.opacity->data_typed<float>();
    const size_t size_o = scene_tfn.opacity->dims.v;

    const vec4f* data_c = scene_tfn.color->data_typed<vec4f>();
    const size_t size_c = scene_tfn.color->dims.v;

    std::vector<float> tfn_colors;
    std::vector<float> tfn_alphas;
    vec2f tfn_value_range = { 1, -1 };

    for (size_t i = 0; i < size_c; ++i) {
      // float p = (float)i / (size_c - 1);
      // tfn_colors.push_back(p);
      tfn_colors.push_back(data_c[i].x);
      tfn_colors.push_back(data_c[i].y);
      tfn_colors.push_back(data_c[i].z);
    }

    for (size_t i = 0; i < size_o; ++i) {
      float p = (float)i / (size_o - 1);
      tfn_alphas.push_back(p);
      tfn_alphas.push_back(data_o[i]);
    }

    tfn_value_range = scene_tfn.value_range;

    set_transfer_function(tfn_colors, tfn_alphas, tfn_value_range);
  }

  current_scene = std::move(scene); // makes sure data will not be released while the program is running
}

} // namespace ovr

std::shared_ptr<ovr::MainRenderer>
create_renderer(std::string name);

ovr::Scene
create_scene_device(std::string filename, std::string device);
