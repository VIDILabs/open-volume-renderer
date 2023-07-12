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
#ifndef OVR_OPTIX7_PARAMS_H
#define OVR_OPTIX7_PARAMS_H

#include "optix7_common.h"
#include "volume.h"

#include "ovr/scene.h"

#include <cuda_misc.h>

#if defined(__cplusplus)
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#endif // defined(__cplusplus)

namespace ovr {
namespace optix7 {

#define OVR_OPTIX7_MASKING_VIA_STREAM_COMPACTION
// #define OVR_OPTIX7_MASKING_VIA_DIRECT_SAMPLING

// #define OVR_OPTIX7_JITTER_RAYS

// ------------------------------------------------------------------
//
// Shared Definitions
//
// ------------------------------------------------------------------

#define VISIBILITY_VOLUME   (1ULL << 0)
#define VISIBILITY_GEOMETRY (1ULL << 1)

struct LaunchParams { // shared global data
  struct DeviceFrameBuffer {
    vec4f* rgba;
    vec3f* grad;
    vec2i size;
    vec2f size_rcp;
  } frame;
  vec4f* frame_accum_rgba;
  vec3f* frame_accum_grad;
  int32_t frame_index{ 0 };

  struct DeviceCamera {
    vec3f position;
    vec3f direction;
    vec3f horizontal;
    vec3f vertical;
  } camera, last_camera;

  OptixTraversableHandle traversable{};

  bool enable_path_tracing{ false };
  bool enable_sparse_sampling{ false };
  bool enable_frame_accumulation{ false };

  vec3f light_directional_pos{ -907.108f, 2205.875f, -400.0267f };
  float light_ambient_intensity{ 1.f };

  float base_noise{ 0.1f };
  vec2f focus_center{ 0.5f, 0.5f };
  float focus_scale{ 0.2f };

  int32_t max_num_scatters{ 24 };
  int32_t sample_per_pixel{ 1 };

  struct {
#ifdef OVR_OPTIX7_MASKING_VIA_DIRECT_SAMPLING
    const int downsample_factor = 4;
    float* dist_uniform{ nullptr };
    float* dist_logistic{ nullptr };
#endif
#ifdef OVR_OPTIX7_MASKING_VIA_STREAM_COMPACTION
    int32_t* xs_and_ys{ nullptr };
#endif
  } sparse_sampling;
};

// ------------------------------------------------------------------
//
//
//
// ------------------------------------------------------------------

#if defined(__cplusplus)
using FrameBuffer = DoubleBufferObject<vec4f, vec3f>;
#endif // define(__cplusplus)

// void update_inference_macrocell(cudaStream_t stream, DeviceStructuredRegularVolume& self);
// void update_inference_macrocell(cudaStream_t stream, DeviceStructuredRegularVolume& self, std::vector<vec3i> list_of_m_macrocell_dims);
// void calculate_macrocell_value_range(vec3i& m_macrocell_dims,
//                                      CUDABuffer m_macrocell_inferencePtr_device_buffer,
//                                      vec3i& volume_dims,
//                                      DeviceStructuredRegularVolume* self,
//                                      int macrocell_size);
// void update_macrocell_and_calculate_value_range(vec3i& m_macrocell_dims,
//                                                 vec3f& m_macrocell_spacings,
//                                                 CUDABuffer& m_macrocell_inferencePtr_device_buffer,
//                                                 CUDABuffer& m_macrocell_max_opacity_inferencePtr_device_buffer,
//                                                 DeviceStructuredRegularVolume* self,
//                                                 vec3i dims);
// void handle_list_of_update_macrocell_and_calculate_value_range(vec3i& m_macrocell_dims,
//                                                               vec3f& m_macrocell_spacings,
//                                                               CUDABuffer& m_macrocell_inferencePtr_device_buffer,
//                                                               CUDABuffer& m_macrocell_max_opacity_inferencePtr_device_buffer,
//                                                               DeviceStructuredRegularVolume* self,
//                                                               vec3i dims,
//                                                               int macrocell_per_size);

} // namespace optix7
} // namespace ovr
#endif // OVR_OPTIX7_PARAMS_H
