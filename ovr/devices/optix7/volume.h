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
#ifndef OVR_OPTIX7_VOLUME_H
#define OVR_OPTIX7_VOLUME_H

#include "optix7_common.h"
#include "array.h"

#include "accel/spatial_partition.h"

#include <array>
#include <vector>

namespace ovr {
namespace optix7 {

// ------------------------------------------------------------------
// Volume Definition
// ------------------------------------------------------------------

// x << y = x * (2 ^ y)
__device__ constexpr int MACROCELL_SIZE_MIP = 4;
__device__ constexpr int MACROCELL_SIZE = 1 << MACROCELL_SIZE_MIP;

struct DeviceStructuredRegularVolume {
  Array3DScalarOptix7 volume;
  float base;
  float step;
  DeviceTransferFunction tfn;

  DeviceSpacePartiton_SingleMC sp;

  // float* __restrict__ macrocell_max_opacity{ nullptr };
  // vec2f* __restrict__ macrocell_value_range{ nullptr };
  // vec3i macrocell_dims;
  // vec3f macrocell_spacings;
  // vec3f macrocell_spacings_rcp;

  int use_dda = 1; // 0 = Not using DDA; 1 = Using Single Layer DDA; 2 = Using Multiple Layer DDA
  // int dda_layers = 3;
  // float* __restrict__ list_of_macrocell_max_opacity{ nullptr };
  // vec2f* __restrict__ list_of_macrocell_value_range{ nullptr };
  // vec3i* list_of_macrocell_dims{ nullptr };
  // size_t* list_of_mip_size_before_current_mip_level{ nullptr };
  // vec3f* list_of_macrocell_spacings;
  // vec3f* list_of_macrocell_spacings_rcp;
};

// ------------------------------------------------------------------
//
// Host Functions
//
// ------------------------------------------------------------------
#if defined(__cplusplus)

struct InstantiableGeometry {
  affine3f matrix;

  /*! compute 3x4 transformation matrix */
  void transform(float transform[12]) const;
};

struct AabbGeometry {
private:
  // the AABBs for procedural geometries
  OptixAabb aabb{ 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };
  CUDABuffer aabbBuffer;
  CUDABuffer asBuffer; // buffer that keeps the (final, compacted) accel structure

public:
  OptixTraversableHandle blas;

  void buildas(OptixDeviceContext optixContext, cudaStream_t stream = 0);

  virtual void* get_sbt_pointer(cudaStream_t stream) = 0;
};

struct StructuredRegularVolume
  : protected HasSbtEquivalent<DeviceStructuredRegularVolume>
  , public AabbGeometry
  , public InstantiableGeometry {
public:
  SpacePartiton_SingleMC sp;

  // float* m_macrocell_max_opacity_inference;
  // CUDABuffer m_macrocell_max_opacity_inferencePtr_device_buffer;
  // vec2f* m_macrocell_inference;
  // CUDABuffer m_macrocell_inferencePtr_device_buffer;
  // vec3i m_macrocell_dims;
  // vec3f m_macrocell_spacings;

  // float* list_m_macrocell_max_opacity_inference;
  // CUDABuffer list_of_m_macrocell_max_opacity_inferencePtr_device_buffer;

  // vec2f* list_m_macrocell_inference;
  // CUDABuffer list_of_m_macrocell_inferencePtr_device_buffer;

  // std::vector<vec3i> list_of_m_macrocell_dims;
  // CUDABuffer list_of_m_macrocell_dims_device_buffer;
  
  // std::vector<vec3f> list_of_m_macrocell_spacings;
  // CUDABuffer list_of_m_macrocell_spacings_device_buffer;
  
  // std::vector<vec3f> list_of_m_macrocell_spacings_rcp;
  // CUDABuffer list_of_m_macrocell_spacings_rcp_device_buffer;

  // CUDABuffer list_of_mip_size_before_current_mip_level_device_buffer;

  std::vector<vec4f> tfn_colors_data;
  std::vector<float> tfn_alphas_data;
  vec2f original_value_range;

  float base = 1.f;
  float rate = 1.f;

  void commit(cudaStream_t stream);
  void* get_sbt_pointer(cudaStream_t stream) override;

  void set_sampling_rate(float r, float b = 0.f);

  void load_from_array3d_scalar(array_3d_scalar_t array, float data_value_min = 1, float data_value_max = -1);

  void set_transfer_function(Array1DFloat4Optix7 c, Array1DScalarOptix7 a, vec2f r);
  void set_transfer_function(array_1d_float4_t c, array_1d_scalar_t a, vec2f r);
  void set_transfer_function(const std::vector<float>& c, const std::vector<float>& o, const vec2f& r);
  void set_value_range(float data_value_min, float data_value_max);

  // void set_macrocell(vec3i dims, vec3f spacings, vec2f* d_value_range, float* d_max_opacity);
  // void set_macrocell(vec3i* dims, vec3f* spacings, vec3f* spacings_rcp, vec2f* d_value_range, float* d_max_opacity, size_t* d_mip_size);

  // void* get_device_volume_ptr(int use_dda);
  // void setup_multi_level_dda(vec3i dims);
};

#endif // #if defined(__cplusplus)

} // namespace optix7
} // namespace ovr
#endif // OVR_OPTIX7_VOLUME_H
