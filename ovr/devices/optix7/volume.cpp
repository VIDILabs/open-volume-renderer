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

#include "volume.h"
#include "params.h"
namespace ovr::optix7 {

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

void
InstantiableGeometry::transform(float transform[12]) const
{
  transform[0]  = matrix.l.row0().x;
  transform[1]  = matrix.l.row0().y;
  transform[2]  = matrix.l.row0().z;
  transform[3]  = matrix.p.x;
  transform[4]  = matrix.l.row1().x;
  transform[5]  = matrix.l.row1().y;
  transform[6]  = matrix.l.row1().z;
  transform[7]  = matrix.p.y;
  transform[8]  = matrix.l.row2().x;
  transform[9]  = matrix.l.row2().y;
  transform[10] = matrix.l.row2().z;
  transform[11] = matrix.p.z;
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

void
AabbGeometry::buildas(OptixDeviceContext optixContext, cudaStream_t stream)
{
  // ==================================================================
  // aabb inputs
  // ==================================================================
  aabbBuffer.alloc_and_upload_async(&aabb, 1, stream);

  CUdeviceptr d_aabb = aabbBuffer.d_pointer();
  uint32_t f_aabb = 0;

  OptixBuildInput volumeInput = {}; // use one AABB input
  volumeInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
#if OPTIX_ABI_VERSION < 23
  auto& customPrimitiveArray = volumeInput.aabbArray;
#else
  auto& customPrimitiveArray = volumeInput.customPrimitiveArray;
#endif
  customPrimitiveArray.aabbBuffers = &d_aabb;
  customPrimitiveArray.numPrimitives = 1;
  customPrimitiveArray.strideInBytes = 0;
  customPrimitiveArray.primitiveIndexOffset = 0;
  customPrimitiveArray.flags = &f_aabb;
  customPrimitiveArray.numSbtRecords = 1;
  customPrimitiveArray.sbtIndexOffsetBuffer = 0;
  customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
  customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

  std::vector<OptixBuildInput> inputs = { volumeInput };
  blas = buildas_exec(optixContext, inputs, asBuffer);
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

void
StructuredRegularVolume::set_transfer_function(Array1DFloat4Optix7 c, Array1DScalarOptix7 a, vec2f r)
{
  self.tfn.color = c;
  self.tfn.opacity = a;

  set_value_range(r.x, r.y);

  CUDA_SYNC_CHECK();
    
  // if (self.use_dda == 1) 
  {
    // update_inference_macrocell(0, self);
    sp.compute_majorant(self.tfn);
  }
  // else if(self.use_dda == 2) {
  //   update_inference_macrocell(0, self, list_of_m_macrocell_dims); 
  // }

  CUDA_SYNC_CHECK();
}

void
StructuredRegularVolume::set_transfer_function(array_1d_float4_t c, array_1d_scalar_t a, vec2f r)
{
  set_transfer_function(CreateArray1DFloat4Optix7(c), CreateArray1DScalarOptix7(a), r);
}

void
StructuredRegularVolume::set_transfer_function(const std::vector<float>& c, const std::vector<float>& o, const vec2f& r)
{
  tfn_colors_data.resize(c.size() / 3);
  for (int i = 0; i < tfn_colors_data.size(); ++i) {
    tfn_colors_data[i].x = c[3 * i + 0];
    tfn_colors_data[i].y = c[3 * i + 1];
    tfn_colors_data[i].z = c[3 * i + 2];
    tfn_colors_data[i].w = 1.f;
  }
  tfn_alphas_data.resize(o.size() / 2);
  for (int i = 0; i < tfn_alphas_data.size(); ++i) {
    tfn_alphas_data[i] = o[2 * i + 1];
  }

  if (!tfn_colors_data.empty() && !tfn_alphas_data.empty())
    set_transfer_function(CreateArray1DFloat4Optix7(tfn_colors_data), CreateArray1DScalarOptix7(tfn_alphas_data), r);

  CUDA_SYNC_CHECK();
}

void
StructuredRegularVolume::set_value_range(float data_value_min, float data_value_max)
{
  Array3DScalarOptix7& volume = self.volume;

  if (data_value_max >= data_value_min) {
    float normalized_max = integer_normalize(data_value_max, volume.type);
    float normalized_min = integer_normalize(data_value_min, volume.type);
    // volume.upper.v = min(original_value_range.y, normalized_max);
    // volume.lower.v = max(original_value_range.x, normalized_min);
    volume.upper.v = normalized_max; // should use the transfer function value range here
    volume.lower.v = normalized_min;    
  }

  volume.scale.v = 1.f / (volume.upper.v - volume.lower.v);

  // Need calculation on max opacity
  auto r_x = max(original_value_range.x, volume.lower.v);
  auto r_y = min(original_value_range.y, volume.upper.v);

  self.tfn.value_range.y = r_y;
  self.tfn.value_range.x = r_x;
  self.tfn.range_rcp_norm = 1.f / (self.tfn.value_range.y - self.tfn.value_range.x);
}

void
StructuredRegularVolume::set_sampling_rate(float r, float b)
{
  rate = r;
  if (b > 0)
    base = b;
}

void*
StructuredRegularVolume::get_sbt_pointer(cudaStream_t stream)
{
  // if (GetSbtPtr() == nullptr)
  //   return CreateSbtPtr(stream); /* upload to GPU */
  return GetSbtPtr();
}

void
StructuredRegularVolume::commit(cudaStream_t stream)
{
  // self.alpha_adjustment = base / rate;
  self.base = base;
  self.step = 1.f / rate;
  if (GetSbtPtr() == nullptr) CreateSbtPtr(stream);
  UpdateSbtData(stream);
}

void
StructuredRegularVolume::load_from_array3d_scalar(array_3d_scalar_t array, float data_value_min, float data_value_max)
{
  Array3DScalarOptix7& output = self.volume;

  output = CreateArray3DScalarOptix7(array);
  original_value_range.x = output.lower.v;
  original_value_range.y = output.upper.v;
  std::cout << "[optix7] volume range = " << original_value_range.x << " " << original_value_range.y << std::endl;

  set_value_range(data_value_min, data_value_max);

CUDA_SYNC_CHECK();
  // compute macrocell

  // misplaced code block ...
  // auto new_tfn_value_range = v.set_value_range(scene_tfn.value_range.x, scene_tfn.value_range.y);
  // parent->params.tfn.assign([&](TransferFunctionData& d) {
  //   d.tfn_value_range = new_tfn_value_range;
  // });
  // Calculate MacroCell Value Range
  // if (parent->current_scene.use_dda == 1) 
  // if (0)
  // {
  //   // void* device_volume_ptr = v.get_device_volume_ptr(scene.use_dda);

  //   auto dims = self.volume.dims;

  //   m_macrocell_dims = div_round_up(dims, vec3i(MACROCELL_SIZE));
  //   m_macrocell_spacings = vec3f(MACROCELL_SIZE) / vec3f(dims);
    
  //   size_t macrocell_size = m_macrocell_dims.long_product();

  //   m_macrocell_inferencePtr_device_buffer.resize(macrocell_size * sizeof(vec2f));
  //   m_macrocell_max_opacity_inferencePtr_device_buffer.resize(macrocell_size * sizeof(float));

  //   calculate_macrocell_value_range(m_macrocell_dims, m_macrocell_inferencePtr_device_buffer, dims, &self, MACROCELL_SIZE);

  //   // update_macrocell_and_calculate_value_range(v.m_macrocell_dims,
  //   //                                             v.m_macrocell_spacings,
  //   //                                             v.m_macrocell_inferencePtr_device_buffer,
  //   //                                             v.m_macrocell_max_opacity_inferencePtr_device_buffer,
  //   //                                             // (DeviceStructuredRegularVolume*)device_volume_ptr,
  //   //                                             // scene_volume.data->dims
  //   //                                             );

  //   set_macrocell(
  //     m_macrocell_dims,
  //     m_macrocell_spacings,
  //     (vec2f*)m_macrocell_inferencePtr_device_buffer.d_pointer(),
  //     (float*)m_macrocell_max_opacity_inferencePtr_device_buffer.d_pointer()
  //   );
  //   CUDA_SYNC_CHECK();
  // }
  // else 
  {
    sp.allocate(self.volume.dims);

    CUDA_SYNC_CHECK();
    
    sp.compute_value_range(self.volume.dims, self.volume.data);
    self.sp = sp.self;

    // set_macrocell(
    //   sp.self.dims,
    //   sp.self.spac,
    //   (vec2f*)sp.self.value_range,
    //   (float*)sp.self.majorant
    // );

  }
CUDA_SYNC_CHECK();
  // else if (parent->current_scene.use_dda == 2) {
  //   v.get_device_volume_ptr(scene.use_dda);
  //   v.setup_multi_level_dda(scene_volume.data->dims);
  // }
}

// void 
// StructuredRegularVolume::set_macrocell(vec3i dims, vec3f spacings, vec2f* d_value_range, float* d_max_opacity)
// {
//   self.macrocell_value_range = d_value_range;
//   self.macrocell_max_opacity = d_max_opacity;

//   self.macrocell_dims = dims;
//   self.macrocell_spacings = spacings;
//   self.macrocell_spacings_rcp = 1.f / spacings;
// }

// void 
// StructuredRegularVolume::set_macrocell(vec3i* dims, vec3f* spacings, vec3f* spacings_rcp, vec2f* d_value_range, float* d_max_opacity, size_t* d_mip_size)
// {
//   self.list_of_macrocell_value_range = d_value_range;
//   self.list_of_macrocell_max_opacity = d_max_opacity;

//   self.list_of_macrocell_dims = dims;
//   self.list_of_macrocell_spacings = spacings;
//   self.list_of_macrocell_spacings_rcp = spacings_rcp;
//   self.list_of_mip_size_before_current_mip_level = d_mip_size;
// }

// void* StructuredRegularVolume::get_device_volume_ptr(/*int use_dda*/)
// {
//   // self.use_dda = use_dda;
//   return (void*)&self;
// }

// void StructuredRegularVolume::setup_multi_level_dda(vec3i dims)
// {
//   std::cout<<"USDA Mode: "<< self.use_dda << std::endl;

//   // Calculate Total Mip Space
//   size_t total_mip_size = 0;
//   for (int mip = 0; mip < self.dda_layers; mip++)
//   {
//     vec3i mc_size = vec3i(1 << (mip + MACROCELL_SIZE_MIP));
//     vec3i mc_dims = (dims + mc_size - 1) / mc_size;
//     total_mip_size += mc_dims.long_product();
//   }

//   // Malloc space
//   list_m_macrocell_max_opacity_inference = (float*) malloc(sizeof(float) * total_mip_size);
//   list_m_macrocell_inference = (vec2f*) malloc(sizeof(vec2f) * total_mip_size);
//   std::vector<size_t> list_of_mip_size_before_current;

//   int start_pos = 0;
//   for (int mip = 0; mip < self.dda_layers; mip++)
//   {

//     vec3i temp_m_macrocell_dims;
//     vec3f temp_m_macrocell_spacings;
//     CUDABuffer temp_m_macrocell_inferencePtr_device_buffer;
//     CUDABuffer temp_m_macrocell_max_opacity_inferencePtr_device_buffer;

//     int macrocell_per_size = mip + MACROCELL_SIZE_MIP;
//     int output_mc = 1 << macrocell_per_size;
//     handle_list_of_update_macrocell_and_calculate_value_range(
//       temp_m_macrocell_dims,
//       temp_m_macrocell_spacings,
//       temp_m_macrocell_inferencePtr_device_buffer,
//       temp_m_macrocell_max_opacity_inferencePtr_device_buffer,
//       (DeviceStructuredRegularVolume*)&self,
//       dims,
//       output_mc
//     );

//     list_of_m_macrocell_dims.push_back(temp_m_macrocell_dims);
//     list_of_mip_size_before_current.push_back(start_pos);
//     list_of_m_macrocell_spacings.push_back(temp_m_macrocell_spacings);
//     list_of_m_macrocell_spacings_rcp.push_back(1.f / temp_m_macrocell_spacings);

//     vec3i mc_size = vec3i(output_mc);
//     vec3i mc_dims = (dims + mc_size - 1) / mc_size;

//     temp_m_macrocell_max_opacity_inferencePtr_device_buffer.download(
//       list_m_macrocell_max_opacity_inference + start_pos,
//       mc_dims.long_product()
//     );
//     temp_m_macrocell_inferencePtr_device_buffer.download(
//       list_m_macrocell_inference + start_pos,
//       mc_dims.long_product()
//     );

//     start_pos += mc_dims.long_product();
//   }

//   list_of_m_macrocell_dims_device_buffer.alloc_and_upload(list_of_m_macrocell_dims);
//   list_of_m_macrocell_spacings_device_buffer.alloc_and_upload(list_of_m_macrocell_spacings);
//   list_of_m_macrocell_spacings_rcp_device_buffer.alloc_and_upload(list_of_m_macrocell_spacings_rcp);
//   list_of_m_macrocell_max_opacity_inferencePtr_device_buffer.alloc_and_upload(list_m_macrocell_max_opacity_inference, total_mip_size);
//   list_of_m_macrocell_inferencePtr_device_buffer.alloc_and_upload(list_m_macrocell_inference, total_mip_size);
//   list_of_mip_size_before_current_mip_level_device_buffer.alloc_and_upload(list_of_mip_size_before_current);

//   set_macrocell(
//     (vec3i*)list_of_m_macrocell_dims_device_buffer.d_pointer(),
//     (vec3f*)list_of_m_macrocell_spacings_device_buffer.d_pointer(),
//     (vec3f*)list_of_m_macrocell_spacings_rcp_device_buffer.d_pointer(),
//     (vec2f*)list_of_m_macrocell_inferencePtr_device_buffer.d_pointer(),
//     (float*)list_of_m_macrocell_max_opacity_inferencePtr_device_buffer.d_pointer(),
//     (size_t*)list_of_mip_size_before_current_mip_level_device_buffer.d_pointer()
//   );

// }

} // namespace ovr::optix7
