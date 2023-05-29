//. ======================================================================== //
//. Copyright 2019-2022 Qi Wu                                                //
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

#include "params.h"

namespace ovr {
namespace optix7 {

// // computing macrocell value range offline
// __global__ void
// macrocell_value_range_kernel(const uint32_t mcDimsX,
//                              const uint32_t mcDimsY,
//                              const uint32_t mcDimsZ,
//                              const uint32_t mcWidth,
//                              vec2f* __restrict__ mcData,
//                              const vec3i volumeDims,
//                              cudaTextureObject_t volumeTexture)
// {
//   // 3D kernel launch
//   vec3i mcID(threadIdx.x+blockIdx.x*blockDim.x,
//              threadIdx.y+blockIdx.y*blockDim.y,
//              threadIdx.z+blockIdx.z*blockDim.z);

//   if (mcID.x >= mcDimsX) return;
//   if (mcID.y >= mcDimsY) return;
//   if (mcID.z >= mcDimsZ) return;

//   int mcIdx = mcID.x + mcDimsX*(mcID.y + mcDimsY*mcID.z);
//   vec2f &mc = mcData[mcIdx];

//   // compute begin/end of VOXELS for this macro-cell
//   vec3i new_mcID(
//     mcID.x * mcWidth - 1,
//     mcID.y * mcWidth - 1,
//     mcID.z * mcWidth - 1
//   );
//   vec3i begin = max(new_mcID, vec3i(0));

//   vec3i new_begin(
//     begin.x + mcWidth + /* plus one for tri-lerp!*/ 1,
//     begin.y + mcWidth + /* plus one for tri-lerp!*/ 1,
//     begin.z + mcWidth + /* plus one for tri-lerp!*/ 1
//   );
//   vec3i end = min(new_begin, volumeDims);

//   vec2f valueRange;
//   for (int iz = begin.z; iz < end.z; iz++)
//     for (int iy = begin.y; iy < end.y; iy++)
//       for (int ix = begin.x; ix < end.x; ix++) {
//           float f;
//           tex3D(&f, volumeTexture, 
//                 (ix + 0.5f) / volumeDims.x, 
//                 (iy + 0.5f) / volumeDims.y, 
//                 (iz + 0.5f) / volumeDims.z);
//           if (f < valueRange.x) {
//             valueRange.x = f;
//           }
//           if (valueRange.y < f) {
//             valueRange.y = f;
//           }
//         }
//   mc.x = valueRange.x - 1.f;
//   mc.y = valueRange.y + 1.f;
// }

// // compute macrocell opacity all together
// __global__ void
// macrocell_max_opacity_kernel(const uint32_t num_cells,
//                              const DeviceTransferFunction tfn,
//                              const vec2f* __restrict__ cell_value_range,
//                              float* __restrict__ cell_max_opacity)
// {
//   extern __shared__ float shared_alphas[];

//   const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
//   assert(blockDim.x == tfn.opacity.dims.long_product());

//   // load tfn into shared memory (assume the number of threads per group equals the length of the alpha array)
//   shared_alphas[threadIdx.x] = ((float*)tfn.opacity.rawptr)[threadIdx.x];
//   __syncthreads();
//   const float* __restrict__ alphas = shared_alphas;

//   // access macrocell value range
//   if (i >= num_cells) return;
//   auto range = cell_value_range[i];
//   range.x += 1.f;
//   range.y -= 1.f; // see function: updare_single_macrocell

//   // compute the max opacity for the cell
//   assert(tfn.opacity.dims.long_product() > 0); // for the first frame, tfn.opacity length might be zero

//   const auto lower = (clamp(range.x, tfn.value_range.x, tfn.value_range.y) - tfn.value_range.x) * tfn.range_rcp_norm;
//   const auto upper = (clamp(range.y, tfn.value_range.x, tfn.value_range.y) - tfn.value_range.x) * tfn.range_rcp_norm;
//   uint32_t i_lower = floorf(fmaf(lower, float(tfn.opacity.dims.long_product()-1), 0.5f)) - 1;
//   uint32_t i_upper = floorf(fmaf(upper, float(tfn.opacity.dims.long_product()-1), 0.5f)) + 1;
//   i_lower = clamp<uint32_t>(i_lower, 0, tfn.opacity.dims.long_product()-1);
//   i_upper = clamp<uint32_t>(i_upper, 0, tfn.opacity.dims.long_product()-1);

//   assert(i_lower < tfn.opacity.dims.long_product());
//   assert(i_upper < tfn.opacity.dims.long_product());

//   float opacity = 0.f;
//   for (auto each_i = i_lower; each_i <= i_upper; ++each_i) {
//     opacity = std::max(opacity, alphas[each_i]);
//   }
//   cell_max_opacity[i] = opacity;
// }

// void update_inference_macrocell(cudaStream_t stream, DeviceStructuredRegularVolume& self) 
// {

//   if (self.tfn.opacity.dims.long_product() <= 0) return;
//   size_t opacity_dim = self.tfn.opacity.dims.long_product();

//   size_t n_elements = self.macrocell_dims.long_product();

//   CUDA_SYNC_CHECK();
//   macrocell_max_opacity_kernel<<< div_round_up(n_elements, opacity_dim), opacity_dim, opacity_dim * sizeof(float), stream>>>(
//     n_elements, 
//     self.tfn,
//     self.macrocell_value_range,
//     self.macrocell_max_opacity
//   );
//   CUDA_SYNC_CHECK();
// }

// void update_inference_macrocell(cudaStream_t stream,  DeviceStructuredRegularVolume& self, std::vector<vec3i> list_of_m_macrocell_dims) 
// {

//   if (self.tfn.opacity.dims.long_product() <= 0) return;
//   size_t opacity_dim = self.tfn.opacity.dims.long_product();

//   int start_pos = 0;
//   for (int each_mip_level = 0; each_mip_level < self.dda_layers; each_mip_level++) {
//     size_t n_elements = list_of_m_macrocell_dims[each_mip_level].long_product();
//     macrocell_max_opacity_kernel<<< div_round_up(n_elements, opacity_dim), opacity_dim, opacity_dim * sizeof(float), stream>>>(
//       n_elements,
//       self.tfn,
//       self.list_of_macrocell_value_range + start_pos,
//       self.list_of_macrocell_max_opacity + start_pos
//     );
//     start_pos += n_elements;
//   }

// }

// void calculate_macrocell_value_range(vec3i& m_macrocell_dims,
//                                      CUDABuffer m_macrocell_inferencePtr_device_buffer,
//                                      vec3i& volume_dims,
//                                      DeviceStructuredRegularVolume* self,
//                                      int macrocell_per_size)
// {
//   trilinear_kernel(macrocell_value_range_kernel,
//                    0, 0,
//                    m_macrocell_dims.x,
//                    m_macrocell_dims.y,
//                    m_macrocell_dims.z, 
//                    macrocell_per_size,
//                    (vec2f*)m_macrocell_inferencePtr_device_buffer.d_pointer(),
//                    volume_dims,
//                    self->volume.data);
// }

// void update_macrocell_and_calculate_value_range(vec3i& m_macrocell_dims,
//                                                 vec3f& m_macrocell_spacings,
//                                                 CUDABuffer& m_macrocell_inferencePtr_device_buffer,
//                                                 CUDABuffer& m_macrocell_max_opacity_inferencePtr_device_buffer,
//                                                 DeviceStructuredRegularVolume* self,
//                                                 vec3i dims)
// {
//   // construct macrocell
//   if (self->volume.data){
//     int macrocell_per_size = MACROCELL_SIZE;
//     m_macrocell_dims = div_round_up(dims, vec3i(macrocell_per_size));
//     m_macrocell_spacings = vec3f(macrocell_per_size) / vec3f(dims);
//     size_t macrocell_size = m_macrocell_dims.long_product();
//     m_macrocell_inferencePtr_device_buffer.resize(macrocell_size * sizeof(vec2f));
//     m_macrocell_max_opacity_inferencePtr_device_buffer.resize(macrocell_size * sizeof(float));
//     calculate_macrocell_value_range(m_macrocell_dims, m_macrocell_inferencePtr_device_buffer, dims, self, macrocell_per_size);
//   }
//   CUDA_SYNC_CHECK();
// }

// void handle_list_of_update_macrocell_and_calculate_value_range(vec3i& m_macrocell_dims,
//                                                               vec3f& m_macrocell_spacings,
//                                                               CUDABuffer& m_macrocell_inferencePtr_device_buffer,
//                                                               CUDABuffer& m_macrocell_max_opacity_inferencePtr_device_buffer,
//                                                               DeviceStructuredRegularVolume* self,
//                                                               vec3i dims,
//                                                               int macrocell_per_size)
// {
//   // construct macrocell
//   if (self->volume.data){
//     m_macrocell_dims = div_round_up(dims, vec3i(macrocell_per_size));
//     m_macrocell_spacings = vec3f(macrocell_per_size) / vec3f(dims);
//     size_t macrocell_size = m_macrocell_dims.long_product();
//     m_macrocell_inferencePtr_device_buffer.resize(macrocell_size * sizeof(vec2f));
//     m_macrocell_max_opacity_inferencePtr_device_buffer.resize(macrocell_size * sizeof(float));
//     calculate_macrocell_value_range(m_macrocell_dims, m_macrocell_inferencePtr_device_buffer, dims, self, macrocell_per_size);
//   }
//   CUDA_SYNC_CHECK();
// }

} // namespace optix7

} // namespace ovr
