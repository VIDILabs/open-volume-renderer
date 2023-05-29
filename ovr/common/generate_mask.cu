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

#include "generate_mask.h"

#include "random/random.h"
#include "random/blue_noise.h"

#include "cuda_buffer.h"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

namespace ovr {

namespace {

using namespace misc;

static random::default_rng_t rng{ 1337 };

[[maybe_unused]] void
generate_logistic_dist(CUDABuffer& buffer, float center, float stddev)
{
  random::generate_random_logistic<float>(rng, buffer.sizeInBytes / sizeof(float), (float*)buffer.d_pointer(), center, stddev);
}

[[maybe_unused]] void
generate_uniform_dist(CUDABuffer& buffer, float lower, float upper)
{
  random::generate_random_uniform<float>(rng, buffer.sizeInBytes / sizeof(float), (float*)buffer.d_pointer(), lower, upper);
}

[[maybe_unused]] void
generate_blue_dist(CUDABuffer& buffer, vec2i size, int frame_index)
{
  generate_blue_noise<float>((uint32_t)(buffer.sizeInBytes / sizeof(float)), (float*)buffer.d_pointer(), size, frame_index);
}

__global__ void
generate_sparse_samples(const size_t n_elements, int32_t* __restrict__ output, const float* __restrict__ dist,
                        vec2i size, int factor, vec2f mean, float sigma_rcp2, float base_noise)
{
  const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_elements)
    return;

  int x = i % size.x;
  int y = i / size.x;

  const float aspect = (float)size.x/size.y;
  const float fx = ((float)x / size.x - mean.x);
  const float fy = ((float)y / size.y - mean.y) / aspect;

  // float p = exp(-0.5f * (fx * fx + fy * fy) * sigma_rcp2);
  float p = (1.0f-base_noise) * __expf(-0.5f * (fx * fx + fy * fy) * sigma_rcp2) + base_noise;

  assert(x < size.x);
  assert(y < size.y);

  if (dist[i] < p) {
    output[2 * i + 0] = x;
    output[2 * i + 1] = y;
  }
  else {
    output[2 * i + 0] = -1;
    output[2 * i + 1] = -1;
  }
}

int64_t
generate_and_compact_coordinates(int32_t* d_allocated_output, const CUDABuffer& dist, vec2i size, int factor, vec2f mean, float sigma, float base_noise)
{
  linear_kernel(generate_sparse_samples, 0, 0, size.x * size.y, d_allocated_output,
                (const float*)dist.d_pointer(), size, factor, mean, 1.f / (sigma * sigma), base_noise);

  thrust::device_ptr<int32_t> begin(d_allocated_output);
  thrust::device_ptr<int32_t> end = thrust::remove(thrust::device, begin, begin + size.x * size.y * 2, -1);

  return end - begin;
}

}

int64_t
generate_sparse_sampling_mask_d(int32_t* d_output, 
                                int frame_index,
                                const vec2i& fbsize,
                                const vec2f& focus_center,
                                float focus_scale,
                                float base_noise)
{
  static CUDABuffer dist_uniform;

  dist_uniform.resize(fbsize.long_product() * sizeof(int32_t));

  /* generate distribution and coordinates */
#ifdef OVR_OPTIX7_MASKING_NOISE_UNIFORM
  generate_uniform_dist(dist_uniform, 0.f, 1.f);
#else
  generate_blue_dist(dist_uniform, fbsize, frame_index);
#endif

  return generate_and_compact_coordinates(d_output, dist_uniform, fbsize, 1, focus_center, focus_scale, base_noise);
}

#ifdef OVR_BUILD_CUDA_DEVICES
int64_t
generate_sparse_sampling_mask_h(int32_t* h_output, 
                                int frame_index,
                                const vec2i& fbsize,
                                const vec2f& focus_center,
                                float focus_scale,
                                float base_noise)
{
  static CUDABuffer output;
  output.resize(fbsize.long_product() * sizeof(int32_t) * 2);

  const auto size = generate_sparse_sampling_mask_d((int32_t*)output.d_pointer(), frame_index, fbsize, 
                                                    focus_center, focus_scale, base_noise);

  output.download(h_output, fbsize.long_product() * 2);
  return size;
}
#endif

} // namespace ovr
