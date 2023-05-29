#include "spatial_partition.h"

#include <cuda_misc.h>

using namespace ovr::misc;

namespace ovr {
namespace optix7 {

__global__ void
value_range_kernel(const vec3i volumeDims,
                   cudaTextureObject_t volumeTexture,
                   const vec3i mcDims,
                   const uint32_t mcWidth,
                   range1f* __restrict__ mcData)
{
  const uint32_t mcDimsX = mcDims.x;
  const uint32_t mcDimsY = mcDims.y;
  const uint32_t mcDimsZ = mcDims.z;

  // 3D kernel launch
  vec3i mcID(threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y,
             threadIdx.z + blockIdx.z * blockDim.z);

  if (mcID.x >= mcDimsX)
    return;
  if (mcID.y >= mcDimsY)
    return;
  if (mcID.z >= mcDimsZ)
    return;

  int mcIdx = mcID.x + mcDimsX * (mcID.y + mcDimsY * mcID.z);
  range1f& mc = mcData[mcIdx];

  // compute begin/end of VOXELS for this macro-cell
  vec3i new_mcID(mcID.x * mcWidth - 1, mcID.y * mcWidth - 1, mcID.z * mcWidth - 1);
  vec3i begin = max(new_mcID, vec3i(0));

  vec3i new_begin(begin.x + mcWidth + /* plus one for tri-lerp!*/ 1, 
                  begin.y + mcWidth + /* plus one for tri-lerp!*/ 1,
                  begin.z + mcWidth + /* plus one for tri-lerp!*/ 1);
  vec3i end = min(new_begin, volumeDims);

  range1f valueRange;
  for (int iz = begin.z; iz < end.z; iz++)
    for (int iy = begin.y; iy < end.y; iy++)
      for (int ix = begin.x; ix < end.x; ix++) {
        float f;
        tex3D(&f, volumeTexture, (ix + 0.5f) / volumeDims.x, (iy + 0.5f) / volumeDims.y, (iz + 0.5f) / volumeDims.z);
        valueRange.extend(f);
      }

  mc = valueRange;
}

__global__ void
majorant_kernel(const uint32_t count,
                const DeviceTransferFunction tfn,
                const range1f* __restrict__ value_ranges,
                float* __restrict__ majorants)
{
  extern __shared__ float shared_alphas[];

  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= count)
    return;

  // load tfn into shared memory (assume the number of threads per group equals the length of the alpha array)
  assert(blockDim.x == tfn.opacity.dims.long_product());
  shared_alphas[threadIdx.x] = ((float*)tfn.opacity.rawptr)[threadIdx.x];
  __syncthreads();
  const float* __restrict__ alphas = shared_alphas;

  // access macrocell value range
  const auto range = value_ranges[i];

  // compute the max opacity for the cell
  assert(tfn.opacity.dims.long_product() > 0); // for the first frame, tfn.opacity length might be zero

  const auto lower =
    (clamp(range.lower, tfn.value_range.x, tfn.value_range.y) - tfn.value_range.x) * tfn.range_rcp_norm;
  const auto upper =
    (clamp(range.upper, tfn.value_range.x, tfn.value_range.y) - tfn.value_range.x) * tfn.range_rcp_norm;
  uint32_t i_lower = floorf(fmaf(lower, float(tfn.opacity.dims.long_product() - 1), 0.5f)) - 1;
  uint32_t i_upper = floorf(fmaf(upper, float(tfn.opacity.dims.long_product() - 1), 0.5f)) + 1;
  i_lower = clamp<uint32_t>(i_lower, 0, tfn.opacity.dims.long_product() - 1);
  i_upper = clamp<uint32_t>(i_upper, 0, tfn.opacity.dims.long_product() - 1);

  assert(i_lower < tfn.opacity.dims.long_product());
  assert(i_upper < tfn.opacity.dims.long_product());

  float opacity = 0.f;
  for (auto each_i = i_lower; each_i <= i_upper; ++each_i) {
    opacity = std::max(opacity, alphas[each_i]);
  }
  majorants[i] = opacity;
}

void
SpacePartiton_SingleMC::allocate(vec3i dims)
{
  self.dims = div_round_up(dims, vec3i(Self::MACROCELL_SIZE));
  self.spac = vec3f(Self::MACROCELL_SIZE) / vec3f(dims);
  self.spac_rcp = 1.f / self.spac;

  value_range_buffer.resize(self.dims.long_product() * sizeof(range1f));
  majorant_buffer.resize(self.dims.long_product() * sizeof(float));

  self.value_ranges = (range1f*)value_range_buffer.d_pointer();
  self.majorants = (float*)majorant_buffer.d_pointer();
}

void
SpacePartiton_SingleMC::compute_value_range(vec3i dims, cudaTextureObject_t data)
{
  trilinear_kernel(value_range_kernel, 0, 0, dims, //
                   data, self.dims, Self::MACROCELL_SIZE,
                   (range1f*)value_range_buffer.d_pointer());
  CUDA_SYNC_CHECK();
}

void
SpacePartiton_SingleMC::compute_majorant(const DeviceTransferFunction& tfn, cudaStream_t stream)
{
  const uint32_t N = (uint32_t)tfn.opacity.dims.long_product();

  if (N <= 0)
    return;

  const uint32_t n_elements = (uint32_t)self.dims.long_product();

  // CUDA_SYNC_CHECK();
  majorant_kernel<<<div_round_up(n_elements, N), N, N * sizeof(float), stream>>>( //
    n_elements, tfn, self.value_ranges, self.majorants //
  ); //
  // CUDA_SYNC_CHECK();
}

} // namespace optix7
} // namespace ovr
