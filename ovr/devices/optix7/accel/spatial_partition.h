#pragma once

#include "../array.h"
#include "../optix7_common.h"

#include "dda.h"

#ifdef __NVCC__
#include <random/random.h>
#endif

#include <array>
#include <vector>

namespace ovr {
namespace optix7 {

#ifdef __NVCC__
typedef random::RandomTEA RandomTEA;
#endif

// single level macrocell + DDA
struct DeviceSpacePartiton_SingleMC {
  enum { MACROCELL_SIZE_MIP = 4, MACROCELL_SIZE = 1 << MACROCELL_SIZE_MIP };
  
  template<bool VARYING_MAJORANT = true>
  struct DeltaTrackingIter;

  range1f* __restrict__ value_ranges{ nullptr };
  float* __restrict__ majorants{ nullptr };
  vec3i dims;
  vec3f spac;
  vec3f spac_rcp;

  inline __device__ float access_majorant(const vec3i& cell) const
  {
    const uint32_t idx = cell.x + cell.y * uint32_t(dims.x) + cell.z * uint32_t(dims.x) * uint32_t(dims.y);
    assert(cell.x < dims.x);
    assert(cell.y < dims.y);
    assert(cell.z < dims.z);
    assert(cell.x >= 0);
    assert(cell.y >= 0);
    assert(cell.z >= 0);
    return majorants[idx];
  }

#ifdef __NVCC__
  inline __device__ DeltaTrackingIter<true> 
  iter(const vec3f& ray_org, const vec3f& ray_dir, const float ray_tnear, const float ray_tfar) const;
#endif

};

#ifdef __NVCC__

  template<>
  struct DeviceSpacePartiton_SingleMC::DeltaTrackingIter<true> : private dda::DDAIter
  {
    using DDAIter::cell;
    using DDAIter::t_next;
    using DDAIter::next_cell_begin;
    __device__ DeltaTrackingIter() {}
    __device__ DeltaTrackingIter(const DeviceSpacePartiton_SingleMC& sp, const vec3f& ray_org, const vec3f& ray_dir, const float ray_tnear, const float ray_tfar)
    {
      const auto& dims = sp.dims;
      const vec3f m_org = ray_org * sp.spac_rcp;
      const vec3f m_dir = ray_dir * sp.spac_rcp;
      DDAIter::init(m_org, m_dir, ray_tnear, ray_tfar, dims);
    }
    __device__ bool hashit(const DeviceSpacePartiton_SingleMC& sp, const vec3f& ray_org, const vec3f& ray_dir, const float ray_tnear, const float ray_tfar, RandomTEA& rng, float& rayt, float& majorant, float& tau)
    {
      const float density_scale = 1.f;
      const float sigma_t = 1.f;
      // const float sigma_s = 1.f;

      const auto& dims = sp.dims;
      const vec3f m_org = ray_org * sp.spac_rcp;
      const vec3f m_dir = ray_dir * sp.spac_rcp;

      bool found_hit = false;
      float t = next_cell_begin + ray_tnear;
      while (DDAIter::next(m_org, m_dir, ray_tnear, ray_tfar, dims, false, [&](const vec3i& c, float t0, float t1) {
        majorant = sp.access_majorant(c) * density_scale;
        if (fabsf(majorant) <= float_epsilon) return true; // move to the next macrocell
        tau -= (t1 - t) * (majorant * sigma_t);
        t = t1;
        if (tau > 0.f) return true; // move to the next macrocell  
        t = t + tau / (majorant * sigma_t); // can have division by zero error
        found_hit = true;
        next_cell_begin = t - ray_tnear;
        rayt = t;
        return false; // found a hit, terminate the loop
      })) {}
      return found_hit;
    }
  };

  template<>
  struct DeviceSpacePartiton_SingleMC::DeltaTrackingIter<false>
  {
    float t = 0;
    __device__ DeltaTrackingIter() {}
    __device__ DeltaTrackingIter(const DeviceSpacePartiton_SingleMC& sp, const vec3f& ray_org, const vec3f& ray_dir, const float ray_tnear, const float ray_tfar) {}
    __device__ bool hashit(const DeviceSpacePartiton_SingleMC& sp, const vec3f& ray_org, const vec3f& ray_dir, const float ray_tnear, const float ray_tfar, RandomTEA& rng, float& rayt, float& majorant, float& tau)
    {   
      const float sigma_t = 1.f;
      // const float sigma_s = 1.f;
      majorant = 1.f;

      t += -logf(1.f - rng.get_floats().x) / (majorant * sigma_t);
      rayt = ray_tnear + t;

      return (rayt <= ray_tfar); 
    }
  };

  inline __device__ DeviceSpacePartiton_SingleMC::DeltaTrackingIter<true> 
  DeviceSpacePartiton_SingleMC::iter(const vec3f& ray_org, const vec3f& ray_dir, const float ray_tnear, const float ray_tfar) const
  {
    return DeltaTrackingIter<true>(*this, ray_org, ray_dir, ray_tnear, ray_tfar);
  }

#endif

struct SpacePartiton_SingleMC {
public:
  using Self = DeviceSpacePartiton_SingleMC;
  Self self;

  CUDABuffer value_range_buffer;
  CUDABuffer majorant_buffer;

public:
  void allocate(vec3i dims);
  void compute_value_range(vec3i dims, cudaTextureObject_t data);
  void compute_majorant(const DeviceTransferFunction& tfn, cudaStream_t stream = 0);
};

} // namespace optix7
} // namespace ovr
