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

#include "params.h"

#include <random/random.h>

#include <optix_device.h>

namespace ovr {
namespace optix7 {

/*! launch parameters in constant memory, filled in by optix upon optixLaunch
    (this gets filled in from the buffer we pass to optixLaunch) */
extern "C" __constant__ LaunchParams optix_launch_params;

struct ShadowPayload {
  /* shadow intensity */
  float alpha = 0.f; 

  /* by ray marching */
  void* rng = nullptr;
  float t_max = 0.f;
};

struct RayMarchingPayload {
  /* radiance output */
  float alpha = 0.f;
  vec3f color = 0.f;
  vec3f gradient = 0.f;
  vec2f optical_flow = 0.f;

  /* by ray marching */
  void* rng = nullptr;
  float t_max = 0.f;

  inline __device__ void reset()
  {
    alpha = 0.f;
    color = 0.f;
    gradient = 0.f;
    optical_flow = 0.f;
  }
};

struct PathTracingPayload {
  /* radiance output */
  float alpha = 0.f;
  vec3f color = 0.f;

  /* by path tracing */
  void* rng = nullptr;
  int32_t scatter_index = 0;
  const affine3f* wto = nullptr; /* world to object transform */

  int mip_level = 0;
};

//------------------------------------------------------------------------------
// important helper functions
// ------------------------------------------------------------------------------

namespace {

inline __device__ float
corrected_value(const float in)
{
#ifdef FORCE_NAN_CORRECTION
  return isnan(in) ? 0.f : clamp(in, 0.f, 1.f);
#else
  return clamp(in, 0.f, 1.f);
#endif
}

inline __device__ vec3f
corrected_value(const vec3f in)
{
  vec3f out;
#ifdef FORCE_NAN_CORRECTION
  out.x = isnan(in.x) ? 0.f : clamp(in.x, 0.f, 1.f);
  out.y = isnan(in.y) ? 0.f : clamp(in.y, 0.f, 1.f);
  out.z = isnan(in.z) ? 0.f : clamp(in.z, 0.f, 1.f);
#else
  out.x = clamp(in.x, 0.f, 1.f);
  out.y = clamp(in.y, 0.f, 1.f);
  out.z = clamp(in.z, 0.f, 1.f);
#endif
  return out;
}

} // namespace

static __forceinline__ __device__ void*
unpack_pointer(uint32_t i0, uint32_t i1)
{
  const auto uptr = static_cast<uint64_t>(i0) << 32U | i1;
  void* ptr = reinterpret_cast<void*>(uptr);
  return ptr;
}

static __forceinline__ __device__ void
pack_pointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
  const auto uptr = reinterpret_cast<uint64_t>(ptr);
  i0 = uptr >> 32U;
  i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T*
get_prd()
{
  const uint32_t u0 = optixGetPayload_0();
  const uint32_t u1 = optixGetPayload_1();
  return reinterpret_cast<T*>(unpack_pointer(u0, u1));
}

template<typename T>
static __forceinline__ __device__ const T&
get_program_data()
{
  return **((const T**)optixGetSbtDataPointer());
}

inline __device__ bool
intersect_box(float& _t0, float& _t1, const vec3f ray_org, const vec3f ray_dir, const vec3f lower, const vec3f upper)
{
  float t0 = _t0;
  float t1 = _t1;
#if 1
  const vec3i is_small =
    vec3i(fabs(ray_dir.x) < float_small, fabs(ray_dir.y) < float_small, fabs(ray_dir.z) < float_small);
  const vec3f rcp_dir = vec3f(__frcp_rn(ray_dir.x), __frcp_rn(ray_dir.y), __frcp_rn(ray_dir.z));
  const vec3f t_lo = vec3f(is_small.x ? float_large : (lower.x - ray_org.x) * rcp_dir.x, //
                           is_small.y ? float_large : (lower.y - ray_org.y) * rcp_dir.y, //
                           is_small.z ? float_large : (lower.z - ray_org.z) * rcp_dir.z  //
  );
  const vec3f t_hi = vec3f(is_small.x ? -float_large : (upper.x - ray_org.x) * rcp_dir.x, //
                           is_small.y ? -float_large : (upper.y - ray_org.y) * rcp_dir.y, //
                           is_small.z ? -float_large : (upper.z - ray_org.z) * rcp_dir.z  //
  );
  t0 = max(t0, reduce_max(min(t_lo, t_hi)));
  t1 = min(t1, reduce_min(max(t_lo, t_hi)));
#else
  const vec3f t_lo = (lower - ray_org) / ray_dir;
  const vec3f t_hi = (upper - ray_org) / ray_dir;
  t0 = max(t0, reduce_max(min(t_lo, t_hi)));
  t1 = min(t1, reduce_min(max(t_lo, t_hi)));
#endif
  _t0 = t0;
  _t1 = t1;
  return t1 > t0;
}

inline __device__ float // it looks like we can only read textures as float
sample_volume_object_space(const Array3DScalarOptix7& self, vec3f p)
{
  p.x = clamp(p.x, 0.f, 1.f);
  p.y = clamp(p.y, 0.f, 1.f);
  p.z = clamp(p.z, 0.f, 1.f);
  return tex3D<float>(self.data, p.x, p.y, p.z);
}

inline __device__ vec3f
compute_volume_gradient_object_space(const Array3DScalarOptix7& self,
                                     const vec3f c, // central position
                                     const float v, // central value
                                     vec3f stp)
{
  assert(stp.x > 0.f && "invalid gradient step size");
  assert(stp.y > 0.f && "invalid gradient step size");
  assert(stp.z > 0.f && "invalid gradient step size");
  vec3f ext = c + stp;
  if (ext.x > 1.f)
    stp.x *= -1.f;
  if (ext.y > 1.f)
    stp.y *= -1.f;
  if (ext.z > 1.f)
    stp.z *= -1.f;
  const vec3f gradient(sample_volume_object_space(self, c + vec3f(stp.x, 0, 0)) - v,
                       sample_volume_object_space(self, c + vec3f(0, stp.y, 0)) - v,
                       sample_volume_object_space(self, c + vec3f(0, 0, stp.z)) - v);
  return gradient / stp;
}

inline __device__ float
sample_volume_world_space(const Array3DScalarOptix7& self, const affine3f& xfm, const vec3f& wpos)
{
  vec3f opos = xfmPoint(xfm, wpos);
  return sample_volume_object_space(self, opos);
}

inline __device__ vec3f
compute_volume_gradient_world_space(const Array3DScalarOptix7& self,
                                    const affine3f& xfm,
                                    const vec3f c,
                                    const float v,
                                    const float s)
{
  assert(s > 0.f && "invalid gradient step size");
  const vec3f gradient(sample_volume_world_space(self, xfm, c + vec3f(s, 0, 0)) - v,
                       sample_volume_world_space(self, xfm, c + vec3f(0, s, 0)) - v,
                       sample_volume_world_space(self, xfm, c + vec3f(0, 0, s)) - v);
  return gradient / s;
}

inline __device__ affine3f
get_xfm_wto()
{
  float mat[12]; /* 3x4 row major */
  optixGetWorldToObjectTransformMatrix(mat);
  affine3f xfm;
  xfm.l.vx = vec3f(mat[0], mat[4], mat[8]);
  xfm.l.vy = vec3f(mat[1], mat[5], mat[9]);
  xfm.l.vz = vec3f(mat[2], mat[6], mat[10]);
  xfm.p = vec3f(mat[3], mat[7], mat[11]);
  return xfm;
}

inline __device__ affine3f
get_xfm_otw()
{
  float mat[12]; /* 3x4 row major */
  optixGetObjectToWorldTransformMatrix(mat);
  affine3f xfm;
  xfm.l.vx = vec3f(mat[0], mat[4], mat[8]);
  xfm.l.vy = vec3f(mat[1], mat[5], mat[9]);
  xfm.l.vz = vec3f(mat[2], mat[6], mat[10]);
  xfm.p = vec3f(mat[3], mat[7], mat[11]);
  return xfm;
}

inline __device__ affine3f
get_xfm_camera_to_world()
{
  const auto& camera = optix_launch_params.camera;
  affine3f xfm;
  xfm.l.vx = normalize(camera.horizontal);
  xfm.l.vy = normalize(camera.vertical);
  xfm.l.vz = -normalize(camera.direction);
  xfm.p = camera.position;
  return xfm;
}

inline __device__ affine3f
get_xfm_world_to_camera()
{
  const auto& camera = optix_launch_params.camera;
  const auto x = normalize(camera.horizontal);
  const auto y = normalize(camera.vertical);
  const auto z = -normalize(camera.direction);
  affine3f xfm;
  xfm.l.vx = vec3f(x.x, y.x, z.x);
  xfm.l.vy = vec3f(x.y, y.y, z.y);
  xfm.l.vz = vec3f(x.z, y.z, z.z);
  xfm.p = -camera.position;
  return xfm;
}

inline vec2f __device__
project_to_screen(const vec3f p, const LaunchParams::DeviceCamera& camera)
{
  vec3f wsvec = p - camera.position;
  vec2f screen;
  const float r = length(camera.horizontal);
  const float t = length(camera.vertical);
  screen.x = dot(wsvec, normalize(camera.horizontal)) / r;
  screen.y = dot(wsvec, normalize(camera.vertical)) / t;
  return screen + 0.5f;
}

inline vec2f __device__
compute_optical_flow(const vec3f p)
{
  auto last = project_to_screen(p, optix_launch_params.last_camera);
  auto curr = project_to_screen(p, optix_launch_params.camera);
  return curr - last;
}

template<typename T, int N>
inline __device__ T
array1d_nodal(const ArrayOptix7<1, N>& array, float v)
{
  assert(array.dims.v > 0 && "invalid array size");
  v = clamp(v, 0.f, 1.f);
  float t = fmaf(v, float(array.dims.v - 1), 0.5f) * __frcp_rn(array.dims.v);
  return tex1D<T>(array.data, t);
}

inline __device__ bool
nearly_equal(float x, float y, float epsilon = 1e-7f)
{
  if (fabs(x - y) < epsilon)
    return true; // they are same
  return false;  // they are not same
}

template<typename T>
inline __device__ T
alpha_blend(const T& fg, const T& bg, float fa, float ba, float alpha)
{
  if (alpha > 0)
    return (fg + (1.f - fa) * ba * bg) / alpha;
  else
    return T(0);
}

inline __device__ vec3f
spherical_to_cartesian(const float phi, const float sinTheta, const float cosTheta)
{
  float sinPhi, cosPhi;
  sincosf(phi, &sinPhi, &cosPhi);
  return vec3f(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
}

inline __device__ vec3f
uniform_sample_sphere(const float radius, const vec2f s)
{
  const float phi = 2 * M_PI * s.x;
  const float cosTheta = radius * (1.f - 2.f * s.y);
  const float sinTheta = 2.f * radius * sqrt(s.y * (1.f - s.y));
  return spherical_to_cartesian(phi, sinTheta, cosTheta);
}

inline __device__ vec4f
sample_transfer_function(const DeviceStructuredRegularVolume& self, const float sample)
{
  const Array3DScalarOptix7& volume = self.volume;
  const Array1DFloat4Optix7& tfn_colors = self.tfn.color;
  const Array1DScalarOptix7& tfn_alphas = self.tfn.opacity;

  const auto v = (clamp(sample, volume.lower.v, volume.upper.v) - volume.lower.v) * volume.scale.v;
  vec4f rgba = array1d_nodal<float4>(tfn_colors, v);
  rgba.w = array1d_nodal<float>(tfn_alphas, v); // followed by the alpha correction
  return rgba;
}

inline __device__ float
luminance(const vec3f c)
{
  return 0.212671f * c.x + 0.715160f * c.y + 0.072169f * c.z;
}

//------------------------------------------------------------------------------
// intersection program that computes customized intersections for an AABB
// ------------------------------------------------------------------------------

extern "C" __global__ void
__intersection__volume()
{
  const vec3f org = optixGetObjectRayOrigin();
  const vec3f dir = optixGetObjectRayDirection();

  float t0 = optixGetRayTmin();
  float t1 = optixGetRayTmax();

  if (intersect_box(t0, t1, org, dir, vec3f(0.f), vec3f(1.f))) {
    optixReportIntersection(t0, 0, /* user defined attributes, for now set to 0 */
                            __float_as_int(t0), __float_as_int(t1));
  }
}

inline __device__ void
compute_screen_position(vec2f& screen, uint32_t& pixel_index)
{
  /* compute a test pattern based on pixel ID */
  const int ix = optixGetLaunchIndex().x;
  const int iy = optixGetLaunchIndex().y;
  const auto rsize = vec2f(1.f) / vec2f(optix_launch_params.frame.size);

  if (!optix_launch_params.enable_sparse_sampling) {
    /* method 1 */
    screen = vec2f((float)ix + .5f, (float)iy + .5f) * rsize;
    pixel_index = ix + iy * optix_launch_params.frame.size.x;
  }

  else {
    /* method 2 directly sample a logistic distribution */
#ifdef OVR_OPTIX7_MASKING_VIA_DIRECT_SAMPLING
    {
      const int factor = optix_launch_params.sparse_sampling.downsample_factor;
      const uint32_t index = ix + iy * optix_launch_params.frame.size.x / factor;
      const uint32_t total = optix_launch_params.frame.size.long_product() / (factor * factor);

      const vec2f& focus_center = optix_launch_params.focus_center;
      const float& focus_scale = optix_launch_params.focus_scale;
      const float r = focus_scale * fabs(optix_launch_params.sparse_sampling.dist_logistic[index]);
      const float theta = optix_launch_params.sparse_sampling.dist_uniform[index] * 2 * M_PI;
      const float fx = r * __cosf(theta) + focus_center.x;
      const float fy = r * __sinf(theta) + focus_center.y;
      if (fx <= 0.f || fx >= 1.f || fy <= 0.f || fy >= 1.f)
        return;

      screen = vec2f(fx, fy);

      const uint32_t sx = uint32_t(optix_launch_params.frame.size.x * fx) % optix_launch_params.frame.size.x;
      const uint32_t sy = uint32_t(optix_launch_params.frame.size.y * fy) % optix_launch_params.frame.size.y;
      assert(sx >= 0);
      assert(sy >= 0);
      assert(sx < optix_launch_params.frame.size.x);
      assert(sy < optix_launch_params.frame.size.y);
      pixel_index = sx + sy * optix_launch_params.frame.size.x;
    }
#endif

    /* method 3 stream compaction */
#ifdef OVR_OPTIX7_MASKING_VIA_STREAM_COMPACTION
    {
      int sx = optix_launch_params.sparse_sampling.xs_and_ys[2 * ix + 0];
      int sy = optix_launch_params.sparse_sampling.xs_and_ys[2 * ix + 1];
      assert(sx >= 0);
      assert(sy >= 0);
      assert(sx < optix_launch_params.frame.size.x);
      assert(sy < optix_launch_params.frame.size.y);
      pixel_index = sx + sy * optix_launch_params.frame.size.x;
      screen = vec2f((float)sx + .5f, (float)sy + .5f) * rsize;
    }
#endif
  }
}



}
}
