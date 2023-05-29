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

#include <optix_device.h>

#include "renderer.h"

namespace ovr {
namespace kernel {

/*! launch parameters in constant memory, filled in by optix upon optixLaunch
    (this gets filled in from the buffer we pass to optixLaunch) */
extern "C" __constant__ LaunchParams optixLaunchParams;

static __forceinline__ __device__ void*
unpackPointer(uint32_t i0, uint32_t i1)
{
  const auto uptr = static_cast<uint64_t>(i0) << 32U | i1;
  void* ptr = reinterpret_cast<void*>(uptr);
  return ptr;
}

static __forceinline__ __device__ void
packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
  const auto uptr = reinterpret_cast<uint64_t>(ptr);
  i0 = uptr >> 32U;
  i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T*
getPRD()
{
  const uint32_t u0 = optixGetPayload_0();
  const uint32_t u1 = optixGetPayload_1();
  return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

template<typename T>
__device__ inline const T&
getProgramData()
{
  return **((const T**)optixGetSbtDataPointer());
}

inline __device__ bool
intersectBox(float& t0, float& t1, const vec3f& ray_org, const vec3f& ray_dir, const vec3f& lower, const vec3f& upper)
{
  const vec3f t_lo = (lower - ray_org) / ray_dir;
  const vec3f t_hi = (upper - ray_org) / ray_dir;
  t0 = max(t0, reduce_max(min(t_lo, t_hi)));
  t1 = min(t1, reduce_min(max(t_lo, t_hi)));
  return t1 > t0;
}

inline __device__ float // it looks like we can only read textures as float
sampleVolumeObjectSpace(const RegularVolumeData& self, vec3f p)
{
  p.x = clamp(p.x, 0.f, 1.f);
  p.y = clamp(p.y, 0.f, 1.f);
  p.z = clamp(p.z, 0.f, 1.f);
  return tex3D<float>(self.data, p.x, p.y, p.z);
}

inline __device__ vec3f
computeVolumeGradientObjectSpace(const RegularVolumeData& self,
                                 const vec3f& c, // central position
                                 const float& v, // central value
                                 const float& s)
{
  vec3f stp = vec3f(s);
  vec3f ext = c + stp;
  if (ext.x > 1.f)
    stp.x *= -1.f;
  if (ext.y > 1.f)
    stp.y *= -1.f;
  if (ext.z > 1.f)
    stp.z *= -1.f;
  const vec3f gradient(sampleVolumeObjectSpace(self, c + vec3f(stp.x, 0, 0)) - v,
                       sampleVolumeObjectSpace(self, c + vec3f(0, stp.y, 0)) - v,
                       sampleVolumeObjectSpace(self, c + vec3f(0, 0, stp.z)) - v);
  return gradient / stp;
}

inline __device__ float
sampleVolumeWorldSpace(const RegularVolumeData& self, const affine3f& xfm, const vec3f& world)
{
  vec3f object = xfmPoint(xfm, world);
  return sampleVolumeObjectSpace(self, object);
}

inline __device__ vec3f
computeVolumeGradientWorldSpace(const RegularVolumeData& self,
                                const affine3f& xfm,
                                const vec3f& c,
                                const float& v,
                                const float& s)
{
  const vec3f gradient(sampleVolumeWorldSpace(self, xfm, c + vec3f(s, 0, 0)) - v,
                       sampleVolumeWorldSpace(self, xfm, c + vec3f(0, s, 0)) - v,
                       sampleVolumeWorldSpace(self, xfm, c + vec3f(0, 0, s)) - v);
  return gradient / s;
}

inline __device__ affine3f
getXfmWTO()
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
getXfmOTW()
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
getXfmCameraToWorld()
{
  const auto& camera = optixLaunchParams.camera;
  affine3f xfm;
  xfm.l.vx = normalize(camera.horizontal);
  xfm.l.vy = normalize(camera.vertical);
  xfm.l.vz = -normalize(camera.direction);
  xfm.p = camera.position;
  return xfm;
}

inline __device__ affine3f
getXfmWorldToCamera()
{
  const auto& camera = optixLaunchParams.camera;
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
projectToScreen(const vec3f& p, const LaunchParams::DeviceCamera& camera)
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
computeOpticalFlow(const vec3f& p)
{
  auto last = projectToScreen(p, optixLaunchParams.last_camera);
  auto curr = projectToScreen(p, optixLaunchParams.camera);
  return curr - last;
}

//------------------------------------------------------------------------------
// intersection program that computes customized intersections for an AABB
// ------------------------------------------------------------------------------

inline __device__ float
corrected_value(const float& in)
{
  return isnan(in) ? 0.f : clamp(in, 0.f, 1.f);
}

inline __device__ vec3f
corrected_value(const vec3f& in)
{
  vec3f out;
  out.x = isnan(in.x) ? 0.f : clamp(in.x, 0.f, 1.f);
  out.y = isnan(in.y) ? 0.f : clamp(in.y, 0.f, 1.f);
  out.z = isnan(in.z) ? 0.f : clamp(in.z, 0.f, 1.f);
  return out;
}

inline __device__ void
shadeVolume(const DeviceStructuredRegularVolume& self,
            // object space ray direction and origin
            const vec3f& ori,
            const vec3f& dir,
            // ray distances
            const float& tMin,
            const float& tMax,
            // output
            RayPayload& payload)
{
  // early terminate
  if (tMin >= tMax) {
    return;
  }

  // get volume and transfer function
  const RegularVolumeData& volume = self.volume;
  const Simple1DArrayData& texColors = self.colors;
  const Simple1DArrayData& texAlphas = self.alphas;

  // initialize constants
  const float step = self.step;

  // start marching
  float t = tMin;
  while (t <= tMax && (payload.alpha < 0.9999f)) {
    /* sample data value */
    const auto p = ori + t * dir; /* object space position */
    const auto sampledValue = sampleVolumeObjectSpace(volume, p);

    // const auto v = sampledValue;
    const auto v = (clamp(sampledValue, volume.lower, volume.upper) - volume.lower) * volume.scale;

    /* classification */
    {
      float sampleAlpha = tex1D<float>(texAlphas.data, v); // followed by the alpha correction
      sampleAlpha = corrected_value(1.f - __powf(1.f - sampleAlpha, self.alpha_adjustment));

      // sample color
      vec3f sampleColor = vec4f(tex1D<float4>(texColors.data, v)).xyz();

      // sample gradient
      const auto otw = getXfmOTW();
      const auto wtc = getXfmWorldToCamera();

      // (from OSPRay) assume that opacity directly correlates to volume scalar
      // field, i.e. that "outside" has lower values; because the gradient point
      // towards increasing values we need to flip it.
      vec3f Ns = -normalize(computeVolumeGradientObjectSpace(volume, p, sampledValue, .001f)); /* object space */
      vec3f Ng = xfmNormal(otw, Ns);                                                           /* world space */
      vec3f Nc = xfmNormal(wtc, Ng);                                                           /* camera space */

      // optical flow
      vec2f flow = computeOpticalFlow(xfmPoint(otw, p));

      const vec3f rayDir = optixGetWorldRayDirection();
      const float cosDN = 0.2f + .8f * fabsf(dot(-rayDir, normalize(Ng)));
      sampleColor = cosDN * sampleColor;

      payload.color += float(1 - payload.alpha) * corrected_value(sampleColor) * sampleAlpha;
      payload.Ng += float(1 - payload.alpha) * corrected_value(normalize(Nc)) * sampleAlpha;
      payload.flow += float(1 - payload.alpha) * flow * sampleAlpha;
      payload.alpha += float(1 - payload.alpha) * sampleAlpha;
    }

    t += step;
  }
}

extern "C" __global__ void
__closesthit__volume()
{
  const auto& self = getProgramData<DeviceStructuredRegularVolume>();
  RayPayload& data = *getPRD<RayPayload>(); // shader volume

  float t0 = __int_as_float(optixGetAttribute_0());
  float t1 = __int_as_float(optixGetAttribute_1());

  RayPayload bgPRD; // shade background
  bgPRD.type = optixGetInstanceId();
  {
    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer(&bgPRD, u0, u1);

    // generate ray direction
    const vec3f rayOri = optixGetWorldRayOrigin();
    const vec3f rayDir = optixGetWorldRayDirection();

    optixTrace(optixLaunchParams.traversable, rayOri + t0 * rayDir, rayDir,
               0.f,   // tmin
               1e20f, // tmax
               0.0f,  // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE, // OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               RADIANCE_RAY_TYPE,   // SBT offset
               RAY_TYPE_COUNT,      // SBT stride
               RADIANCE_RAY_TYPE,   // missSBTIndex
               u0, u1);
  }
  t1 = min(bgPRD.tmax + t0, t1);

  RayPayload fgPRD = data;
  {
    // object ray direction and geometry
    const vec3f objOri(__int_as_float(optixGetAttribute_2()), // passing the obj space ray
                       __int_as_float(optixGetAttribute_3()), // from intersection shader
                       __int_as_float(optixGetAttribute_4()));

    const vec3f objDir(__int_as_float(optixGetAttribute_5()), // passing the obj space ray
                       __int_as_float(optixGetAttribute_6()), // from intersection shader
                       __int_as_float(optixGetAttribute_7()));

    shadeVolume(self, objOri, objDir, t0, t1, fgPRD); // !! pre-multiplied color
    fgPRD.tmax = optixGetRayTmax();                   /* == t0 */
  }

  // compose them
  data.alpha = fgPRD.alpha + float(1 - fgPRD.alpha) * bgPRD.alpha;
  data.color = (fgPRD.color + float(1 - fgPRD.alpha) * bgPRD.color * bgPRD.alpha) / data.alpha;
  data.Ng = (fgPRD.Ng + float(1 - fgPRD.alpha) * bgPRD.Ng * bgPRD.alpha) / data.alpha;
  data.flow = (fgPRD.flow + float(1 - fgPRD.alpha) * bgPRD.flow * bgPRD.alpha) / data.alpha;
}

extern "C" __global__ void
__closesthit__volume_shadow()
{
  /* not going to be used */
}

extern "C" __global__ void
__anyhit__volume() /*! for this simple example, this will remain empty */
{
  RayPayload& prd = *getPRD<RayPayload>();
  /* avoid self intersection */
  if (prd.type == optixGetInstanceId())
    optixIgnoreIntersection();
}

extern "C" __global__ void
__anyhit__volume_shadow()
{ /* not going to be used */
}

extern "C" __global__ void
__intersection__volume()
{
  const vec3f ori = optixGetObjectRayOrigin();
  const vec3f dir = optixGetObjectRayDirection();

  float t0 = optixGetRayTmin();
  float t1 = optixGetRayTmax();

  if (intersectBox(t0, t1, ori, dir, vec3f(0.f), vec3f(1.f))) {
    optixReportIntersection(t0, 0, /* user defined attributes, for now set to 0 */
                            __float_as_int(t0), __float_as_int(t1),
                            __float_as_int(ori.x), //
                            __float_as_int(ori.y), //
                            __float_as_int(ori.z), //
                            __float_as_int(dir.x), //
                            __float_as_int(dir.y), //
                            __float_as_int(dir.z));
  }
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void
__miss__background()
{
  RayPayload& payload = *getPRD<RayPayload>();
  payload.tmax = optixGetRayTmax();
}

extern "C" __global__ void
__miss__shadow()
{
  // we didn't hit anything, so the light is visible
  vec3f& prd = *(vec3f*)getPRD<vec3f>();
  prd = vec3f(1.f);
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void
__raygen__renderFrame()
{
  // compute a test pattern based on pixel ID
  const int ix = optixGetLaunchIndex().x;
  const int iy = optixGetLaunchIndex().y;

  const auto& camera = optixLaunchParams.camera;

  // our per-ray data for this example. what we initialize it to
  // won't matter, since this value will be overwritten by either
  // the miss or hit program, anyway
  RayPayload payload;

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer(&payload, u0, u1);

  // normalized screen plane position, in [0,1]^2
  const vec2f screen(vec2f((float)ix + .5f, (float)iy + .5f) / vec2f(optixLaunchParams.frame.size));

  // generate ray direction
  vec3f rayDir = normalize(camera.direction +                      /* -z axis */
                           (screen.x - 0.5f) * camera.horizontal + /* x shift */
                           (screen.y - 0.5f) * camera.vertical);   /* y shift */

  optixTrace(optixLaunchParams.traversable, camera.position, rayDir,
             0.f,         // tmin
             float_large, // tmax
             0.0f,        // rayTime
             OptixVisibilityMask(255),
             OPTIX_RAY_FLAG_NONE, // OPTIX_RAY_FLAG_DISABLE_ANYHIT,
             RADIANCE_RAY_TYPE,   // SBT offset
             RAY_TYPE_COUNT,      // SBT stride
             RADIANCE_RAY_TYPE,   // missSBTIndex
             u0, u1);

  // pixel index
  const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;

  // and write to frame buffer ...
  {
    const uint32_t r(255.99f * payload.color.x);
    const uint32_t g(255.99f * payload.color.y);
    const uint32_t b(255.99f * payload.color.z);
    const uint32_t a(255.99f * payload.alpha);
    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy) ...
    const uint32_t rgba = (r << 0U) | (g << 8U) | (b << 16U) | (a << 24U);
    optixLaunchParams.frame.rgba[fbIndex] = rgba;
  }
}

} // namespace kernel
} // namespace ovr
