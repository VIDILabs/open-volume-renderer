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

#include "shaders_common.h"
#include <optix_device.h>

#define inf float_large

namespace ovr {
namespace optix7 {

// for this simple example, we have a single ray type
enum RayType { RAYMARCHING_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

inline __device__ void
raymarching_shadow(const DeviceStructuredRegularVolume& self,
                   // world space ray direction and origin
                   const vec3f org,
                   const vec3f dir,
                   // ray distances
                   const float t_min,
                   const float t_max,
                   // performance tuning
                   const float sampling_scale,
                   // output
                   ShadowPayload& payload)
{
  if (t_min >= t_max) {
    return;
  }

  const Array3DScalarOptix7& volume = self.volume;
  const auto wto = get_xfm_wto();

  float2 t = make_float2(t_min, min(t_max, t_min + sampling_scale * self.step));
  while ((t.y > t.x) && (payload.alpha < 0.9999f)) {
    /* sample data value */
    const auto pos = org + 0.5f * (t.x + t.y) * dir; /* world space position */
    const auto pos_object_space = xfmPoint(wto, pos);
    const auto sample = sample_volume_object_space(volume, pos_object_space);

    /* sampling */
    vec4f rgba = sample_transfer_function(self, sample);
    const auto step = t.y - t.x;
    const auto alpha_adjustment = self.base * step;
    if (!nearly_equal(alpha_adjustment, 1.f)) {
      rgba.w = corrected_value(1.f - __powf(1.f - rgba.w, alpha_adjustment));
    }

    /* blending */
    payload.alpha += (1.f - payload.alpha) * rgba.w;

    t.x = t.y;
    t.y = min(t.x + sampling_scale * self.step, t_max);
  }
}

inline __device__ void
raymarching(const DeviceStructuredRegularVolume& self,
            // world space ray direction and origin
            const vec3f org,
            const vec3f dir,
            // ray distances
            const float t_min,
            const float t_max,
            // output
            RayMarchingPayload& payload)
{
  if (t_min >= t_max) {
    return;
  }

  const Array3DScalarOptix7& volume = self.volume;
  const vec3f& rdim = vec3f(1.f) / vec3f(volume.dims);

  const auto otw = get_xfm_otw();
  const auto wto = get_xfm_wto();
  const auto wtc = get_xfm_world_to_camera();

  float2 t = make_float2(t_min, min(t_max, t_min + self.step));
  while ((t.y > t.x) && (payload.alpha < 0.9999f)) {
    /* sample data value */
    const auto pos = org + 0.5f * (t.x + t.y) * dir; /* world space position */
    const auto pos_object_space = xfmPoint(wto, pos);
    const auto sample = sample_volume_object_space(volume, pos_object_space);

    /* sampling */
    vec4f rgba = sample_transfer_function(self, sample);
    const auto step = t.y - t.x;
    const auto alpha_adjustment = self.base * step;
    if (!nearly_equal(alpha_adjustment, 1.f)) {
      rgba.w = corrected_value(1.f - __powf(1.f - rgba.w, alpha_adjustment));
    }

    /* sample gradient */
    /* (from OSPRay) assume that opacity directly correlates to volume scalar
      field, i.e. that "outside" has lower values; because the gradient point
      towards increasing values we need to flip it. */
    const vec3f normal_o =
      -normalize(compute_volume_gradient_object_space(volume, pos_object_space, sample, rdim)); /* object space */
    const vec3f normal_w = normalize(xfmNormal(otw, normal_o));                                 /* world space */
    const vec3f normal_c = normalize(xfmNormal(wtc, normal_w));                                 /* camera space */

    /* optical flow*/
    const vec2f optical_flow = compute_optical_flow(pos);

    /* shade volume */ // TODO better shading //
    vec3f light_dir = normalize(optix_launch_params.light_directional_pos);
    vec3f light_rgb = vec3f(2.f);
    {
      ShadowPayload shadow;
      shadow.rng = payload.rng;
      shadow.t_max = 0.f;
      uint32_t u0, u1;
      pack_pointer(&shadow, u0, u1);
      // while (shadow.t_max < inf) 
      {
        optixTrace(optix_launch_params.traversable,
                   /*org=*/pos, /*dir=*/light_dir, /*tmin=*/shadow.t_max, /*tmax=*/inf, /*time=*/0.0f,
                   OptixVisibilityMask(255), /* not just volume */
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                   SHADOW_RAY_TYPE, // SBT offset
                   RAY_TYPE_COUNT,  // SBT stride
                   SHADOW_RAY_TYPE, // miss SBT index
                   u0, u1);
      }
      const float cosNL = fabs(dot(light_dir, normal_w));
      rgba.xyz() *= 0.5f + 0.5f * cosNL * light_rgb * (1.f - shadow.alpha);
    }

    /* blending */ /* clang-format off */
    const float tr        = 1.f - payload.alpha;
    payload.color        += tr * corrected_value(rgba.xyz()) * rgba.w;
    payload.gradient     += tr * corrected_value(normal_c) * rgba.w;
    payload.optical_flow += tr * optical_flow * rgba.w;
    payload.alpha        += tr * rgba.w;
    /* clang-format on */

    t.x = t.y;
    t.y = min(t.x + self.step, t_max);
  }
}


//------------------------------------------------------------------------------
// closest hit program that gets called for the closest intersection
// ------------------------------------------------------------------------------

extern "C" __global__ void
__closesthit__volume_raymarching()
{
  const auto& self = get_program_data<DeviceStructuredRegularVolume>();
  auto& data = *get_prd<RayMarchingPayload>();

  float t0 = optixGetRayTmax(); // __int_as_float(optixGetAttribute_0());
  float t1 = __int_as_float(optixGetAttribute_1());

  const vec3f ray_org = optixGetWorldRayOrigin();
  const vec3f ray_dir = optixGetWorldRayDirection();

  // ASSUME: volumes cannot overlap with each other
  // assert(!optix_launch_params.enable_path_tracing);

  // integrate the current volume between [t0, t1]
#ifdef OVR_OPTIX7_JITTER_RAYS
  auto rng = (RandomTEA* const)data.rng;
  t0 += rng->get_floats().x * self.step;
#endif

  /* return pre-multiplied color */
  raymarching(self, ray_org, ray_dir, t0, t1, data);

  data.t_max = t1 + float_small;
}

extern "C" __global__ void
__closesthit__volume_shadow()
{
  const auto& self = get_program_data<DeviceStructuredRegularVolume>();
  auto& data = *get_prd<ShadowPayload>();

  float t0 = optixGetRayTmax(); // __int_as_float(optixGetAttribute_0());
  float t1 = __int_as_float(optixGetAttribute_1());

  const vec3f ray_org = optixGetWorldRayOrigin();
  const vec3f ray_dir = optixGetWorldRayDirection();

  // TODO what to do for path tracer
  // assert(!optix_launch_params.enable_path_tracing);

  /* we use a lower sampling rate for shadow */
  const float step = self.step * 10.f;

#ifdef OVR_OPTIX7_JITTER_RAYS
  auto rng = (RandomTEA* const)data.rng;
  t0 += rng->get_floats().x * step;
#endif
  raymarching_shadow(self, ray_org, ray_dir, t0, t1, step, data);
  data.t_max = t1 + float_small;
}


//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
// ------------------------------------------------------------------------------

extern "C" __global__ void
__miss__raymarching()
{
  auto& payload = *get_prd<RayMarchingPayload>();
  payload.t_max = optixGetRayTmax();
  if (optixGetRayVisibilityMask() & ~VISIBILITY_VOLUME) {
    payload.color = vec3f(0.0, 0.0, 0.0);
    payload.alpha = 0.f;
  }
}

extern "C" __global__ void
__miss__shadow()
{
  auto& payload = *get_prd<ShadowPayload>();
  payload.t_max = optixGetRayTmax();
}


//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------

inline __device__ void
render_raymarching(vec3f org,
                   vec3f dir,
                   void* rng,
                   float& _alpha,
                   vec3f& _color,
                   vec3f& _gradient,
                   vec2f& _optical_flow)
{
  RayMarchingPayload payload;
  payload.rng = rng;

  uint32_t u0, u1;
  pack_pointer(&payload, u0, u1);

  /* trace non-volumes */
  struct {
    float alpha = 0.f;
    vec3f color = 0.f;
    vec3f gradient = 0.f;
    vec2f optical_flow = 0.f;
  } background;

  optixTrace(optix_launch_params.traversable,
             /*org=*/org, /*dir=*/dir, /*tmin=*/0.f, /*tmax=*/inf, /*time=*/0.0f,
             OptixVisibilityMask(~VISIBILITY_VOLUME), /* non-volume */
             OPTIX_RAY_FLAG_DISABLE_ANYHIT,
             RAYMARCHING_RAY_TYPE, // SBT offset
             RAY_TYPE_COUNT,       // SBT stride
             RAYMARCHING_RAY_TYPE, // miss SBT index
             u0, u1);

  background.alpha = payload.alpha;
  background.color = payload.color;
  background.gradient = payload.gradient;
  background.optical_flow = payload.optical_flow;

  payload.reset();
  payload.t_max = 0.f;

  /* trace volumes */
  // TODO test with multiple volumes
  // while (payload.t_max < inf) 
  {
    optixTrace(optix_launch_params.traversable,
               /*org=*/org, /*dir=*/dir, /*tmin=*/payload.t_max, /*tmax=*/inf, /*time=*/0.0f,
               OptixVisibilityMask(VISIBILITY_VOLUME), /* just volume */
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               RAYMARCHING_RAY_TYPE, // SBT offset
               RAY_TYPE_COUNT,       // SBT stride
               RAYMARCHING_RAY_TYPE, // miss SBT index
               u0, u1);
  }

  /* clang-format off */
  const float alpha = payload.alpha + float(1 - payload.alpha) * background.alpha;
  _alpha        += alpha;
  _color        += alpha_blend(payload.color,        background.color,        payload.alpha, background.alpha, alpha);
  _gradient     += alpha_blend(payload.gradient,     background.gradient,     payload.alpha, background.alpha, alpha);
  _optical_flow += alpha_blend(payload.optical_flow, background.optical_flow, payload.alpha, background.alpha, alpha);
  /* clang-format on */
}

extern "C" __global__ void
__raygen__render_frame()
{
  assert(optix_launch_params.frame.size.x > 0 && "invalid framebuffer size");
  assert(optix_launch_params.frame.size.y > 0 && "invalid framebuffer size");

  /* normalized screen plane position, in [0,1]^2 */
  vec2f screen_coord;
  uint32_t pixel_index;   /* output pixel index */
  compute_screen_position(screen_coord, pixel_index);
  const auto& camera = optix_launch_params.camera;
  const auto& rsize = optix_launch_params.frame.size_rcp;

  RandomTEA rng_state(optix_launch_params.frame_index, pixel_index);

  struct {
    float alpha = 0.f;
    vec3f color = 0.f;
    vec3f gradient = 0.f;
    vec2f optical_flow = 0.f;
  } output;
  output.alpha = 0.f;
  output.color = 0.f;
  output.gradient = 0.f;
  output.optical_flow = 0.f;

  int spp = optix_launch_params.sample_per_pixel;
  assert(optix_launch_params.sample_per_pixel > 0 && "'sample_per_pixel' should always be positive");
  for (int s = 0; s < spp; s++) {
    /* jitter pixel for anti-aliasing */
    auto screen = screen_coord;
    if (spp > 1) {
      const auto pixel_jitter = vec2f(rng_state.get_floats()) - 0.5f;
      screen += pixel_jitter * rsize;
    }

    /* generate ray */
    vec3f ray_dir = normalize(camera.direction +                      /* -z axis */
                              (screen.x - 0.5f) * camera.horizontal + /* x shift */
                              (screen.y - 0.5f) * camera.vertical);   /* y shift */

    /* the values we store the PRD pointer in: */
    // uint32_t u0, u1;

    /* our per-ray data for this example. initialization matters! */
    render_raymarching(camera.position, ray_dir, &rng_state, //
                       output.alpha, output.color, output.gradient, output.optical_flow);
  }

  float rspp = 1.f / spp;
  output.alpha *= rspp;
  output.color *= rspp;
  output.gradient *= rspp;
  output.optical_flow *= vec2f(rspp);

  /* and write to frame buffer ... */
  // {
  //   const uint32_t r(255.99f * output.color.x);
  //   const uint32_t g(255.99f * output.color.y);
  //   const uint32_t b(255.99f * output.color.z);
  //   const uint32_t a(255.99f * output.alpha);
  //   /* convert to 32-bit rgba value */
  //   const uint32_t code = (r << 0U) | (g << 8U) | (b << 16U) | (a << 24U);
  //   optix_launch_params.frame.rgba[pixel_index] = code;
  // }

  if (optix_launch_params.enable_frame_accumulation) {
    assert(optix_launch_params.frame_index > 0 && "frame index should always be positive");
    if (optix_launch_params.frame_index == 1) {
      optix_launch_params.frame_accum_rgba[pixel_index] = vec4f(output.color, output.alpha);
      optix_launch_params.frame.rgba[pixel_index] = vec4f(output.color, output.alpha);
    }
    else {
      const vec4f rgba = optix_launch_params.frame_accum_rgba[pixel_index] + vec4f(output.color, output.alpha);
      optix_launch_params.frame_accum_rgba[pixel_index] = rgba;
      optix_launch_params.frame.rgba[pixel_index] = rgba / vec4f(optix_launch_params.frame_index);
    }
  }
  else {
    optix_launch_params.frame.rgba[pixel_index] = vec4f(output.color, output.alpha);
  }

  /* gradient field */
  optix_launch_params.frame.grad[pixel_index] = output.gradient;

  /* to visualize optix flows (for now...) */
  // optix_launch_params.frame.grad[pixel_index] = vec3f(output.optical_flow.x, output.optical_flow.y, 1.f);

  /* to visualize sparse sampling results */
  // optix_launch_params.frame.grad[pixel_index] = vec3f(1.f);
}

}
}
