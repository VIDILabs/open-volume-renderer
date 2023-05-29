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

#define inf float_large

// #define FORCE_NAN_CORRECTION
#define VARYING_MAJORANT 1
#define USE_DELTA_TRACKING_ITER 1

namespace ovr {
namespace optix7 {

// for this simple example, we have a single ray type
enum RayType { PATHTRACING_RAY_TYPE = 0, RAY_TYPE_COUNT };

//------------------------------------------------------------------------------
// important helper functions
// ------------------------------------------------------------------------------

// ------------------------------------------------------------------
// dda iter helper
// ------------------------------------------------------------------

// struct DeltaTrackingIter 
// #if VARYING_MAJORANT
//   : private dda::DDAIter
// #endif
// {
// #if VARYING_MAJORANT
//   using DDAIter::cell;
//   using DDAIter::t_next;
//   using DDAIter::next_cell_begin;
// #else
//   float t;
// #endif
//   __device__ DeltaTrackingIter() {}
//   __device__ DeltaTrackingIter(const DeviceStructuredRegularVolume& self, const vec3f& ray_org, const vec3f& ray_dir, const float ray_tnear, const float ray_tfar);
//   // __device__ DeltaTrackingIter(const DeviceStructuredRegularVolume& self, const vec3f& ray_org, const vec3f& ray_dir, const float ray_tnear, const float ray_tfar, const int each_mip_level);
//   __device__ bool hashit(const DeviceStructuredRegularVolume& self, const vec3f& ray_org, const vec3f& ray_dir, const float ray_tnear, const float ray_tfar, RandomTEA& rng, float& t, float& majorant, float& tau);
//   // __device__ bool hashit_for_multilevel(const DeviceStructuredRegularVolume& self, const vec3f& ray_org, const vec3f& ray_dir, const float ray_tnear, const float ray_tfar, RandomTEA& rng, float& t, float& majorant, float& initial_mip, bool& want_to_go_on, float& tau, const bool world_space, int& mip_level);
// };

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

// inline __device__ float
// opacityUpperBound(const DeviceStructuredRegularVolume& self, const vec3i& cell)
// {
//   const auto& dims = self.macrocell_dims;

//   const uint32_t idx = cell.x + cell.y * uint32_t(dims.x) + cell.z * uint32_t(dims.x) * uint32_t(dims.y);
//   assert(cell.x < dims.x);
//   assert(cell.y < dims.y);
//   assert(cell.z < dims.z);
//   assert(cell.x >= 0);
//   assert(cell.y >= 0);
//   assert(cell.z >= 0);

//   return self.macrocell_max_opacity[idx];
// }

// inline __device__ float
// opacityUpperBound(const DeviceStructuredRegularVolume& self, const vec3i& cell, const int each_mip_level)
// {
//   const auto& current_dim = self.list_of_macrocell_dims[each_mip_level];

//   const uint32_t idx = cell.x + cell.y * uint32_t(current_dim.x) + cell.z * uint32_t(current_dim.x) * uint32_t(current_dim.y);
//   assert(cell.x < current_dim.x);
//   assert(cell.y < current_dim.y);
//   assert(cell.z < current_dim.z);
//   assert(cell.x >= 0);
//   assert(cell.y >= 0);
//   assert(cell.z >= 0);

//   // size_t total_mip_size_before_current_mip_level = 0;
//   // for (int each_mip = 0; each_mip < each_mip_level; each_mip++){
//   //   total_mip_size_before_current_mip_level += self.list_of_macrocell_dims[each_mip].long_product();
//   // }
//   // return self.list_of_macrocell_max_opacity[total_mip_size_before_current_mip_level + idx];
  
//   return self.list_of_macrocell_max_opacity[self.list_of_mip_size_before_current_mip_level[each_mip_level] + idx];
// }

// __device__
// DeltaTrackingIter::DeltaTrackingIter(const DeviceStructuredRegularVolume& self, const vec3f& ray_org, const vec3f& ray_dir, const float ray_tnear, const float ray_tfar)
// {
// #if VARYING_MAJORANT
//   const auto& dims = self.sp.dims;
//   const vec3f m_org = ray_org * self.sp.spac_rcp;
//   const vec3f m_dir = ray_dir * self.sp.spac_rcp;
//   DDAIter::init(m_org, m_dir, ray_tnear, ray_tfar, dims);
// #else
//   t = 0;
// #endif
// }

// __device__
// DeltaTrackingIter::DeltaTrackingIter(const DeviceStructuredRegularVolume& self, const vec3f& ray_org, const vec3f& ray_dir, const float ray_tnear, const float ray_tfar, const int mip_level)
// {
// #if VARYING_MAJORANT
//   const auto& dims = self.list_of_macrocell_dims[mip_level];
//   auto spacing_rcp = self.list_of_macrocell_spacings_rcp[mip_level];
//   const vec3f m_org = ray_org * spacing_rcp;
//   const vec3f m_dir = ray_dir * spacing_rcp;
//   DDAIter::init(m_org, m_dir, ray_tnear, ray_tfar, dims);
// #else
//   t = 0;
// #endif
// }

// __device__ bool
// DeltaTrackingIter::hashit(const DeviceStructuredRegularVolume& self,
//                           const vec3f& ray_org,
//                           const vec3f& ray_dir,
//                           const float  ray_tnear,
//                           const float  ray_tfar,
//                           RandomTEA& rng,
//                           float& rayt,
//                           float& majorant,
//                           float& tau)
// {
// #if VARYING_MAJORANT
//   const float density_scale = 1.f;
//   const float sigma_t = 1.f;
//   const float sigma_s = 1.f;

//   const auto& dims = self.sp.dims;
//   const vec3f m_org = ray_org * self.sp.spac_rcp;
//   const vec3f m_dir = ray_dir * self.sp.spac_rcp;

//   bool found_hit = false;
//   float t = next_cell_begin + ray_tnear;
//   while (DDAIter::next(m_org, m_dir, ray_tnear, ray_tfar, dims, false, [&](const vec3i& c, float t0, float t1) {
//     majorant = self.sp.access_majorant(c) * density_scale;
//     if (fabsf(majorant) <= float_epsilon) return true; // move to the next macrocell
//     tau -= (t1 - t) * (majorant * sigma_t);
//     t = t1;
//     if (tau > 0.f) return true; // move to the next macrocell  
//     t = t + tau / (majorant * sigma_t); // can have division by zero error
//     found_hit = true;
//     next_cell_begin = t - ray_tnear;
//     rayt = t;
//     return false; // found a hit, terminate the loop
//   })) {}
//   return found_hit;

// #else

//   const float sigma_t = 1.f;
//   const float sigma_s = 1.f;
//   majorant = 1.f;

//   t += -logf(1.f - rng.get_float()) / (majorant * sigma_t);
//   rayt = ray_tnear + t;

//   return (rayt <= ray_tfar);

// #endif
// }

// __device__ bool
// DeltaTrackingIter::hashit_for_multilevel(const DeviceStructuredRegularVolume& self,
//                                          const vec3f& ray_org,
//                                          const vec3f& ray_dir,
//                                          const float ray_tnear,
//                                          const float ray_tfar,
//                                          RandomTEA& rng,
//                                          float& rayt,
//                                          float& majorant,
//                                          float& initial_mip,
//                                          bool& want_to_go_on,
//                                          float& tau,
//                                          const bool world_space,
//                                          int& mip_level)
// {
// #if VARYING_MAJORANT

//   const auto& dims = self.list_of_macrocell_dims[mip_level];
//   auto spacing_rcp = self.list_of_macrocell_spacings_rcp[mip_level];
//   const vec3f m_org = ray_org * spacing_rcp;
//   const vec3f m_dir = ray_dir * spacing_rcp;
//   const auto& density_scale = 1.f;
//   const float sigma_t = 1.f;
//   const float sigma_s = 1.f;
//   bool found_hit = false;
//   next_cell_begin = 0;
//   float t = rayt;
//   while (DDAIter::next(m_org, m_dir, rayt, ray_tfar, dims, false, [&](const vec3i& c, float t0, float t1) {
//     majorant = opacityUpperBound(self, c, mip_level) * density_scale;
//     if (fabsf(majorant) <= float_epsilon) return true; // move to the next macrocell

//     float dtau = (t1 - t) * (majorant * sigma_t);
//     tau -= dtau;
//     t = t1;

//     if (tau > 0.f) {
//       initial_mip = min(initial_mip + 0.33f, (float)self.dda_layers - 1);
//       if (mip_level == round(initial_mip)) return true;
//       rayt = t;
//       mip_level = round(initial_mip);
//       want_to_go_on = true;
//       return false;
//     }

//     t = t + tau / (majorant * sigma_t); // can have division by zero error
//     next_cell_begin = t - rayt;
//     rayt = t;

//     if (mip_level > 0) {
//       tau += dtau;
//       initial_mip = max(0.f, (float)initial_mip - 2);
//       mip_level = round(initial_mip);
//       want_to_go_on = true;
//       return false;
//     }
    
//     found_hit = true;
//     return false; // found a hit, terminate the loop
//   })) {}

//   return found_hit;

// #else

//   const float sigma_t = 1.f;
//   const float sigma_s = 1.f;
//   majorant = self.density_scale;

//   t += -logf(1.f - rng.get_float()) / (majorant * sigma_t);
//   rayt = ray_tnear + t;

//   return (rayt <= ray_tfar);

// #endif
// }

inline __device__ bool
delta_tracking(const DeviceStructuredRegularVolume& self,
               RandomTEA* const rng,
               const vec3f ray_org,
               const vec3f ray_dir,
               const float t_min,
               const float t_max,
               const bool world_space,
               float& _t,
               vec3f& _albedo,
               int& mip_level)
{
  const auto wto = get_xfm_wto();
  const auto density_scale = 1.f;
  const auto max_opacity = 1.f;
  const auto mu_max = density_scale * max_opacity;

  const float sigma_t = 1.f;

  float t = t_min;
  vec3f albedo(0);
  bool found_hit = false;

  if (self.use_dda == 2) {
    // #if USE_DELTA_TRACKING_ITER

    //   float tau = -logf(1.f - rng->get_floats().x);
    //   float initial_mip = (float)mip_level;
    //   const vec3f convert_ray_org = (world_space ? xfmPoint(wto, ray_org) : ray_org);
    //   const vec3f convert_ray_dir = (world_space ? xfmPoint(wto, ray_dir) : ray_dir);
    //   while (true) {
    //     bool want_to_go_on = false;
    //     float majorant;
    //     DeltaTrackingIter iter(self, convert_ray_org, convert_ray_dir, t_min, t_max, mip_level);
    //     while (iter.hashit_for_multilevel(self, convert_ray_org, convert_ray_dir,
    //                                       t_min, t_max, *rng, t, majorant, initial_mip,
    //                                       want_to_go_on, tau, world_space, mip_level)) {
    //       auto c = world_space ? xfmPoint(wto, ray_org + t * ray_dir) // convert to the object space
    //                           : ray_org + t * ray_dir;               // already in the object space
    //       const auto sample = sample_volume_object_space(self.volume, c);
    //       const auto rgba = sample_transfer_function(self, sample);
    //       if (rng->get_floats().x * majorant < rgba.w * density_scale) {
    //         albedo = rgba.xyz();
    //         found_hit = true;
    //         break;
    //       }
    //       tau = -logf(1.f - rng->get_floats().x);
    //     }
    //     if(found_hit) break;
    //     if(!want_to_go_on) break;
    //   }

    // #else

    //   float tau = -logf(1.f - rng->get_floats().x);
    //   float initial_mip = (float)mip_level;
    //   while (true) {
    //     bool want_to_go_on = false;
    //     auto dims = self.list_of_macrocell_dims[mip_level];
    //     auto spacing_rcp = self.list_of_macrocell_spacings_rcp[mip_level];
    //     const vec3f m_org = (world_space ? xfmPoint(wto, ray_org) : ray_org) * spacing_rcp;
    //     const vec3f m_dir = (world_space ? xfmPoint(wto, ray_dir) : ray_dir) * spacing_rcp;

    //     dda::dda3(m_org, m_dir, t, t_max, dims, false, [&](const vec3i& cell, float t0, float t1) {

    //       const float majorant = opacityUpperBound(self, cell, mip_level) * density_scale;
    //       if (fabsf(majorant) <= float_epsilon) return true;

    //       while (t < t1) {

    //         const float dt = t1 - t;
    //         const float dtau = dt * (majorant * sigma_t);

    //         t = t1;
    //         tau -= dtau;

    //         if (tau > 0.f) {
    //           initial_mip = min(initial_mip + 0.33f, (float)self.dda_layers - 1);
    //           if (mip_level == round(initial_mip)) return true;
    //           t++;
    //           mip_level = round(initial_mip);
    //           want_to_go_on = true;
    //           return false;
    //         }

    //         t = t + tau / (majorant * sigma_t); // can have division by zero error
    //         if (mip_level > 0) {
    //           tau += dtau;
    //           initial_mip = max(0.f, (float)initial_mip - 2);
    //           mip_level = round(initial_mip);
    //           want_to_go_on = true;
    //           return false;
    //         }

    //         auto c = world_space ? xfmPoint(wto, ray_org + t * ray_dir) // convert to the object space
    //                           : ray_org + t * ray_dir;               // already in the object space
    //         const auto sample = sample_volume_object_space(self.volume, c);
    //         const auto rgba = sample_transfer_function(self, sample);                                    
    //         if (rng->get_floats().x * majorant < rgba.w * density_scale) {
    //           albedo = rgba.xyz();
    //           found_hit = true;
    //           return false;
    //         }

    //         tau = -logf(1.f - rng->get_floats().x);
    //       }

    //       return true;

    //     });

    //     if(found_hit) break;
    //     if(!want_to_go_on) break;
    //   }
    // #endif
  }
  else if (self.use_dda == 1) {
    #if USE_DELTA_TRACKING_ITER

      float majorant;
      const vec3f obj_ray_org = (world_space ? xfmPoint(wto, ray_org) : ray_org);
      const vec3f obj_ray_dir = (world_space ? xfmPoint(wto, ray_dir) : ray_dir);

      auto iter = self.sp.iter(obj_ray_org, obj_ray_dir, t_min, t_max);
      float tau = -logf(1.f - rng->get_floats().x);
      while (iter.hashit(self.sp, obj_ray_org, obj_ray_dir, t_min, t_max, *rng, t, majorant, tau)) {
        const auto c = obj_ray_org + t * obj_ray_dir;
        const auto sample = sample_volume_object_space(self.volume, c);
        const auto rgba = sample_transfer_function(self, sample);
        if (rng->get_floats().x * majorant < rgba.w * density_scale) {
          albedo = rgba.xyz();
          found_hit = true;
          break;
        }
        tau = -logf(1.f - rng->get_floats().x);
      }

    #else

      const auto dims = self.sp.dims;
      const vec3f m_org = (world_space ? xfmPoint(wto, ray_org) : ray_org) * self.sp.spac_rcp;
      const vec3f m_dir = (world_space ? xfmPoint(wto, ray_dir) : ray_dir) * self.sp.spac_rcp;

      float tau = -logf(1.f - rng->get_floats().x);
      dda::dda3(m_org, m_dir, t_min, t_max, dims, false, [&](const vec3i& cell, float t0, float t1) {

        const float majorant = self.sp.access_majorant(cell) * density_scale;
        if (fabsf(majorant) <= float_epsilon) return true;

        while (t < t1) {

          const float dt = t1 - t;
          const float dtau = dt * (majorant * sigma_t);

          t = t1;
          tau -= dtau;

          if (tau > 0.f) return true;

          t = t + tau / (majorant * sigma_t); // can have division by zero error
          auto c = world_space ? xfmPoint(wto, ray_org + t * ray_dir) // convert to the object space
                            : ray_org + t * ray_dir;               // already in the object space
          const auto sample = sample_volume_object_space(self.volume, c);
          const auto rgba = sample_transfer_function(self, sample);                                    
          if (rng->get_floats().x * majorant < rgba.w * density_scale) {
            albedo = rgba.xyz();
            found_hit = true;
            return false;
          }

          tau = -logf(1.f - rng->get_floats().x);
        }

        return true;

      });
    #endif
  }  
  else {

    while (true) {
      const auto xi = rng->get_floats();
      t = t + -logf(1.f - xi.x) / mu_max;

      if (t > t_max) {
        found_hit = false;
        break;
      }

      auto c = world_space ? xfmPoint(wto, ray_org + t * ray_dir) // convert to the object space
                          : ray_org + t * ray_dir;               // already in the object space
      const auto sample = sample_volume_object_space(self.volume, c);
      const auto rgba = sample_transfer_function(self, sample);
      albedo = rgba.xyz();

      const auto mu_t = density_scale * rgba.w;
      if (xi.y < mu_t / mu_max) {
        found_hit = true;
        break;
      }
    }
  }

  _t = t;
  _albedo = albedo;
  return found_hit;
}

inline __device__ void
pathtracing(const DeviceStructuredRegularVolume& self,
            // ray direction and origin
            const vec3f ray_org,
            const vec3f ray_dir,
            // ray distances
            const float t_min,
            const float t_max,
            // output
            PathTracingPayload& payload)
{
  const auto wto = payload.wto ? *payload.wto : get_xfm_wto();
  const auto in_world_space = payload.wto == nullptr;

  auto rng = (RandomTEA* const)payload.rng;
  auto scatter_index = payload.scatter_index;

  vec3f Le(0, 0, 0);

  vec3f albedo;
  float t;
  auto& mip_level = payload.mip_level;
  if (!delta_tracking(self, rng, ray_org, ray_dir, t_min, t_max, in_world_space, t, albedo, mip_level)) {
    if (scatter_index != 0) {
      Le = Le + vec3f(optix_launch_params.light_ambient_intensity); // ambient light
    }
  }

  // new scattering event at sample point
  else {
    scatter_index++;
    if (scatter_index <= optix_launch_params.max_num_scatters) {

      const vec3f _org = ray_org + t * ray_dir;
      const vec3f _dir = uniform_sample_sphere(1.f, rng->get_floats());
      const vec3f scattering_ray_org = in_world_space ? xfmPoint(wto, _org) : _org;
      const vec3f scattering_ray_dir = xfmVector(wto, _dir);

      PathTracingPayload scattering;
      scattering.rng = rng;
      scattering.scatter_index = scatter_index + 1;
      scattering.wto = &wto;
      scattering.mip_level = mip_level;
      {
        uint32_t u0, u1;
        pack_pointer(&scattering, u0, u1);
        optixTrace(optixGetGASTraversableHandle(),
                   /*org=*/scattering_ray_org,
                   /*dir=*/scattering_ray_dir,
                   /*tmin=*/0.f, /*tmax=*/inf, /*rayTime=*/0.0f,
                   OptixVisibilityMask(255),      //
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT, //
                   PATHTRACING_RAY_TYPE,          // SBT offset
                   RAY_TYPE_COUNT,                // SBT stride
                   PATHTRACING_RAY_TYPE,          // miss SBT index
                   u0, u1);
      }

      const vec3f sigma_s_sample = 1.f * albedo;
      Le = Le + sigma_s_sample * scattering.color;
    }
  }

  payload.color = Le;
  payload.alpha = 1.f;
}

//------------------------------------------------------------------------------
// closest hit program that gets called for the closest intersection
// ------------------------------------------------------------------------------

extern "C" __global__ void
__closesthit__volume_pathtracing()
{
  const auto& self = get_program_data<DeviceStructuredRegularVolume>();
  auto& data = *get_prd<PathTracingPayload>();
  // data.mip_level = self.dda_layers - 1;

  float t0 = optixGetRayTmax(); // __int_as_float(optixGetAttribute_0());
  float t1 = __int_as_float(optixGetAttribute_1());

  const vec3f ray_org = optixGetWorldRayOrigin();
  const vec3f ray_dir = optixGetWorldRayDirection();

  assert(optix_launch_params.enable_path_tracing);

  pathtracing(self, ray_org, ray_dir, t0, t1, data);
}

//------------------------------------------------------------------------------
// anyhit program that gets called for any non-opaque intersections
// ------------------------------------------------------------------------------

// extern "C" __global__ void
// __anyhit__volume_raymarching()
// {
//   /* not going to be used */
// }

// extern "C" __global__ void
// __anyhit__volume_pathtracing()
// {
//   /* not going to be used */
// }

// extern "C" __global__ void
// __anyhit__volume_shadow()
// {
//   /* not going to be used */
// }

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
// ------------------------------------------------------------------------------

extern "C" __global__ void
__miss__pathtracing()
{
  // sample ambient light
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------

inline __device__ void
render_pathtracing(vec3f org,
                   vec3f dir,
                   void* rng,
                   float& _alpha,
                   vec3f& _color,
                   vec3f& _gradient,
                   vec2f& _optical_flow)
{
  PathTracingPayload payload;
  payload.rng = rng;

  uint32_t u0, u1;
  pack_pointer(&payload, u0, u1);
  optixTrace(optix_launch_params.traversable,
             /*org=*/org, /*dir=*/dir,
             /*tmin=*/0.f, /*tmax=*/inf, /*time=*/0.0f,
             OptixVisibilityMask(255), // trace toward everything
             OPTIX_RAY_FLAG_DISABLE_ANYHIT,
             PATHTRACING_RAY_TYPE, // SBT offset
             RAY_TYPE_COUNT,       // SBT stride
             PATHTRACING_RAY_TYPE, // miss SBT index
             u0, u1);

  _alpha += payload.alpha;
  _color += payload.color;
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
    uint32_t u0, u1;

    /* our per-ray data for this example. initialization matters! */
    render_pathtracing(camera.position, ray_dir, &rng_state, //
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

} // namespace optix7
} // namespace ovr
