// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "gdt/math/mat.h"

/*! \namespace glfwapp */
namespace glfwapp
{
  using namespace gdt;

  struct CameraFrame
  {
    CameraFrame(const float worldScale)
        : motionSpeed(worldScale)
    {
    }

    inline float computeStableEpsilon(float f) const
    {
      return abs(f) * float(1. / (1 << 21));
    }

    inline float computeStableEpsilon(const vec3f v) const
    {
      return max(max(computeStableEpsilon(v.x),
                     computeStableEpsilon(v.y)),
                 computeStableEpsilon(v.z));
    }

    inline vec3f get_from() const { return position; }
    inline vec3f get_position() const { return get_from(); }
    inline void set_position(const vec3f &p) { position = p; }

    inline vec3f get_accurate_up() const { return get_frame_y(); }
    inline vec3f get_up() const { return up_vector; }
    inline void set_up(const vec3f &up) { up_vector = up; }

    inline vec3f get_poi() const { return position - poi_distance * frame.vz; }

    inline float get_focal_length() const { return poi_distance; }
    inline void set_focal_length(const float &l) { poi_distance = l; }

    inline linear3f get_frame() const { return frame; }
    inline vec3f get_frame_x() const { return frame.vx; }
    inline vec3f get_frame_z() const { return frame.vz; }
    inline vec3f get_frame_y() const { return frame.vy; }

    /*! re-compute all orientation related fields from given
      'user-style' camera parameters */
    void setOrientation(/* camera origin    : */ const vec3f &origin,
                        /* point of interest: */ const vec3f &interest,
                        /* up-vector        : */ const vec3f &up)
    {
      poi_distance = length(interest - origin);
      position = origin;
      up_vector = up;

      /* negative because we use NEGATIZE z axis (Z) */
      frame.vz = (interest == origin) ? vec3f(0, 0, 1) : -normalize(interest - origin);
      /* right vector (X) */
      frame.vx = cross(up, frame.vz);
      frame.vx = normalize(frame.vx);
      /* fixed up (Y) */
      frame.vy = normalize(cross(frame.vz, frame.vx));

      /* rotation via quaternion */
      rotation = quat3f(frame.vx, frame.vy, frame.vz);

      modified = true;
    }

    // void rotate_frame_by_angle(float deg_u, float deg_v)
    // {
    //   const vec3f poi = get_poi();
    //   float rad_u = -M_PI / 180.f * deg_u;
    //   float rad_v = -M_PI / 180.f * deg_v;
    //   frame = linear3f::rotate(frame.vy, rad_u) * linear3f::rotate(frame.vx, rad_v) * frame;
    //   set_position(poi + get_focal_length() * get_frame_z());
    //   rotation = quat3f(frame.vx, frame.vy, frame.vz);
    //   modified = true;
    // }

    void rotate_frame(vec2f curr, vec2f prev, const float &speed)
    {
      /*! The callback functions receives the cursor position, measured in screen coordinates but
        relative to the top-left corner of the window content area. On platforms that provide it, 
        the full sub-pixel cursor position is passed on. */
      const auto mouse_curr_ball = screen_to_arcball(vec2f(curr.x, 1.f - curr.y));
      const auto mouse_prev_ball = screen_to_arcball(vec2f(prev.x, 1.f - prev.y));
      const auto delta = mouse_prev_ball * mouse_curr_ball;
      rotation = rotation * delta;
      frame = linear3f(rotation);
      modified = true;
    }

    /*! multiplier how fast the camera should move in world space
      for each unit of "user specifeid motion" (ie, pixel
      count). Initial value typically should depend on the world
      size, but can also be adjusted. This is actually something
      that should be more part of the manipulator widget(s), but
      since that same value is shared by multiple such widgets
      it's easiest to attach it to the camera here ...*/
    float motionSpeed{1.f};

    /*! gets set to true every time a manipulator changes the camera values */
    bool modified{true};

  private:
    linear3f frame{one};

    /*! distance to the 'point of interst' (poi); e.g., the point we will rotate around */
    float poi_distance{1.f};
    vec3f up_vector{0.f, 1.f, 0.f};
    vec3f position{0.f, -1.f, 0.f};

    quat3f rotation;

    quat3f
    screen_to_arcball(vec2f p)
    {
      p = p * 2.f - 1.f;
      const float dist = dot(p, p);

      // If radius < 0.5 we project the point onto the sphere
      if (dist <= 0.5f)
      {
        return quat3f(0.0f, normalize(vec3f(p.x, p.y, sqrt(1.f - dist))));
      }

      // otherwise we project the point onto the hyperbolic sheet
      else
      {

        return quat3f(0.0f, normalize(vec3f(p.x, p.y, 0.5f / sqrt(dist))));
      }
    }
  };
}
