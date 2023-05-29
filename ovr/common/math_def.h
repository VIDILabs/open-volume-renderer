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

#pragma once
#ifndef OVR_MATH_COMMON_H
#define OVR_MATH_COMMON_H

// ------------------------------------------------------------------
// Math Functions
// ------------------------------------------------------------------
#include <gdt/math/mat.h>
#include <gdt/math/vec.h>

#include <limits>

namespace ovr {

namespace math = gdt;

using vec2f = math::vec2f;
using vec2i = math::vec2i;
using vec3f = math::vec3f;
using vec3i = math::vec3i;
using vec4f = math::vec4f;
using vec4i = math::vec4i;
using box3f = math::box3f;

using range1f = math::range1f;

using affine3f = math::affine3f;
using linear3f = math::linear3f;
using math::clamp;
using math::max;
using math::min;
using math::cross;
using math::normalize;
using math::length;
using math::xfmNormal;
using math::xfmVector;
using math::xfmPoint;
using math::infty;

constexpr float float_large   = std::numeric_limits<float>::max();
constexpr float float_small   = std::numeric_limits<float>::min();
constexpr float float_epsilon = std::numeric_limits<float>::epsilon();

} // namespace ovr

#endif // OVR_MATH_COMMON_H
