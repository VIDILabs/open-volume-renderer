//
// Automatically generated file, do not modify.
//
// ======================================================================== //
// Copyright 2019-2020 Qi Wu                                                //
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
//
// clang-format off
//
// available colormap keys:
//
// diverging/BrBG
// diverging/bwr
// diverging/coolwarm
// diverging/PiYG
// diverging/PRGn
// diverging/PuOr
// diverging/RdBu
// diverging/RdGy
// diverging/RdYlBu
// diverging/RdYlGn
// diverging/seismic
// diverging/Spectral
// perceptual/inferno
// perceptual/magma
// perceptual/plasma
// perceptual/viridis
// sequential/Blues
// sequential/BuGn
// sequential/BuPu
// sequential/GnBu
// sequential/Greens
// sequential/Greys
// sequential/Oranges
// sequential/OrRd
// sequential/PuBu
// sequential/PuBuGn
// sequential/PuRd
// sequential/Purples
// sequential/RdPu
// sequential/Reds
// sequential/YlGn
// sequential/YlGnBu
// sequential/YlOrBr
// sequential/YlOrRd
// sequential2/afmhot
// sequential2/autumn
// sequential2/binary
// sequential2/bone
// sequential2/cool
// sequential2/copper
// sequential2/gist_gray
// sequential2/gist_heat
// sequential2/gist_yarg
// sequential2/gray
// sequential2/hot
// sequential2/pink
// sequential2/spring
// sequential2/summer
// sequential2/winter
// sequential2/Wistia
#ifndef TFN_COLORMAP_H
#define TFN_COLORMAP_H
#include <vector>
#include <array>
#include <string>
#include <unordered_map>
namespace colormap {
struct color_t { float r, g, b, a; };
extern const std::unordered_map<std::string, const std::vector<color_t>*> data;
extern const std::vector<std::string> name;
}
#endif // TFN_COLORMAP_H