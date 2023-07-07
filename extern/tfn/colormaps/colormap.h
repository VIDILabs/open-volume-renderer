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
// diverging/RdYlGn
// diverging/RdBu
// diverging/RdYlBu
// diverging/bwr
// diverging/Spectral
// diverging/RdGy
// diverging/seismic
// diverging/coolwarm
// diverging/PRGn
// diverging/PuOr
// diverging/PiYG
// perceptual/magma
// perceptual/inferno
// perceptual/viridis
// perceptual/plasma
// sequential/Purples
// sequential/PuBuGn
// sequential/Oranges
// sequential/Blues
// sequential/YlGn
// sequential/PuBu
// sequential/GnBu
// sequential/Greens
// sequential/PuRd
// sequential/BuPu
// sequential/Greys
// sequential/YlOrBr
// sequential/RdPu
// sequential/YlOrRd
// sequential/Reds
// sequential/YlGnBu
// sequential/BuGn
// sequential/OrRd
// sequential2/hot
// sequential2/Wistia
// sequential2/gist_gray
// sequential2/bone
// sequential2/winter
// sequential2/pink
// sequential2/binary
// sequential2/autumn
// sequential2/spring
// sequential2/gist_yarg
// sequential2/copper
// sequential2/gray
// sequential2/afmhot
// sequential2/cool
// sequential2/gist_heat
// sequential2/summer

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
