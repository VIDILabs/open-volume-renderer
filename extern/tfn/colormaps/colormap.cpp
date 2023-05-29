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
#include <vector>
#include <string>
#include <unordered_map>
namespace colormap {
struct color_t { float r, g, b, a; };
extern const std::unordered_map<std::string, const std::vector<color_t>*> data;
extern const std::vector<std::string> name;
extern const std::vector<color_t> data_diverging_BrBG;
extern const std::vector<color_t> data_diverging_bwr;
extern const std::vector<color_t> data_diverging_coolwarm;
extern const std::vector<color_t> data_diverging_PiYG;
extern const std::vector<color_t> data_diverging_PRGn;
extern const std::vector<color_t> data_diverging_PuOr;
extern const std::vector<color_t> data_diverging_RdBu;
extern const std::vector<color_t> data_diverging_RdGy;
extern const std::vector<color_t> data_diverging_RdYlBu;
extern const std::vector<color_t> data_diverging_RdYlGn;
extern const std::vector<color_t> data_diverging_seismic;
extern const std::vector<color_t> data_diverging_Spectral;
extern const std::vector<color_t> data_perceptual_inferno;
extern const std::vector<color_t> data_perceptual_magma;
extern const std::vector<color_t> data_perceptual_plasma;
extern const std::vector<color_t> data_perceptual_viridis;
extern const std::vector<color_t> data_sequential_Blues;
extern const std::vector<color_t> data_sequential_BuGn;
extern const std::vector<color_t> data_sequential_BuPu;
extern const std::vector<color_t> data_sequential_GnBu;
extern const std::vector<color_t> data_sequential_Greens;
extern const std::vector<color_t> data_sequential_Greys;
extern const std::vector<color_t> data_sequential_Oranges;
extern const std::vector<color_t> data_sequential_OrRd;
extern const std::vector<color_t> data_sequential_PuBu;
extern const std::vector<color_t> data_sequential_PuBuGn;
extern const std::vector<color_t> data_sequential_PuRd;
extern const std::vector<color_t> data_sequential_Purples;
extern const std::vector<color_t> data_sequential_RdPu;
extern const std::vector<color_t> data_sequential_Reds;
extern const std::vector<color_t> data_sequential_YlGn;
extern const std::vector<color_t> data_sequential_YlGnBu;
extern const std::vector<color_t> data_sequential_YlOrBr;
extern const std::vector<color_t> data_sequential_YlOrRd;
extern const std::vector<color_t> data_sequential2_afmhot;
extern const std::vector<color_t> data_sequential2_autumn;
extern const std::vector<color_t> data_sequential2_binary;
extern const std::vector<color_t> data_sequential2_bone;
extern const std::vector<color_t> data_sequential2_cool;
extern const std::vector<color_t> data_sequential2_copper;
extern const std::vector<color_t> data_sequential2_gist_gray;
extern const std::vector<color_t> data_sequential2_gist_heat;
extern const std::vector<color_t> data_sequential2_gist_yarg;
extern const std::vector<color_t> data_sequential2_gray;
extern const std::vector<color_t> data_sequential2_hot;
extern const std::vector<color_t> data_sequential2_pink;
extern const std::vector<color_t> data_sequential2_spring;
extern const std::vector<color_t> data_sequential2_summer;
extern const std::vector<color_t> data_sequential2_winter;
extern const std::vector<color_t> data_sequential2_Wistia;
}
// definitions
const std::unordered_map<std::string, const std::vector<colormap::color_t>*>
  colormap::data = /* NOLINT(cert-err58-cpp) */
{
{ "diverging/BrBG", &colormap::data_diverging_BrBG },
{ "diverging/bwr", &colormap::data_diverging_bwr },
{ "diverging/coolwarm", &colormap::data_diverging_coolwarm },
{ "diverging/PiYG", &colormap::data_diverging_PiYG },
{ "diverging/PRGn", &colormap::data_diverging_PRGn },
{ "diverging/PuOr", &colormap::data_diverging_PuOr },
{ "diverging/RdBu", &colormap::data_diverging_RdBu },
{ "diverging/RdGy", &colormap::data_diverging_RdGy },
{ "diverging/RdYlBu", &colormap::data_diverging_RdYlBu },
{ "diverging/RdYlGn", &colormap::data_diverging_RdYlGn },
{ "diverging/seismic", &colormap::data_diverging_seismic },
{ "diverging/Spectral", &colormap::data_diverging_Spectral },
{ "perceptual/inferno", &colormap::data_perceptual_inferno },
{ "perceptual/magma", &colormap::data_perceptual_magma },
{ "perceptual/plasma", &colormap::data_perceptual_plasma },
{ "perceptual/viridis", &colormap::data_perceptual_viridis },
{ "sequential/Blues", &colormap::data_sequential_Blues },
{ "sequential/BuGn", &colormap::data_sequential_BuGn },
{ "sequential/BuPu", &colormap::data_sequential_BuPu },
{ "sequential/GnBu", &colormap::data_sequential_GnBu },
{ "sequential/Greens", &colormap::data_sequential_Greens },
{ "sequential/Greys", &colormap::data_sequential_Greys },
{ "sequential/Oranges", &colormap::data_sequential_Oranges },
{ "sequential/OrRd", &colormap::data_sequential_OrRd },
{ "sequential/PuBu", &colormap::data_sequential_PuBu },
{ "sequential/PuBuGn", &colormap::data_sequential_PuBuGn },
{ "sequential/PuRd", &colormap::data_sequential_PuRd },
{ "sequential/Purples", &colormap::data_sequential_Purples },
{ "sequential/RdPu", &colormap::data_sequential_RdPu },
{ "sequential/Reds", &colormap::data_sequential_Reds },
{ "sequential/YlGn", &colormap::data_sequential_YlGn },
{ "sequential/YlGnBu", &colormap::data_sequential_YlGnBu },
{ "sequential/YlOrBr", &colormap::data_sequential_YlOrBr },
{ "sequential/YlOrRd", &colormap::data_sequential_YlOrRd },
{ "sequential2/afmhot", &colormap::data_sequential2_afmhot },
{ "sequential2/autumn", &colormap::data_sequential2_autumn },
{ "sequential2/binary", &colormap::data_sequential2_binary },
{ "sequential2/bone", &colormap::data_sequential2_bone },
{ "sequential2/cool", &colormap::data_sequential2_cool },
{ "sequential2/copper", &colormap::data_sequential2_copper },
{ "sequential2/gist_gray", &colormap::data_sequential2_gist_gray },
{ "sequential2/gist_heat", &colormap::data_sequential2_gist_heat },
{ "sequential2/gist_yarg", &colormap::data_sequential2_gist_yarg },
{ "sequential2/gray", &colormap::data_sequential2_gray },
{ "sequential2/hot", &colormap::data_sequential2_hot },
{ "sequential2/pink", &colormap::data_sequential2_pink },
{ "sequential2/spring", &colormap::data_sequential2_spring },
{ "sequential2/summer", &colormap::data_sequential2_summer },
{ "sequential2/winter", &colormap::data_sequential2_winter },
{ "sequential2/Wistia", &colormap::data_sequential2_Wistia },
};
const std::vector<std::string> colormap::name = /* NOLINT(cert-err58-cpp) */
{
"diverging/BrBG",
"diverging/bwr",
"diverging/coolwarm",
"diverging/PiYG",
"diverging/PRGn",
"diverging/PuOr",
"diverging/RdBu",
"diverging/RdGy",
"diverging/RdYlBu",
"diverging/RdYlGn",
"diverging/seismic",
"diverging/Spectral",
"perceptual/inferno",
"perceptual/magma",
"perceptual/plasma",
"perceptual/viridis",
"sequential/Blues",
"sequential/BuGn",
"sequential/BuPu",
"sequential/GnBu",
"sequential/Greens",
"sequential/Greys",
"sequential/Oranges",
"sequential/OrRd",
"sequential/PuBu",
"sequential/PuBuGn",
"sequential/PuRd",
"sequential/Purples",
"sequential/RdPu",
"sequential/Reds",
"sequential/YlGn",
"sequential/YlGnBu",
"sequential/YlOrBr",
"sequential/YlOrRd",
"sequential2/afmhot",
"sequential2/autumn",
"sequential2/binary",
"sequential2/bone",
"sequential2/cool",
"sequential2/copper",
"sequential2/gist_gray",
"sequential2/gist_heat",
"sequential2/gist_yarg",
"sequential2/gray",
"sequential2/hot",
"sequential2/pink",
"sequential2/spring",
"sequential2/summer",
"sequential2/winter",
"sequential2/Wistia",
};
