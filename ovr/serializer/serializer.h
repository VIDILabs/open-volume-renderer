#pragma once

#include "ovr/scene.h"

#include <json/json.hpp>

#include <fstream>
#include <string>

namespace ovr {
using json = nlohmann::json;
}

namespace ovr { 
namespace scene {

Scene
create_json_scene_diva(json root, std::string workdir);

Scene
create_json_scene_vidi(json root, std::string workdir);

#ifdef OVR_BUILD_USD
Scene 
create_usda_scene(std::string filename);
#endif

Scene
create_json_scene(std::string filename);

TransferFunction
create_tfn(std::string filename);

Scene
create_scene_visualization(
    Volume main_volume, TransferFunction main_tfn,
    Volume contours_volume, std::vector<float> contours_values,
    Volume contours_cmap_volume, TransferFunction contours_cmap_tfn,
    std::vector<vec4f> points_data, std::vector<vec4f> points_color,
    Volume points_cmap_volume, TransferFunction points_cmap_tfn
);

}
}

ovr::Scene
create_scene_default(std::string filename);

// namespace ovr::scene
