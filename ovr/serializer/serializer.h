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

#ifdef OVR_BUILD_SCENE_USD
Scene 
create_usda_scene(std::string filename);
#endif

Scene
create_json_scene(std::string filename);

TransferFunction
create_tfn(std::string filename);

} 
}

ovr::Scene
create_scene_default(std::string filename);

// namespace ovr::scene
