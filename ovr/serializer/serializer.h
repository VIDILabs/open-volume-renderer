#pragma once

#include "ovr/scene.h"

// #include <3rdparty/json.hpp>
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

ovr::Scene
create_json_scene_vidi3d(json root, std::string workdir);

#ifdef OVR_BUILD_SCENE_USD
Scene 
create_json_scene_usda(std::string filename);
#endif

Scene
create_json_scene(std::string filename);

#ifdef OVR_BUILD_SCENE_USD
inline Scene
create_usda_scene(std::string filename)
{
  return create_json_scene_usda(filename);
}
#endif

Scene
create_scene(std::string filename);

} 
}
// namespace ovr::scene
