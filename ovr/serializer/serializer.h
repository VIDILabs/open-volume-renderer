#pragma once

#include "ovr/scene.h"

// #include <3rdparty/json.hpp>
#include <json/json.hpp>

#include <fstream>
#include <string>

#include <filesystem> // C++17 is required by this project

namespace ovr {
using json = nlohmann::json;
}

namespace ovr::scene {

Scene
create_json_scene_diva(json root, std::string workdir);

ovr::Scene
create_json_scene_vidi3d(json root, std::string workdir);

#ifdef OVR_BUILD_SCENE_USD
Scene 
create_json_scene_usda(std::string filename);
#endif

inline Scene
create_json_scene(std::string filename)
{
  std::ifstream file(filename);
  std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  json root = json::parse(text, nullptr, true, true);

  std::filesystem::path p = filename;
  std::string workdir = p.remove_filename().string();
  workdir = workdir.empty() ? "." : workdir; // make sure workdir is never empty

  assert(root.is_object());
  if (root.contains("version")) {
    if (root["version"] == "DIVA") {
      return create_json_scene_diva(root, workdir);
    }
    else if (root["version"] == "VIDI3D") {
      return create_json_scene_vidi3d(root, workdir);
    }
    throw std::runtime_error("unknown JSON configuration format");
  }
  return create_json_scene_vidi3d(root, workdir);
}

#ifdef OVR_BUILD_SCENE_USD
inline Scene
create_usda_scene(std::string filename)
{
  return create_json_scene_usda(filename);
}
#endif

inline Scene
create_scene(std::string filename)
{
  const auto ext = filename.substr(filename.find_last_of(".") + 1);
  if (ext == "json") return create_json_scene(filename);
#ifdef OVR_BUILD_SCENE_USD
  if (ext == "usda") return create_usda_scene(filename);
#endif
  throw std::runtime_error("unknown scene format");
}

} // namespace ovr::scene
