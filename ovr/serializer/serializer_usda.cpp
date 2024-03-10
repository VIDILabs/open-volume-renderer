// (Qi): include headers belongs to the current project first
#include "serializer.h"

// (Qi): then include external library headers
// (Qi): use <> to include external headers
#include <pxr/base/tf/token.h>
#include <pxr/usd/sdf/childrenView.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>

// (Qi): finally include standard library headers
#include <filesystem> // C++17 (or Microsoft-specific implementation in C++14)

// (Qi): if this inclusion order doesn't work, there is something wrong with your headers and libraries.

PXR_NAMESPACE_USING_DIRECTIVE

namespace ovr::usda {

static const vec3f&
to_vec3f(const GfVec3f& input_vec3f) { return *(const vec3f*)&input_vec3f; }

static float
collect_float(UsdPrim ref, std::string token_name)
{
  float value;
  if (ref.GetAttribute(TfToken(token_name)).Get(&value)) {
    return value;
  }
  else {
    std::cerr << "[usd] float token '" << token_name << "' not found." << std::endl;
    throw std::runtime_error("float token '" + token_name + "' not found.");
    // exit(1); // (Qi): do not use exit, throw an error or return an error number instead
  }
}

static vec3f
collect_vec3f(UsdPrim ref, std::string token_name)
{
  GfVec3f vec;
  if (ref.GetAttribute(TfToken(token_name)).Get(&vec)) {
    return to_vec3f(vec);
  }
  else {
    std::cerr << "[usd] vec3f token '" << token_name << "' not found." << std::endl;
    throw std::runtime_error("vec3f token '" + token_name + "' not found.");
    // exit(1); // (Qi): do not use exit, throw an error or return an error number instead
  }
}

static /*inline*/ void // (Qi): why inline
import_camera_from_usda(Scene& scene, UsdPrim prim_ref)
{
  auto camera_data_prim = prim_ref.GetChild(TfToken("camera"));

  if (camera_data_prim) {
    scene.camera.eye = collect_vec3f(camera_data_prim, "from");
    scene.camera.at   = collect_vec3f(camera_data_prim, "at");
    scene.camera.up   = collect_vec3f(camera_data_prim, "up");
  }
  else {
    std::cerr << "[usd] no 'camera' Setting is found." << std::endl;
    throw std::runtime_error("[usd] no 'camera' setting is found.");
  }
}

// (Qi): static function makes the function visible only within the translation unit.
//       so we do not worry about potential name conflicts.
static /*inline*/ void
import_light_from_usda(Scene& scene, UsdPrim ref)
{
  std::cout << "[usd] collect lights: " << std::endl;
  auto light_prims = ref.GetChild(TfToken("light"));
  if (!light_prims) {
    throw std::runtime_error("[usd] didn't find 'light' in usda file.");
  }

  for (const std::string& name : TfToStringVector(light_prims.GetChildrenNames())) {
    std::cout << "[usd] ... light: " << name << std::endl;

    Light light;

    if (name == "ambient") {
      light.type = Light::AMBIENT;

      const auto prims = light_prims.GetChild(TfToken("ambient"));
      for (const auto& n : TfToStringVector(prims.GetChildrenNames())) {
        const auto p = prims.GetChild(TfToken(n));
        light.intensity = collect_float(p, "intensity");
        light.color     = collect_vec3f(p, "color");
        scene.lights.push_back(light);
      }
    }
    else if (name == "directional") {
      light.type = Light::DIRECTIONAL;

      const auto prims = light_prims.GetChild(TfToken("directional"));
      for (const auto& n : TfToStringVector(prims.GetChildrenNames())) {
        const auto p = prims.GetChild(TfToken(n));
        light.intensity = collect_float(p, "intensity");
        light.color     = collect_vec3f(p, "color");
        light.directional.direction = collect_vec3f(p, "direction");
        scene.lights.push_back(light);
      }
    }
    else {
      throw std::runtime_error("[usd] unknown light type.");
    }
    
  }

  std::cout << "[usd] collected " << TfToStringVector(light_prims.GetChildrenNames()).size() << " lights" << std::endl;
}

static std::string
dirname(const std::string& fname)
{
  const auto str = std::filesystem::path(fname).remove_filename().string();
  return str.empty() ? "." : str;
}

} // namespace usda

namespace ovr::scene {

Scene
create_usda_scene(std::string filename)
{
  using namespace ovr::usda;

  std::cout << "[usd] loading USDA file for scene path: " << filename << std::endl;

  // 'stage' needs to be alive throughout the entire function  
  const UsdStageRefPtr stage = UsdStage::Open(filename);
  const UsdPrim ref = stage->GetPrimAtPath(SdfPath("/scene"));

  std::string data_path;

  const auto volume = ref.GetChild(TfToken("volume"));
  if (volume) {
    if (volume.GetAttribute(TfToken("data_path")).Get(&data_path)) {
      std::cout << "[usd] inputted volume data path: " << data_path << std::endl;
    }
    else {
      throw std::runtime_error("[usd] didn't find volume 'data_path'");
    }
  }
  else {
    throw std::runtime_error("[usd] didn't find 'volume'");
  }

  int use_dda;
  bool parallel_view;
  bool simple_path_tracing;
  const auto rendering_setting = ref.GetChild(TfToken("rendering"));
  if (rendering_setting){
    if (rendering_setting.GetAttribute(TfToken("use_dda")).Get(&use_dda)) {
      if (use_dda > 2) {
        throw std::runtime_error("[usd] 'use_dda' should be only using '0' for No DDA, '1' for single layer DDA, and '2' for two layers DDA");
      }
      std::cout << "[usd] use dda: " << use_dda << std::endl;
    }
    else {
      throw std::runtime_error("[usd] didn't find rendering 'use_dda'");
    }

    if (rendering_setting.GetAttribute(TfToken("parallel_view")).Get(&parallel_view)) {
      std::cout << "[usd] use parallel view: " << parallel_view << std::endl;
    }
    else {
      throw std::runtime_error("[usd] didn't find rendering 'parallel_view'");
    }

    if (rendering_setting.GetAttribute(TfToken("simple_path_tracing")).Get(&simple_path_tracing)) {
      std::cout << "[usd] use simple path tracing: " << simple_path_tracing << std::endl;
    }
    else {
      throw std::runtime_error("[usd] didn't find rendering 'simple_path_tracing'");
    }
  }

  std::filesystem::path path(data_path);
  if (path.is_absolute()) {
    // arriving here if path = "C:/xxx".
    // arriving here in non-windows if path = "/xxx".
    data_path = data_path;
  }
  else if (path.is_relative()) {
    // arriving here if path = "".
    // arriving here if path = "xxx".
    // arriving here in windows if path = "/xxx".
    data_path = dirname(filename) + "/" + data_path;
  }
  std::cout << "[usd] constructed volume data path: " << data_path << std::endl;

  auto scene = create_json_scene(data_path);
  import_camera_from_usda(scene, ref);
  std::cout << "[usd] updated camera" << std::endl;

  scene.lights.clear();
  import_light_from_usda(scene, ref);
  std::cout << "[usd] updated lights, total # of lights: " << scene.lights.size() << std::endl;

  scene.use_dda = use_dda;
  scene.parallel_view = parallel_view;
  scene.simple_path_tracing = simple_path_tracing;
  return scene;
}

} // namespace ovr::scene
