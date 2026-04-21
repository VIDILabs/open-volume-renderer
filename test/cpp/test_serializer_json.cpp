// ======================================================================== //
// Unit tests for the JSON serializer. Uses the generated synthetic scene  //
// fixture (see test/generate_synthetic_volume.cmake) so the test is      //
// hermetic and does not depend on an out-of-tree data file.              //
// ======================================================================== //

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <scene.h>
#include <serializer/serializer.h>

#include <cstdio>
#include <fstream>
#include <string>

#ifndef OVR_TEST_GENERATED_DIR
#  error "OVR_TEST_GENERATED_DIR must be defined by the test's CMakeLists.txt"
#endif

using namespace ovr;

namespace {
std::string generated(const char* name)
{
  return std::string(OVR_TEST_GENERATED_DIR) + "/" + name;
}
}

TEST_CASE("create_json_scene: loads synthetic DiVA scene") {
  auto scene = scene::create_json_scene(generated("synthetic_scene.json"));

  CHECK(scene.instances.size() == 1);
  REQUIRE(scene.instances[0].models.size() == 1);
  CHECK(scene.instances[0].models[0].type == scene::Model::VOLUMETRIC_MODEL);

  REQUIRE(!scene.textures.empty());
  const auto& tex = scene.textures[0];
  CHECK(tex.type == scene::Texture::VOLUME_TEXTURE);
  CHECK(tex.volume.volume.type == scene::Volume::STRUCTURED_REGULAR_VOLUME);
  CHECK(tex.volume.volume.structured_regular.data->dims.x == 32);
  CHECK(tex.volume.volume.structured_regular.data->dims.y == 32);
  CHECK(tex.volume.volume.structured_regular.data->dims.z == 32);

  // Camera was deserialized.
  CHECK(scene.camera.eye.z == doctest::Approx(96.0f).epsilon(1e-3));
  CHECK(scene.camera.type == scene::Camera::PERSPECTIVE);
}

TEST_CASE("create_json_scene: bounds span the full volume extent") {
  auto scene = scene::create_json_scene(generated("synthetic_scene.json"));
  auto bounds = scene.get_bounds();

  // Volume is 32^3 with grid_spacing=1 and grid_origin=0, so bounds should
  // cover roughly [0, 32]^3 (subject to texture-box extensions).
  CHECK(bounds.upper.x - bounds.lower.x >= doctest::Approx(32.f));
  CHECK(bounds.upper.y - bounds.lower.y >= doctest::Approx(32.f));
  CHECK(bounds.upper.z - bounds.lower.z >= doctest::Approx(32.f));
}

TEST_CASE("create_json_scene: malformed JSON surfaces as exception") {
  std::string path = std::string(OVR_TEST_GENERATED_DIR) + "/malformed.json";
  {
    std::ofstream f(path);
    f << "{ this is not valid json";
  }
  CHECK_THROWS(scene::create_json_scene(path));
  std::remove(path.c_str());
}

TEST_CASE("create_json_scene: missing file throws") {
  CHECK_THROWS(scene::create_json_scene(generated("definitely_absent.json")));
}

TEST_CASE("create_scene_default dispatches by extension") {
  // JSON should dispatch to create_json_scene and succeed.
  CHECK_NOTHROW(create_scene_default(generated("synthetic_scene.json")));

  // Unknown extension should throw (or at least not return a valid scene).
  std::string path = std::string(OVR_TEST_GENERATED_DIR) + "/unknown.xyz";
  { std::ofstream f(path); f << "dummy"; }
  CHECK_THROWS(create_scene_default(path));
  std::remove(path.c_str());
}
