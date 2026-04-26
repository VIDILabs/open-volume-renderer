// ======================================================================== //
// Unit tests for ovr::count_tfn - the small helper in renderer.h that    //
// finds the "active" transfer function across the scene's instances and  //
// textures.                                                              //
// ======================================================================== //

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <renderer.h>
#include <scene.h>

#include <vector>

using namespace ovr;

namespace {

scene::TransferFunction make_tfn(float r, float g, float b)
{
  scene::TransferFunction t;
  std::vector<vec4f> colors{{r, g, b, 1.f}};
  std::vector<float> opacity{1.f};
  t.color   = CreateArray1DFloat4(colors, false);
  t.opacity = CreateArray1DScalar<float>(opacity, false);
  t.value_range = vec2f(0.f, 1.f);
  return t;
}

scene::Scene make_scene_with_volume()
{
  scene::Scene s;

  scene::Volume vol;
  vol.type = scene::Volume::STRUCTURED_REGULAR_VOLUME;
  vol.structured_regular.grid_origin  = vec3f(0);
  vol.structured_regular.grid_spacing = vec3f(1);
  vol.structured_regular.data = std::make_shared<Array<3>>();
  vol.structured_regular.data->type = VALUE_TYPE_FLOAT;
  vol.structured_regular.data->dims = math::vec3i(2);
  std::vector<float> v(8, 0.f);
  vol.structured_regular.data->allocate(v.data());

  scene::Texture tex;
  tex.type = scene::Texture::VOLUME_TEXTURE;
  tex.volume.volume = vol;
  s.textures.push_back(tex);

  return s;
}

} // namespace

TEST_CASE("count_tfn: empty scene returns zero") {
  scene::Scene s;
  scene::TransferFunction found;
  CHECK(count_tfn(s, found) == 0);
}

TEST_CASE("count_tfn: TF on the volume model counts as one") {
  auto s = make_scene_with_volume();

  scene::Model model;
  model.type = scene::Model::VOLUMETRIC_MODEL;
  model.volume_model.volume_texture = 0;
  model.volume_model.transfer_function = make_tfn(1, 0, 0);

  scene::Instance inst;
  inst.models.push_back(model);
  s.instances.push_back(inst);

  scene::TransferFunction found;
  CHECK(count_tfn(s, found) == 1);
  REQUIRE(found.color);
  CHECK(found.color->data_typed<vec4f>()[0].x == doctest::Approx(1.f));
}

TEST_CASE("count_tfn: TF on a texture entry counts as one") {
  auto s = make_scene_with_volume();

  scene::Texture tfn_tex;
  tfn_tex.type = scene::Texture::TRANSFER_FUNCTION_TEXTURE;
  tfn_tex.transfer_function.transfer_function = make_tfn(0, 1, 0);
  tfn_tex.transfer_function.volume_texture = 0;
  s.textures.push_back(tfn_tex);

  scene::TransferFunction found;
  CHECK(count_tfn(s, found) == 1);
  REQUIRE(found.color);
  CHECK(found.color->data_typed<vec4f>()[0].y == doctest::Approx(1.f));
}

TEST_CASE("count_tfn: both sources present sums to two") {
  auto s = make_scene_with_volume();

  scene::Model model;
  model.type = scene::Model::VOLUMETRIC_MODEL;
  model.volume_model.volume_texture = 0;
  model.volume_model.transfer_function = make_tfn(1, 0, 0);
  scene::Instance inst;
  inst.models.push_back(model);
  s.instances.push_back(inst);

  scene::Texture tfn_tex;
  tfn_tex.type = scene::Texture::TRANSFER_FUNCTION_TEXTURE;
  tfn_tex.transfer_function.transfer_function = make_tfn(0, 1, 0);
  tfn_tex.transfer_function.volume_texture = 0;
  s.textures.push_back(tfn_tex);

  scene::TransferFunction found;
  CHECK(count_tfn(s, found) == 2);
}
