// ======================================================================== //
// Unit tests for ovr::Array<N> and the CreateArray1D* factory family, plus //
// the small free functions exposed in ovr/scene.h.                         //
// ======================================================================== //

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <scene.h>

#include <vector>

using namespace ovr;

TEST_CASE("Array<1>: allocate+set_data ownership switches") {
  Array<1> arr;
  arr.type = VALUE_TYPE_FLOAT;
  arr.dims = 4;

  std::vector<float> source{1.f, 2.f, 3.f, 4.f};
  arr.allocate(source.data());

  // `allocate` copies into owned storage -> pointers differ.
  CHECK(arr.data() != reinterpret_cast<char*>(source.data()));
  CHECK(arr.data_typed<float>()[0] == doctest::Approx(1.f));
  CHECK(arr.data_typed<float>()[3] == doctest::Approx(4.f));

  // Mutating the source must not affect the owned copy.
  source[0] = 99.f;
  CHECK(arr.data_typed<float>()[0] == doctest::Approx(1.f));

  // Switch to non-owning mode.
  arr.set_data(source.data());
  CHECK(arr.data() == reinterpret_cast<char*>(source.data()));
  CHECK(arr.data_typed<float>()[0] == doctest::Approx(99.f));
}

TEST_CASE("Array<1>::data_typed throws on mismatched type") {
  Array<1> arr;
  arr.type = VALUE_TYPE_FLOAT;
  arr.dims = 2;
  std::vector<float> src{0.f, 1.f};
  arr.allocate(src.data());

  CHECK_THROWS_AS(arr.data_typed<double>(), std::runtime_error);
  CHECK_THROWS_AS(arr.data_typed<int32_t>(), std::runtime_error);
  CHECK_NOTHROW(arr.data_typed<float>());
}

TEST_CASE("Array<1>::size returns product of dims") {
  Array<1> a1;
  a1.dims = 10;
  CHECK(a1.size() == 10);

  Array<3> a3;
  a3.dims = math::vec3i(2, 3, 4);
  CHECK(a3.size() == 24);
}

TEST_CASE("CreateArray1DScalar: shared vs owned semantics") {
  std::vector<float> src{10.f, 20.f, 30.f};

  auto shared = CreateArray1DScalar<float>(src, /*shared=*/true);
  CHECK(shared->size() == src.size());
  CHECK(shared->data() == reinterpret_cast<char*>(src.data()));

  auto owned = CreateArray1DScalar<float>(src, /*shared=*/false);
  CHECK(owned->size() == src.size());
  CHECK(owned->data() != reinterpret_cast<char*>(src.data()));
  CHECK(owned->data_typed<float>()[1] == doctest::Approx(20.f));
}

TEST_CASE("CreateArray1DFloat3 preserves vec3f payload") {
  std::vector<vec3f> src{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto arr = CreateArray1DFloat3(src, /*shared=*/false);
  CHECK(arr->size() == 3);
  CHECK(arr->data_typed<vec3f>()[2].x == doctest::Approx(7.f));
  CHECK(arr->data_typed<vec3f>()[2].z == doctest::Approx(9.f));
}

TEST_CASE("CreateColorMap: known name returns non-empty array") {
  // Staple colormaps shipped under extern/tfn/colormaps/. At least one of
  // these must resolve for any sane build.
  // Keys are of the form "<category>/<name>"; see extern/tfn/colormaps/.
  const char* candidates[] = {
    "diverging/coolwarm",
    "perceptual/viridis",
    "sequential2/hot",
    "diverging/bwr",
  };
  array_1d_float4_t arr;
  for (auto name : candidates) {
    try {
      arr = CreateColorMap(name);
      break;
    } catch (const std::runtime_error&) {
      continue;
    }
  }
  REQUIRE(arr);
  CHECK(arr->size() > 0);
}

TEST_CASE("CreateColorMap: unknown name throws") {
  CHECK_THROWS_AS(CreateColorMap("__definitely_not_a_colormap__"), std::runtime_error);
}

TEST_CASE("value_type_size: covers every declared type") {
  CHECK(value_type_size(VALUE_TYPE_UINT8)   == sizeof(uint8_t));
  CHECK(value_type_size(VALUE_TYPE_INT8)    == sizeof(int8_t));
  CHECK(value_type_size(VALUE_TYPE_UINT16)  == sizeof(uint16_t));
  CHECK(value_type_size(VALUE_TYPE_INT16)   == sizeof(int16_t));
  CHECK(value_type_size(VALUE_TYPE_UINT32)  == sizeof(uint32_t));
  CHECK(value_type_size(VALUE_TYPE_INT32)   == sizeof(int32_t));
  CHECK(value_type_size(VALUE_TYPE_UINT64)  == sizeof(uint64_t));
  CHECK(value_type_size(VALUE_TYPE_INT64)   == sizeof(int64_t));
  CHECK(value_type_size(VALUE_TYPE_FLOAT)   == sizeof(float));
  CHECK(value_type_size(VALUE_TYPE_DOUBLE)  == sizeof(double));
  CHECK(value_type_size(VALUE_TYPE_FLOAT3)  == sizeof(vec3f));
  CHECK(value_type_size(VALUE_TYPE_DOUBLE3) == 3 * sizeof(double));

  CHECK_THROWS_AS(value_type_size(static_cast<ValueType>(-1)), std::runtime_error);
}

TEST_CASE("value_type<T> template specializations") {
  CHECK(value_type<uint8_t>()  == VALUE_TYPE_UINT8);
  CHECK(value_type<float>()    == VALUE_TYPE_FLOAT);
  CHECK(value_type<vec3f>()    == VALUE_TYPE_FLOAT3);
  CHECK(value_type<double>()   == VALUE_TYPE_DOUBLE);
}

// ------------------------------------------------------------------ scene
namespace {

// Build a minimal programmatic scene with one structured-regular volume,
// a volumetric model, and a single instance - enough to exercise
// Scene::get_bounds() and parse_single_volume_scene().
scene::Scene make_minimal_scene()
{
  scene::Scene s;

  // Tiny 4^3 scalar volume.
  std::vector<float> voxels(4 * 4 * 4, 0.5f);
  scene::Volume vol;
  vol.type = scene::Volume::STRUCTURED_REGULAR_VOLUME;
  vol.structured_regular.grid_origin  = vec3f(0);
  vol.structured_regular.grid_spacing = vec3f(1);
  vol.structured_regular.data = std::make_shared<Array<3>>();
  vol.structured_regular.data->type = VALUE_TYPE_FLOAT;
  vol.structured_regular.data->dims = math::vec3i(4, 4, 4);
  vol.structured_regular.data->allocate(voxels.data());

  scene::Texture tex;
  tex.type = scene::Texture::VOLUME_TEXTURE;
  tex.volume.volume = vol;
  s.textures.push_back(tex);

  scene::Model model;
  model.type = scene::Model::VOLUMETRIC_MODEL;
  model.volume_model.volume_texture = 0;
  // leave transfer_function default-constructed

  scene::Instance inst;
  inst.models.push_back(model);
  inst.transform = affine3f(math::one);
  s.instances.push_back(inst);

  return s;
}

} // namespace

TEST_CASE("Scene::get_bounds: single 4^3 volume yields [0,4]^3") {
  auto s = make_minimal_scene();
  auto bounds = s.get_bounds();
  CHECK(bounds.lower.x == doctest::Approx(0.f));
  CHECK(bounds.lower.y == doctest::Approx(0.f));
  CHECK(bounds.lower.z == doctest::Approx(0.f));
  CHECK(bounds.upper.x == doctest::Approx(4.f));
  CHECK(bounds.upper.y == doctest::Approx(4.f));
  CHECK(bounds.upper.z == doctest::Approx(4.f));
}

TEST_CASE("parse_single_volume_scene: happy path returns the stored volume") {
  auto s = make_minimal_scene();
  const auto& v = parse_single_volume_scene(s);
  CHECK(v.type == scene::Volume::STRUCTURED_REGULAR_VOLUME);
  CHECK(v.structured_regular.data->dims.x == 4);
}

TEST_CASE("parse_single_volume_scene: throws when more than one instance") {
  auto s = make_minimal_scene();
  s.instances.push_back(s.instances[0]);
  CHECK_THROWS_AS(parse_single_volume_scene(s), std::runtime_error);
}

TEST_CASE("parse_single_volume_scene: throws when model is not volumetric") {
  auto s = make_minimal_scene();
  s.instances[0].models[0].type = scene::Model::GEOMETRIC_MODEL;
  CHECK_THROWS_AS(parse_single_volume_scene(s), std::runtime_error);
}

TEST_CASE("add_materials_for_isosurfaces: grows the materials vector") {
  auto s = make_minimal_scene();

  scene::TransferFunction tfn;
  std::vector<vec4f> colors{{1, 0, 0, 1}, {0, 1, 0, 1}};
  std::vector<float> opacities{0.f, 1.f};
  tfn.color   = CreateArray1DFloat4(colors, false);
  tfn.opacity = CreateArray1DScalar<float>(opacities, false);
  tfn.value_range = vec2f(0.f, 1.f);

  REQUIRE(s.materials.empty());
  auto mtls = s.add_materials_for_isosurfaces({0.25f, 0.75f}, tfn);
  CHECK(mtls.size() == 2);
  CHECK(s.materials.size() == 2);
}
