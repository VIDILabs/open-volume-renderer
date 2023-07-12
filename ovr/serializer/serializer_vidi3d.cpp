#include "serializer.h"

// #include "devices/vidi3d/dictionary.h"
// #include <v3d/Serializer/CameraSerializer.h>
// #include <v3d/Serializer/TransferFunctionSerializer.h>
// #include <v3d/Serializer/VectorSerializer.h>
// #include <v3d/Serializer/VolumeSerializer.h>
namespace tfn {
typedef ovr::math::vec2f vec2f;
typedef ovr::math::vec2i vec2i;
typedef ovr::math::vec3f vec3f;
typedef ovr::math::vec3i vec3i;
typedef ovr::math::vec4f vec4f;
typedef ovr::math::vec4i vec4i;
} // namespace tfn
#define TFN_MODULE_EXTERNAL_VECTOR_TYPES
#include "tfn/core.h"

// JSON I/O
#include "tfn/json.h"
using json = nlohmann::json;

#include <filesystem> // C++17 (or Microsoft-specific implementation in C++14)

// ------------------------------------------------------------------
// ------------------------------------------------------------------


#define VOLUME "volume"
#define TRANSFER_FUNCTION "transferFunction"
#define SCALAR_MAPPING_RANGE_UNNORMALIZED "scalarMappingRangeUnnormalized"
#define SCALAR_MAPPING_RANGE "scalarMappingRange"
#define FORMAT "format"
#define REGULAR_GRID_RAW_BINARY "REGULAR_GRID_RAW_BINARY"
#define FILE_NAME "fileName"
#define DIMENSIONS "dimensions"
#define SCALES "scales"
#define TYPE "type"
#define OFFSET "offset"
#define FILE_UPPER_LEFT "fileUpperLeft"
#define ENDIAN "endian"
#define EYE "eye"
#define CENTER "center"
#define UP "up"
#define FOVY "fovy"
#define CAMERA "camera"
#define DATA_SOURCE "dataSource"
#define VIEW "view"
#define POSITION "position"
#define DIFFUSE "diffuse"
#define LIGHT_SOURCE "lightSource"
#define ADDITIONAL_LIGHT_SOURCES "additionalLightSources"
#define SAMPLING_DISTANCE "sampleDistance"
#define DIRECTIONAL_LIGHT "DIRECTIONAL_LIGHT"

namespace ovr {

NLOHMANN_JSON_SERIALIZE_ENUM(ValueType, {
  { ValueType::VALUE_TYPE_INT8, "BYTE" },
  { ValueType::VALUE_TYPE_UINT8, "UNSIGNED_BYTE" },
  { ValueType::VALUE_TYPE_INT16, "SHORT" },
  { ValueType::VALUE_TYPE_UINT16, "UNSIGNED_SHORT" },
  { ValueType::VALUE_TYPE_INT32, "INT" },
  { ValueType::VALUE_TYPE_UINT32, "UNSIGNED_INT" },
  { ValueType::VALUE_TYPE_FLOAT, "FLOAT" },
  { ValueType::VALUE_TYPE_DOUBLE, "DOUBLE" },
}); // clang-format on

}

#define assert_throw(x, msg) { if (!(x)) throw std::runtime_error(msg); }

namespace ovr::vidi3d {
  
enum Endianness { OVR_LITTLE_ENDIAN, OVR_BIG_ENDIAN };
NLOHMANN_JSON_SERIALIZE_ENUM(Endianness, {
  { OVR_LITTLE_ENDIAN, "LITTLE_ENDIAN" },
  { OVR_BIG_ENDIAN, "BIG_ENDIAN" },
}); // clang-format on

#define define_vector_serialization(T)                      \
   NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ovr::math::vec2##T, x, y);       \
   NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ovr::math::vec3##T, x, y, z);    \
   NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ovr::math::vec4##T, x, y, z, w); \
   NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ovr::math::range1##T, minimum, maximum);
define_vector_serialization(i);
define_vector_serialization(f);
#undef define_vector_serialization

template<typename ScalarT>
inline ScalarT
scalar_from_json(const json& in);

#define define_scalar_serialization(T) template<> inline T scalar_from_json<T>(const json& in) { return in.get<T>(); }
define_scalar_serialization(std::string);
define_scalar_serialization(bool);
define_scalar_serialization(int64_t);
define_scalar_serialization(uint64_t);
#ifdef __APPLE__
define_scalar_serialization(size_t);
#endif
define_scalar_serialization(double);

template<typename ScalarT/*, typename std::enable_if_t<!std::is_arithmetic<ScalarT>::value> = true*/>
inline ScalarT
scalar_from_json(const json& in)
{
  ScalarT v;
  from_json(in, v);
  return v;
}

vec3f
scalar_from_json(const json& in)
{
  if (!in.contains("r") || !in.contains("g") || !in.contains("b")) return vec3f(0.0, 0.0, 0.0);
  return vec3f(in["r"].get<float>(), in["g"].get<float>(), in["b"].get<float>());
}

template<typename ScalarT>
inline ScalarT
scalar_from_json(const json& in, const std::string& key)
{
  assert_throw(in.is_object(), "has to be a JSON object");
  assert_throw(in.contains(key), "incorrect key: " + key);
  return scalar_from_json<ScalarT>(in[key]);
}

template<typename ScalarT>
inline ScalarT
scalar_from_json(const json& in, const std::string& key, const ScalarT& value)
{
  assert_throw(in.is_object(), "has to be a JSON object");
  if (in.contains(key)) {
    return scalar_from_json<ScalarT>(in[key]);
  }
  else {
    return value;
  }
}

inline vec2f
range_from_json(json jsrange)
{
  if (!jsrange.contains("minimum") || !jsrange.contains("maximum")) return vec2f(0.0, 0.0);
  return vec2f(jsrange["minimum"].get<float>(), jsrange["maximum"].get<float>());
}

// using namespace ovr::math;
// using namespace ovr::scene;

static bool
file_exists_test(std::string name, const std::string& dir, std::string& out)
{
  std::filesystem::path path(name);
  if (path.is_relative()) path = dir / path;
  std::ifstream f(path.c_str());
  out = path.string();
  return f.good();
}

static std::string
valid_filename(const json& in, std::string dir, const std::string& key)
{
  std::string file;
  if (in.contains(key)) {
    auto& js = in[key];
    if (js.is_array()) {
      for (auto& s : js) {
        if (file_exists_test(s.get<std::string>(), dir, file)) {
          return file;
        }
      }
      throw std::runtime_error("Cannot find volume file.");
    }
    else {
      if (file_exists_test(js.get<std::string>(), dir, file)) {
        return file;
      }
      throw std::runtime_error("Cannot find volume file.");
    }
  }
  else {
    throw std::runtime_error("JSON key '" + key + "' doesnot exist");
  }
}

ovr::scene::TransferFunction
create_scene_tfn(const json& jsview, ValueType type)
{
  ovr::scene::TransferFunction ret{};

  const auto& jstfn = jsview[VOLUME][TRANSFER_FUNCTION];
  const auto& jsvolume = jsview[VOLUME];

  tfn::TransferFunctionCore tf;
  tfn::loadTransferFunction(jstfn, tf);

  auto* table = (vec4f*)tf.data();
  std::vector<vec4f> color(tf.resolution());
  std::vector<float> alpha(tf.resolution());
  for (int i = 0; i < tf.resolution(); ++i) {
    auto rgba = table[i];
    color[i] = vec4f(rgba.xyz(), 1.f);
    alpha[i] = rgba.w;
  }
  if (alpha[0] < 0.01f) alpha[0] = 0.f;
  if (alpha[tf.resolution()-1] < 0.01f) alpha[tf.resolution()-1] = 0.f;

  ret.color   = CreateArray1DFloat4(color);
  ret.opacity = CreateArray1DScalar(alpha);

  if (jsvolume.contains(SCALAR_MAPPING_RANGE_UNNORMALIZED)) {
    auto r = range_from_json(jsvolume[SCALAR_MAPPING_RANGE_UNNORMALIZED]);
    ret.value_range.x = r.x;
    ret.value_range.y = r.y;
  }

  /* try it ... */
  else if (jsvolume.contains(SCALAR_MAPPING_RANGE)) {
    auto r = range_from_json(jsvolume[SCALAR_MAPPING_RANGE]);
    switch (type) {
    case VALUE_TYPE_UINT8:
      ret.value_range.x = std::numeric_limits<uint8_t>::max() * r.x;
      ret.value_range.y = std::numeric_limits<uint8_t>::max() * r.y;
      break;
    case VALUE_TYPE_INT8:
      ret.value_range.x = std::numeric_limits<int8_t>::max() * r.x;
      ret.value_range.y = std::numeric_limits<int8_t>::max() * r.y;
      break;
    case VALUE_TYPE_UINT16:
      ret.value_range.x = std::numeric_limits<uint16_t>::max() * r.x;
      ret.value_range.y = std::numeric_limits<uint16_t>::max() * r.y;
      break;
    case VALUE_TYPE_INT16:
      ret.value_range.x = std::numeric_limits<int16_t>::max() * r.x;
      ret.value_range.y = std::numeric_limits<int16_t>::max() * r.y;
      break;
    case VALUE_TYPE_UINT32:
      ret.value_range.x = std::numeric_limits<uint32_t>::max() * r.x;
      ret.value_range.y = std::numeric_limits<uint32_t>::max() * r.y;
      break;
    case VALUE_TYPE_INT32:
      ret.value_range.x = std::numeric_limits<int32_t>::max() * r.x;
      ret.value_range.y = std::numeric_limits<int32_t>::max() * r.y;
      break;
    case VALUE_TYPE_FLOAT:
    case VALUE_TYPE_DOUBLE:
      ret.value_range.x = r.x;
      ret.value_range.y = r.y;
      break;
    default: throw std::runtime_error("unknown data type");
    }
  }

  else {
    /* calculate the volume value range ... */
    throw std::runtime_error("unknown data range");
  }

  return ret;
}

ovr::scene::Volume
create_scene_volume(const json& jsdata, std::string workdir)
{
  scene::Volume volume{};

  const auto format = jsdata[FORMAT].get<std::string>();

  if (format == REGULAR_GRID_RAW_BINARY) {
    auto filename = valid_filename(jsdata, workdir, FILE_NAME);
    auto dims = scalar_from_json<vec3i>(jsdata[DIMENSIONS]);
    auto type = scalar_from_json<ValueType>(jsdata[TYPE]);
    auto offset = scalar_from_json<size_t>(jsdata, OFFSET, 0);
    auto flipped = scalar_from_json<bool>(jsdata, FILE_UPPER_LEFT, false);
    auto is_big_endian = scalar_from_json<Endianness>(jsdata, ENDIAN, OVR_LITTLE_ENDIAN) == OVR_BIG_ENDIAN;

    volume.type = ovr::scene::Volume::STRUCTURED_REGULAR_VOLUME;
    volume.structured_regular.data = CreateArray3DScalarFromFile(filename, dims, type, offset, is_big_endian);
    volume.structured_regular.grid_origin = vec3f(0, 0, 0);

    if (jsdata.contains(SCALES)) {
      auto scales = scalar_from_json<vec3f>(jsdata[SCALES]);
      volume.structured_regular.grid_spacing = scales;
    }
  }
  else {
    throw std::runtime_error("data type unimplemented");
  }

  return volume;
}

Camera
create_scene_camera(const json& jsview)
{
  const auto& jscamera = jsview[CAMERA];

  Camera camera;

  camera.from = scalar_from_json<vec3f>(jscamera[EYE]);
  camera.at = scalar_from_json<vec3f>(jscamera[CENTER]);
  camera.up = scalar_from_json<vec3f>(jscamera[UP]);
  camera.perspective.fovy = jscamera[FOVY].get<float>();

  return camera;
}

} // namespace ovr::vidi3d

// ------------------------------------------------------------------
// ------------------------------------------------------------------

namespace ovr::scene {

using namespace ovr::vidi3d;

Scene
create_json_scene_vidi3d(json root, std::string workdir)
{
  Instance instance;
  instance.transform = affine3f::translate(vec3f(0));

  for (auto& ds : root[DATA_SOURCE]) {
    auto volume = vidi3d::create_scene_volume(ds, workdir);

    auto tfn = vidi3d::create_scene_tfn(root[VIEW], volume.structured_regular.data->type);

    if (!root[VIEW][VOLUME].contains(SCALAR_MAPPING_RANGE_UNNORMALIZED)) {
      auto type = scalar_from_json<ValueType>(ds[TYPE]);
      if (type != VALUE_TYPE_FLOAT && type != VALUE_TYPE_DOUBLE) {
        std::cerr << "[vidi3d] An unnormalized value range cannot be found for "
                     "transfer function, incorrect results can be produced."
                  << std::endl;
      }
    }

    Model model;
    model.type = Model::VOLUMETRIC_MODEL;
    model.volume_model.volume = volume;
    model.volume_model.transfer_function = tfn;

    instance.models.push_back(model);
  }

  Scene scene;
  scene.instances.push_back(instance);

  if (root[VIEW].contains(LIGHT_SOURCE)) {
    assert(root[VIEW][LIGHT_SOURCE][TYPE] == DIRECTIONAL_LIGHT);
    Light light;
    light.type = Light::DIRECTIONAL;
    light.directional.direction = scalar_from_json<vec3f>(root[VIEW][LIGHT_SOURCE][POSITION]);
    light.color = scalar_from_json(root[VIEW][LIGHT_SOURCE][DIFFUSE]);
    scene.lights.push_back(light);
  }
  
  if (root[VIEW].contains(ADDITIONAL_LIGHT_SOURCES)) {
    for (auto& li : root[VIEW][ADDITIONAL_LIGHT_SOURCES]) {
      assert(li[TYPE] == DIRECTIONAL_LIGHT);
      Light light;
      light.type = Light::DIRECTIONAL;
      light.directional.direction = scalar_from_json<vec3f>(li[POSITION]);
      light.color = scalar_from_json(li[DIFFUSE]);
      scene.lights.push_back(light);
    }
  }

  if (scene.lights.empty()) {
    Light light;
    light.type = Light::DIRECTIONAL;
    light.directional.direction = vec3f(1, 1, 1);
    light.color = vec3f(1, 1, 1);
    scene.lights.push_back(light);
  }

  std::cout << "scene.lights = " << scene.lights.size() << std::endl;

  scene.camera = vidi3d::create_scene_camera(root[VIEW]);
  scene.volume_sampling_rate = 1.f / (float)scalar_from_json<double>(root[VIEW][VOLUME][SAMPLING_DISTANCE]);
  if (scene.volume_sampling_rate > 1) {
    std::cout << "scene.volume_sampling_rate = " << scene.volume_sampling_rate << std::endl;
  }

  return scene;
}

} // namespace ovr::scene
