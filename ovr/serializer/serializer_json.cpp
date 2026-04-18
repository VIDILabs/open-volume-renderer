#include "serializer.h"

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

// #include <filesystem> // C++17 (or Microsoft-specific implementation in C++14)

// ------------------------------------------------------------------
// ------------------------------------------------------------------

#define MINIMUM "minimum"
#define MAXIMUM "maximum"
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
#define POSITIONS "positions"
#define DIFFUSE "diffuse"
#define LIGHT_SOURCE "lightSource"
#define ADDITIONAL_LIGHT_SOURCES "additionalLightSources"
#define SAMPLING_DISTANCE "sampleDistance"
#define DIRECTIONAL_LIGHT "DIRECTIONAL_LIGHT"
#define ISOSUFRACES "isosurfaces"
#define ISOVALUES "isovalues"
#define VISIBLE "visible"
#define DATA_ID "dataId"
#define AO_SAMPLES "aoSamples"
#define TRANSFER_FUNCTION_DATA_ID "transferFunctionDataId"
#define SPHERES "spheres"

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

namespace ovr::diva {}

namespace ovr::vidi {
  
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
  if (!jsrange.contains(MINIMUM)  || !jsrange.contains(MAXIMUM)) { return vec2f(0.0, 0.0); }
  return vec2f (jsrange[MINIMUM].get<float>(), jsrange[MAXIMUM].get<float>());
}

static bool
file_exists_test(std::string name)
{
  std::ifstream f(name.c_str());
  return f.good();
}

static bool
file_exists_test(std::string name, const std::string& dir, std::string& out)
{
  if (file_exists_test(name)) {
    out = name;
    return true;
  }
  else if (file_exists_test(dir + "/" + name)) {
    out = dir + "/" + name;
    return true;
  }
  else if (file_exists_test(dir + "\\" + name)) {
    out = dir + "/" + name;
    return true;
  }
  return false;
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
create_scene_tfn(const json& jsvolume, ValueType type)
{
  ovr::scene::TransferFunction ret{};

  // const auto& jsvolume = jsview[VOLUME];
  const auto& jstfn = jsvolume[TRANSFER_FUNCTION];

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
    // we assume the OpenGL data normalization rule being applied here:
    // -- https://www.khronos.org/opengl/wiki/Normalized_Integer
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
    // throw std::runtime_error("unknown data range");
    ret.value_range.x =  std::numeric_limits<float>::max();
    ret.value_range.y = -std::numeric_limits<float>::max();
  }

  return ret;
}

ovr::scene::Volume
create_scene_volume(const json& jsdata, std::string workdir)
{
  ovr::scene::Volume volume{};

  const auto format = jsdata[FORMAT].get<std::string>();

  if (format == REGULAR_GRID_RAW_BINARY) {
    auto filename = valid_filename(jsdata, workdir, FILE_NAME);
    auto dims = scalar_from_json<vec3i>(jsdata[DIMENSIONS]);
    auto type = scalar_from_json<ValueType>(jsdata[TYPE]);
    auto offset = scalar_from_json<size_t>(jsdata, OFFSET, 0);
    auto flipped = scalar_from_json<bool>(jsdata, FILE_UPPER_LEFT, false);
    auto is_big_endian = scalar_from_json<Endianness>(jsdata, ENDIAN, OVR_LITTLE_ENDIAN) == OVR_BIG_ENDIAN;

    volume.type = ovr::scene::Volume::STRUCTURED_REGULAR_VOLUME;
    volume.structured_regular.data = CreateArray3DScalarFromFile(filename.c_str(), dims, type, offset, is_big_endian);

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

ovr::scene::Camera
create_scene_camera(const json& jsview)
{
  const auto& jscamera = jsview[CAMERA];

  ovr::scene::Camera camera;

  camera.eye = scalar_from_json<vec3f>(jscamera[EYE]);
  camera.at = scalar_from_json<vec3f>(jscamera[CENTER]);
  camera.up = scalar_from_json<vec3f>(jscamera[UP]);
  camera.perspective.fovy = jscamera[FOVY].get<float>();

  return camera;
}

} // namespace ovr::vidi

using namespace ovr::vidi;

// ------------------------------------------------------------------
// ------------------------------------------------------------------

ovr::scene::TransferFunction
ovr::scene::create_tfn(std::string filename)
{
  std::ifstream file(filename);
  std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  json root = json::parse(text, nullptr, true, true);
  return create_scene_tfn(root[VIEW][VOLUME], ovr::ValueType::VALUE_TYPE_DOUBLE);
}

ovr::scene::Scene
ovr::scene::create_json_scene_diva(json root, std::string workdir)
{
  throw std::runtime_error("unimplemented by now");
}

ovr::scene::Scene
ovr::scene::create_json_scene_vidi(json root, std::string workdir)
{

  std::vector<scene::Volume> volumes;
  int ao_samples = 0;

  // Parse All Volumes
  for (auto& ds : root[DATA_SOURCE]) {
    auto volume = create_scene_volume(ds, workdir);
    if (!root[VIEW][VOLUME].contains(SCALAR_MAPPING_RANGE_UNNORMALIZED)) {
      auto type = scalar_from_json<ValueType>(ds[TYPE]);
      if (type != VALUE_TYPE_FLOAT && type != VALUE_TYPE_DOUBLE) {
        std::cerr << "[vidi3d] An unnormalized value range cannot be found for "
                     "transfer function, incorrect results can be produced."
                  << std::endl;
      }
    }
    volumes.push_back(volume);
  }

  if (!root[VIEW].contains(VOLUME)) {
    throw std::runtime_error("tag \"view/volume\" found in the scene");
  }

  // -------- // 

  Volume main_volume;
  TransferFunction main_tfn;

  std::vector<float> contours_values;
  Volume contours_volume;
  Volume contours_cmap_volume;
  TransferFunction contours_cmap_tfn;

  std::vector<vec4f> points_data;
  std::vector<vec4f> points_color;
  Volume points_cmap_volume;
  TransferFunction points_cmap_tfn;

  // -------- // 

  auto visible = [] (const json& js) {
    return js.contains(VISIBLE) ? js[VISIBLE].get<bool>() : true;
  };

  auto dataid = [] (const json& js) {
    return js.contains(DATA_ID) ? (js[DATA_ID].get<int>() - 1) : 0;
  };

  const bool main_volume_visible = visible(root[VIEW][VOLUME]);
  const int main_volume_ID = dataid(root[VIEW][VOLUME]);
  main_tfn = create_scene_tfn(root[VIEW][VOLUME], volumes[main_volume_ID].structured_regular.data->type);
  if (main_volume_visible) {
    main_volume = volumes[main_volume_ID];
    // std::cout << "main volume visible, data ID = " << main_volume_ID << std::endl;
  }

  // Add Isosurfaces
  if (root[VIEW].contains(ISOSUFRACES)) {
    const bool isosurfaces_visible = visible(root[VIEW][ISOSUFRACES]);
    const int ID = dataid(root[VIEW][ISOSUFRACES]);
    const int TFN_ID = root[VIEW][ISOSUFRACES].contains(TRANSFER_FUNCTION_DATA_ID) 
      ? (root[VIEW][ISOSUFRACES][TRANSFER_FUNCTION_DATA_ID].get<int>() - 1) 
      : ID;

    if (isosurfaces_visible) {
      const std::vector<float> values = root[VIEW][ISOSUFRACES][ISOVALUES];
      contours_values = values;
      contours_volume = volumes[ID];
      contours_cmap_volume = volumes[TFN_ID];
      contours_cmap_tfn = root[VIEW][ISOSUFRACES].contains(TRANSFER_FUNCTION)
        ? create_scene_tfn(root[VIEW][ISOSUFRACES], volumes[TFN_ID].structured_regular.data->type)
        : main_tfn;
    }

    ao_samples = root[VIEW][ISOSUFRACES].contains(AO_SAMPLES) 
      ? root[VIEW][ISOSUFRACES][AO_SAMPLES].get<int>() 
      : 0;
  }

  // Add Spheres
  if (root[VIEW].contains(SPHERES)) {
    const bool spheres_visible = visible(root[VIEW][SPHERES]);
    if (spheres_visible) {
      auto& positions = root[VIEW][SPHERES][POSITIONS];
      size_t n_spheres = positions.size();
      points_data.resize(n_spheres);
      for (size_t i = 0; i < n_spheres; ++i) {
        auto xyz = scalar_from_json<vec3f>(positions[i]);
        float radius = 1.f;
        points_data[i] = vec4f(xyz, radius);
      }
    }
  }

  Scene scene = create_scene_visualization(
    main_volume, main_tfn,
    contours_volume, contours_values,
    contours_cmap_volume, contours_cmap_tfn,
    points_data, points_color,
    points_cmap_volume, points_cmap_tfn
  );

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

  scene.ao_samples = ao_samples;
  scene.camera = create_scene_camera(root[VIEW]);
  scene.volume_sampling_rate = 1.f / (float)scalar_from_json<double>(root[VIEW][VOLUME][SAMPLING_DISTANCE]);

  return scene;
}

ovr::scene::Scene
ovr::scene::create_json_scene(std::string filename)
{
  std::ifstream file(filename);
  std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  json root = json::parse(text, nullptr, true, true);

  // find the base path from filename using pure c++ 11
  std::string workdir = filename.substr(0, filename.find_last_of("/\\"));
  workdir = workdir.empty() ? "." : workdir; // make sure workdir is never empty

  assert(root.is_object());
  if (root.contains("version")) {
    if (root["version"] == "DIVA") {
      return create_json_scene_diva(root, workdir);
    }
    else {
      throw std::runtime_error("unknown scene format: " + root["version"].get<std::string>());
    }
  }
  return create_json_scene_vidi(root, workdir);
}

ovr::scene::Scene
create_scene_default(std::string filename)
{
  const auto ext = filename.substr(filename.find_last_of(".") + 1);
  if (ext == "json") return ovr::scene::create_json_scene(filename);
#ifdef OVR_BUILD_USD
  if (ext == "usda") return ovr::scene::create_usda_scene(filename);
#endif
  throw std::runtime_error("unknown scene format");
}

static ovr::scene::TransferFunction
copy_scene_tfn(const ovr::scene::TransferFunction& tfn)
{
  using namespace ovr;
  ovr::scene::TransferFunction ret{};
  std::vector<vec4f> color(tfn.color->size());
  std::vector<float> opacity(tfn.opacity->size());
  for (int i = 0; i < tfn.color->size(); ++i) {
    color[i] = tfn.color->data_typed<vec4f>()[i];
  }
  for (int i = 0; i < tfn.opacity->size(); ++i) {
    opacity[i] = tfn.opacity->data_typed<float>()[i];
  }
  ret.color   = CreateArray1DFloat4(color);
  ret.opacity = CreateArray1DScalar(opacity);
  ret.value_range = tfn.value_range;
  return ret;
}

ovr::scene::Scene
ovr::scene::create_scene_visualization(
  Volume main_volume, TransferFunction main_tfn,
  Volume contours_volume, std::vector<float> contours_values,
  Volume contours_cmap_volume, TransferFunction contours_cmap_tfn,
  std::vector<vec4f> points_data, std::vector<vec4f> points_color,
  Volume points_cmap_volume, TransferFunction points_cmap_tfn
)
{
  Scene scene;

  Instance instance;
  instance.transform = affine3f::translate(vec3f(0));

  // 1. Main Volume
  if (main_volume.type != Volume::INVALID) {
    if (main_volume.type != Volume::STRUCTURED_REGULAR_VOLUME) {
      throw std::runtime_error("main volume is not a structured regular volume");
    }

    Texture texture;
    texture.type = Texture::VOLUME_TEXTURE;
    texture.volume.volume = main_volume;
    scene.textures.push_back(texture);
    const int32_t main_volume_tex = scene.textures.size() - 1;

    // Create a model for the main volume
    Model model;
    model.type = Model::VOLUMETRIC_MODEL;
    model.volume_model.volume_texture = main_volume_tex;
    model.volume_model.transfer_function = main_tfn;
    instance.models.push_back(model);
  }

  // 2. Contours
  if (!contours_values.empty()) {
    if (contours_volume.type == Volume::INVALID) {
      throw std::runtime_error("contours volume is invalid");
    }

    // Create a model for the contours
    Texture texture;
    memset(&texture, 0, sizeof(texture));
    texture.type = Texture::VOLUME_TEXTURE;
    texture.volume.volume = contours_volume;
    scene.textures.push_back(texture);
    const int32_t contours_volume_tex = scene.textures.size() - 1;

    Model model;
    model.type = ovr::scene::Model::GEOMETRIC_MODEL;
    model.geometry_model.geometry.type = ovr::scene::Geometry::ISOSURFACE_GEOMETRY;
    model.geometry_model.geometry.isosurfaces.volume_texture = contours_volume_tex;
    model.geometry_model.geometry.isosurfaces.isovalues = CreateArray1DScalar(contours_values);

    // Create a volume OBJ material for the contours
    if (contours_cmap_volume.type != Volume::INVALID) {

      contours_cmap_tfn = copy_scene_tfn(contours_cmap_tfn);
      for (int i = 0; i < contours_cmap_tfn.opacity->size(); ++i) {
        ((float*)contours_cmap_tfn.opacity->data())[i] = 1.f;
      }

      memset(&texture, 0, sizeof(texture));
      texture.type = Texture::VOLUME_TEXTURE;
      texture.volume.volume = contours_cmap_volume;
      scene.textures.push_back(texture);
      const int32_t contours_cmap_volume_tex = scene.textures.size() - 1;

      memset(&texture, 0, sizeof(texture));
      texture.type = Texture::TRANSFER_FUNCTION_TEXTURE;
      texture.transfer_function.transfer_function = contours_cmap_tfn;
      texture.transfer_function.volume_texture = contours_cmap_volume_tex;
      scene.textures.push_back(texture);
      const int32_t contours_cmap_tex = scene.textures.size() - 1;

      Material material;
      material.type = Material::OBJ_MATERIAL;
      material.obj.map_kd = contours_cmap_tex;
      scene.materials.push_back(material);
      const int32_t contours_cmap_mtl = scene.materials.size() - 1;

      // Set the material for the contours
      model.geometry_model.mtl = contours_cmap_mtl;
      // model.geometry_model.mtls = scene.add_materials_for_isosurfaces(contours_values, contours_cmap_tfn);
    }

    // Finalize
    instance.models.push_back(model);
  }

  // 3. Points
  if (!points_data.empty()) {
    const size_t n_points = points_data.size();
    std::vector<vec3f> points_position(n_points);
    std::vector<float> points_radius(n_points);
    for (int i = 0; i < n_points; ++i) {
      points_position[i] = points_data[i].xyz();
      points_radius[i] = points_data[i].w;
    }

    Model model;
    model.type = ovr::scene::Model::GEOMETRIC_MODEL;
    model.geometry_model.geometry.type = ovr::scene::Geometry::SPHERES_GEOMETRY;

    auto& spheres = model.geometry_model.geometry.spheres;
    spheres.sphere.position = CreateArray1DFloat3(points_position);
    spheres.sphere.radius = CreateArray1DScalar(points_radius);

    // Finalize
    instance.models.push_back(model);
  }

  // 4. Finalize
  scene.instances.push_back(instance);
  return scene;
}
