//. ======================================================================== //
//. Copyright 2019-2020 Qi Wu                                                //
//.                                                                          //
//. Licensed under the Apache License, Version 2.0 (the "License");          //
//. you may not use this file except in compliance with the License.         //
//. You may obtain a copy of the License at                                  //
//.                                                                          //
//.     http://www.apache.org/licenses/LICENSE-2.0                           //
//.                                                                          //
//. Unless required by applicable law or agreed to in writing, software      //
//. distributed under the License is distributed on an "AS IS" BASIS,        //
//. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
//. See the License for the specific language governing permissions and      //
//. limitations under the License.                                           //
//. ======================================================================== //

#pragma once

#include <math_def.h>

#if defined(__cplusplus)
#include <cstring>
#include <vector>
#endif

namespace ovr {

// ------------------------------------------------------------------
// Scalar Definitions
// ------------------------------------------------------------------

enum ValueType {
  VALUE_TYPE_UINT8 = 100,
  VALUE_TYPE_INT8,

  VALUE_TYPE_UINT16 = 200,
  VALUE_TYPE_INT16,

  VALUE_TYPE_UINT32 = 300,
  VALUE_TYPE_INT32,

  VALUE_TYPE_FLOAT = 400,
  VALUE_TYPE_FLOAT2,
  VALUE_TYPE_FLOAT3,
  VALUE_TYPE_FLOAT4,

  VALUE_TYPE_DOUBLE = 500,
  VALUE_TYPE_DOUBLE2,
  VALUE_TYPE_DOUBLE3,
  VALUE_TYPE_DOUBLE4,

  VALUE_TYPE_VOID = 1000,
};

inline int
value_type_size(ValueType type)
{
  switch (type) {
  case VALUE_TYPE_UINT8: return sizeof(uint8_t);
  case VALUE_TYPE_INT8: return sizeof(int8_t);
  case VALUE_TYPE_UINT16: return sizeof(uint16_t);
  case VALUE_TYPE_INT16: return sizeof(int16_t);
  case VALUE_TYPE_UINT32: return sizeof(uint32_t);
  case VALUE_TYPE_INT32: return sizeof(int32_t);
  case VALUE_TYPE_FLOAT: return sizeof(float);
  case VALUE_TYPE_DOUBLE: return sizeof(double);

  case VALUE_TYPE_FLOAT2: return sizeof(vec2f);
  case VALUE_TYPE_FLOAT3: return sizeof(vec3f);
  case VALUE_TYPE_FLOAT4: return sizeof(vec4f);

  default: throw std::runtime_error("unknown type encountered");
  }
}

template<typename T>
ValueType
value_type();

#define ovr_instantiate_value_type_function(TYPE, type) \
  template<>                                            \
  inline ValueType value_type<type>()                   \
  {                                                     \
    return TYPE;                                        \
  }
/* clang-format off */
ovr_instantiate_value_type_function(VALUE_TYPE_UINT8,  uint8_t);
ovr_instantiate_value_type_function(VALUE_TYPE_INT8,   int8_t);
ovr_instantiate_value_type_function(VALUE_TYPE_UINT16, uint16_t);
ovr_instantiate_value_type_function(VALUE_TYPE_INT16,  int16_t);
ovr_instantiate_value_type_function(VALUE_TYPE_UINT32, uint32_t);
ovr_instantiate_value_type_function(VALUE_TYPE_INT32,  int32_t);
ovr_instantiate_value_type_function(VALUE_TYPE_FLOAT,  float);
ovr_instantiate_value_type_function(VALUE_TYPE_FLOAT2, vec2f);
ovr_instantiate_value_type_function(VALUE_TYPE_FLOAT3, vec3f);
ovr_instantiate_value_type_function(VALUE_TYPE_FLOAT4, vec4f);
ovr_instantiate_value_type_function(VALUE_TYPE_DOUBLE, double);
/* clang-format on */
// #undef ovr_instantiate_value_type_function

// ------------------------------------------------------------------
// Array Definitions
// ------------------------------------------------------------------

template<int DIM>
struct Array {
  enum { VECTOR_DIMENSION = DIM };

  ValueType type;
  math::vec_t<int, DIM> dims{ 0 };

  Array() {}
  ~Array() {}

  size_t size()
  {
    if (DIM == 0) return 0;
    size_t size = 1;
    for (int i = 0; i < DIM; i++)
      size *= dims[i];

    return size;
  }

  void allocate(void* ptr = nullptr)
  {
    owned_buffer.reset(new char[dims.long_product() * value_type_size(type)]);
    buffer = owned_buffer.get();
    if (ptr)
      memcpy(buffer, ptr, dims.long_product() * value_type_size(type));
  }

  void set_data(void* ptr)
  {
    owned_buffer.reset();
    buffer = (char*)ptr;
  }

  void acquire_data(std::shared_ptr<char[]> ptr)
  {
    owned_buffer = ptr;
    buffer = owned_buffer.get();
  }

  char* data()
  {
    return buffer;
  }

  char* data() const
  {
    return buffer;
  }

  template<typename T>
  T* data_typed()
  {
    if (type != value_type<T>())
      throw std::runtime_error("mismatched type!");

    return (T*)buffer;
  }

  template<typename T>
  T* data_typed() const
  {
    if (type != value_type<T>())
      throw std::runtime_error("mismatched type!");

    return (T*)buffer;
  }

private:
  char* buffer{ nullptr };
  std::shared_ptr<char[]> owned_buffer;
};

using Array1DScalar = Array<1>;
using Array1DFloat4 = Array<1>;
using Array3DScalar = Array<3>;
using array_1d_scalar_t = std::shared_ptr<Array1DScalar>;
using array_1d_float4_t = std::shared_ptr<Array1DFloat4>;
using array_3d_scalar_t = std::shared_ptr<Array3DScalar>;

#if defined(__cplusplus)

// ------------------------------------------------------------------
// Scene Definitions
// ------------------------------------------------------------------
struct Camera {
  // camera position - *from* where we are looking
  vec3f from;
  vec3f at; // which point we are looking *at*
  vec3f up; // up direction of the camera

  enum {
    PERSPECTIVE,
    ORTHOGRAPHIC,
  } type = PERSPECTIVE;

  // TODO:
  // affine3f	transform: additional world-space transform, overridden by motion.* arrays
  // float nearClip: 10-6 near clipping distance
  // vec2f imageStart: (0,0) start of image region (lower left corner)
  // vec2f imageEnd:   (1,1) end of image region (upper right corner)

  struct PerspectiveCamera {
    float fovy = 60.f;
    // TODO:
    // float aspect;
    // float apertureRadius;
    // float focusDistance;
    // float architectural;
    // float interpupillaryDistance;
  } perspective;

  struct OrthographicCamera {
    float	height = 200.f;
  } orthographic;
};

namespace scene {

struct TransferFunction {
  array_1d_float4_t color;
  array_1d_scalar_t opacity;
  vec2f value_range;
};

struct Volume {
  enum {
    STRUCTURED_REGULAR_VOLUME,
  } type;

  struct VolumeStructuredRegular {
    vec3f grid_origin  = vec3f(0, 0, 0);
    vec3f grid_spacing = vec3f(1, 1, 1);
    array_3d_scalar_t data;
  } structured_regular;
};

struct Geometry {
  enum {
    TRIANGLES_GEOMETRY,
  } type;

  struct GeometryTriangles { /* TODO */ } triangles;
};

struct Model {
  enum {
    VOLUMETRIC_MODEL,
    GEOMETRIC_MODEL,
  } type;

  struct VolumetricModel {
    TransferFunction transfer_function;
    Volume volume;
  } volume_model;

  struct GeometricModel {
    Geometry geometry;
  } geometry_model;
};

struct Instance {
  std::vector<Model> models;
  affine3f transform;
};

struct Light {
  enum {
    AMBIENT,
    DIRECTIONAL,
    POINT
  } type;

  float intensity = 1.f;
  vec3f color = vec3f(1.f);

  // definition of directional light
  struct {
    vec3f direction;
  } directional;

  // definition of point light
  struct {
    vec3f direction;
    vec3f position;
    float radius;
  } point;
};

struct Scene {
  std::vector<scene::Instance> instances;
  std::vector<scene::Light> lights;
  Camera camera;

  int ao_samples = 0;
  int spp = 1;
  /* volume rendering */
  float volume_sampling_rate = 1.f;
  /* path tracer */
  int roulette_path_length = 1;
  int max_path_length = 1;

  int use_dda = 0;
  bool parallel_view = false;
  bool simple_path_tracing = false;
};

} // namespace scene

using Scene = scene::Scene;

// ------------------------------------------------------------------
// Factory Functions
// ------------------------------------------------------------------

template<typename T>
array_1d_scalar_t
CreateArray1DScalar(const std::vector<T>& input, bool shared = false);
template<typename T>
array_1d_scalar_t
CreateArray1DScalar(const T* input, size_t len, bool shared = false);

array_1d_float4_t
CreateArray1DFloat4(const std::vector<vec4f>& input, bool shared = false);
array_1d_float4_t
CreateArray1DFloat4(const vec4f* input, size_t len, bool shared = false);
array_1d_float4_t
CreateColorMap(const std::string& name);

array_3d_scalar_t
CreateArray3DScalarFromFile(const std::string& filename, vec3i dims, ValueType type, size_t offset, bool is_big_endian);

#endif // defined(__cplusplus)

} // namespace ovr
