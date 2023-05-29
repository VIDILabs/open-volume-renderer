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

/**
 * Geometry Types Defined by the Application
 */
#ifndef OVR_VOLUME_H
#define OVR_VOLUME_H

#include "cuda_common.h"

#include <array>
#include <colormap.h>
#include <vector>

namespace ovr {

#if defined(__cplusplus)
namespace host {

#define ALIGN_SBT __align__(OPTIX_SBT_RECORD_ALIGNMENT)

/*! SBT record for a raygen program */
struct ALIGN_SBT RaygenRecord {
  ALIGN_SBT char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
  // just a dummy value - later examples will use more interesting data here
  void* data{};
};

/*! SBT record for a miss program */
struct ALIGN_SBT MissRecord {
  ALIGN_SBT char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
  // just a dummy value - later examples will use more interesting data here
  void* data{};
};

/*! SBT record for a hitgroup program */
struct ALIGN_SBT HitgroupRecord {
  ALIGN_SBT char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
  void* data{};
};

} // namespace host
#endif // defined(__cplusplus)

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
};

inline int
value_type_size(ValueType type)
{
  switch (type) {
  case VALUE_TYPE_UINT8:
  case VALUE_TYPE_INT8: return sizeof(char);
  case VALUE_TYPE_UINT16:
  case VALUE_TYPE_INT16: return sizeof(short);
  case VALUE_TYPE_UINT32:
  case VALUE_TYPE_INT32: return sizeof(int);
  case VALUE_TYPE_FLOAT: return sizeof(float);
  case VALUE_TYPE_DOUBLE: return sizeof(double);
  default: return 0;
  }
}

template<typename T>
ValueType
value_type();

#define instantiate_value_type_function(TYPE, type) \
  template<>                                        \
  inline ValueType value_type<type>()               \
  {                                                 \
    return TYPE;                                    \
  }
instantiate_value_type_function(VALUE_TYPE_UINT8, uint8_t);
instantiate_value_type_function(VALUE_TYPE_INT8, int8_t);
instantiate_value_type_function(VALUE_TYPE_UINT16, uint16_t);
instantiate_value_type_function(VALUE_TYPE_INT16, int16_t);
instantiate_value_type_function(VALUE_TYPE_UINT32, uint32_t);
instantiate_value_type_function(VALUE_TYPE_INT32, int32_t);
instantiate_value_type_function(VALUE_TYPE_FLOAT, float);
instantiate_value_type_function(VALUE_TYPE_DOUBLE, double);
#undef instantiate_value_type_function

// ------------------------------------------------------------------
// Array Definitions
// ------------------------------------------------------------------

template<int DIM, typename T>
struct ScalableArray {
  ValueType type;
  T lower{ 0 }; // value range for the texture
  T upper{ 0 }; // value range for the texture
  T scale{ 1 };
  math::vec_t<int, DIM> dims{ 0 };
  cudaTextureObject_t data{}; // the storage of the data on texture unit
};

using Array1D = ScalableArray<1, float>;
using Array2D = ScalableArray<2, float>;
using Array3D = ScalableArray<3, float>;

using Simple1DArrayData = ScalableArray<1, float>;
using RegularVolumeData = Array3D;

#if defined(__cplusplus)

namespace host {

template<typename T>
std::pair<T, T>
compute_scalar_minmax(void* _array, size_t count, size_t stride = 0);

template<typename T>
inline Array1D
CreateArray1D(std::vector<T> input)
{
  static_assert(std::is_scalar<T>::value, "expecting a scalar type");

  Array1D output;

  output.type = value_type<T>();
  output.dims = input.size();
  std::tie(output.lower, output.upper) = compute_scalar_minmax<T>(input.data(), input.size());
  auto array_handler = createCudaArray1D<T>(input.data(), input.size());
  output.data = createCudaTexture<T>(array_handler, true, cudaFilterModeLinear);

  return output;
}

template<>
inline Array1D
CreateArray1D<vec4f>(std::vector<vec4f> input)
{
  Array1D output;

  output.type = VALUE_TYPE_FLOAT4;
  output.dims = input.size();

  /* TODO value range is not properly calculated */
  // auto* dx = (float*)input.data();
  // auto* dy = dx + 1;
  // auto* dz = dx + 2;
  // auto* dw = dx + 3;
  // std::tie(output.lower.x, output.upper.x) = compute_scalar_minmax<float>(dx, input.size(), sizeof(vec4f));
  // std::tie(output.lower.x, output.upper.x) = compute_scalar_minmax<float>(dy, input.size(), sizeof(vec4f));
  // std::tie(output.lower.x, output.upper.x) = compute_scalar_minmax<float>(dz, input.size(), sizeof(vec4f));
  // std::tie(output.lower.x, output.upper.x) = compute_scalar_minmax<float>(dw, input.size(), sizeof(vec4f));

  auto array_handler = createCudaArray1D<float4>(input.data(), input.size());
  output.data = createCudaTexture<float4>(array_handler, true, cudaFilterModeLinear);

  return output;
}

inline Array1D
CreateColorMap(const std::string& name)
{
  if (colormap::data.count(name) > 0) {
    std::vector<vec4f>& arr = *((std::vector<vec4f>*)colormap::data.at(name));
    return CreateArray1D(arr);
  }
  else {
    throw std::runtime_error("Unexpected colormap name: " + name);
  }
}

template<typename T>
inline Array3D
CreateArray3D(void* input, vec3i dims)
{
  size_t elem_count = (size_t)dims.x * dims.y * dims.z;

  Array3D output;

  output.type = value_type<T>();
  output.dims = dims;
  std::tie(output.lower, output.upper) = compute_scalar_minmax<T>(input, elem_count);
  auto array_handler = createCudaArray3D<T>(input, dims);
  output.data = createCudaTexture<T>(array_handler, true);

  return output;
}

} // namespace host
#endif // defined(__cplusplus)

// ------------------------------------------------------------------
// Volume Definition
// ------------------------------------------------------------------

struct DeviceStructuredRegularVolume {
  RegularVolumeData volume;
  Simple1DArrayData colors;
  Simple1DArrayData alphas;
  float alpha_adjustment;
  float step;
};

// ------------------------------------------------------------------
//
// Host Functions
//
// ------------------------------------------------------------------
#if defined(__cplusplus)

namespace host {

struct HasSbtEquivalent {
private:
  CUDABuffer sbt_buffer;
  mutable char* sbt_data{ NULL };
  mutable size_t sbt_size{ 0 };

public:
  ~HasSbtEquivalent()
  {
    sbt_buffer.free();
  }

  void UpdateSbtData(cudaStream_t stream)
  {
    sbt_buffer.upload_async(sbt_data, sbt_size, stream);
  }

  template<typename T>
  void* CreateSbtPtr(const T& self, cudaStream_t stream)
  {
    sbt_data = (char*)&self;
    sbt_size = sizeof(T);

    /* create and upload to GPU */
    sbt_buffer.alloc_and_upload_async(&self, 1, stream);
    return (void*)sbt_buffer.d_pointer();
  }
};

struct InstantiableGeometry {
  // geometric transformation
  vec3f center = vec3f(0.f);
  vec3f scale = vec3f(1.f); // geometric scaling
  float rotate[9] = { 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f };

  /*! compute 3x4 transformation matrix */
  void transform(float transform[12]) const
  {
    transform[0] = rotate[0] * scale.x;
    transform[1] = rotate[1];
    transform[2] = rotate[2];
    transform[3] = center.x - 0.5f * scale.x;
    transform[4] = rotate[3];
    transform[5] = rotate[4] * scale.y;
    transform[6] = rotate[5];
    transform[7] = center.y - 0.5f * scale.y;
    transform[8] = rotate[6];
    transform[9] = rotate[7];
    transform[10] = rotate[8] * scale.z;
    transform[11] = center.z - 0.5f * scale.z;
  }
};

struct AabbGeometry {
private:
  // the AABBs for procedural geometries
  OptixAabb aabb{ 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };
  CUDABuffer aabbBuffer;
  CUDABuffer asBuffer; // buffer that keeps the (final, compacted) accel structure

public:
  OptixTraversableHandle buildas(OptixDeviceContext optixContext, cudaStream_t stream = 0)
  {
    // ==================================================================
    // aabb inputs
    // ==================================================================
    aabbBuffer.alloc_and_upload_async(&aabb, 1, stream);

    CUdeviceptr d_aabb = aabbBuffer.d_pointer();
    uint32_t f_aabb = 0;

    OptixBuildInput volumeInput = {}; // use one AABB input
    volumeInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
#if OPTIX_ABI_VERSION < 23
    auto& customPrimitiveArray = volumeInput.aabbArray;
#else
    auto& customPrimitiveArray = volumeInput.customPrimitiveArray;
#endif
    customPrimitiveArray.aabbBuffers = &d_aabb;
    customPrimitiveArray.numPrimitives = 1;
    customPrimitiveArray.strideInBytes = 0;
    customPrimitiveArray.primitiveIndexOffset = 0;
    customPrimitiveArray.flags = &f_aabb;
    customPrimitiveArray.numSbtRecords = 1;
    customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
    customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

    std::vector<OptixBuildInput> inputs = { volumeInput };
    return buildas_exec(optixContext, inputs, asBuffer);
  }
};

struct StructuredRegularVolume
  : public AabbGeometry
  , public InstantiableGeometry
  , protected HasSbtEquivalent {
public:
  DeviceStructuredRegularVolume self;

  std::vector<vec4f> _colorsData;
  std::vector<float> _alphasData;
  vec2f _valueRange;

  float base = 32.f;
  float rate = 128.f;

  void load_from_file(const std::string& filename,
                      vec3i dimensions,
                      ValueType value_type,
                      float value_min = 1,
                      float valuue_max = -1,
                      size_t offset = 0,
                      bool is_big_endian = false);

  /*! bind data */
  void set_volume(RegularVolumeData& v)
  {
    self.volume = v;
  }

  void set_transfer_function(Simple1DArrayData c, Simple1DArrayData a)
  {
    self.colors = c;
    self.alphas = a;
  }

  void set_transfer_function(const std::vector<float>& c, const std::vector<float>& o, const vec2f& r)
  {
    _colorsData.resize(c.size() / 3);
    for (int i = 0; i < _colorsData.size(); ++i) {
      _colorsData[i].x = c[3 * i + 0];
      _colorsData[i].y = c[3 * i + 1];
      _colorsData[i].z = c[3 * i + 2];
      _colorsData[i].w = 1.f;
    }
    _alphasData.resize(o.size() / 2);
    for (int i = 0; i < _alphasData.size(); ++i) {
      _alphasData[i] = o[2 * i + 1];
    }
    _valueRange = r;

    if (!_colorsData.empty())
      self.colors = CreateArray1D(_colorsData);
    if (!_alphasData.empty())
      self.alphas = CreateArray1D(_alphasData);
  }

  void set_sampling_rate(float r, float b = 0.f)
  {
    rate = r;
    if (b > 0)
      base = b;
  }

  void* get_sbt_pointer(cudaStream_t stream)
  {
    return CreateSbtPtr(self, stream); /* upload to GPU */
  }

  void commit(cudaStream_t stream)
  {
    self.alpha_adjustment = base / rate;
    self.step = 1.f / rate;
    UpdateSbtData(stream);
  }
};

} // namespace host
#endif // #if defined(__cplusplus)

} // namespace ovr
#endif // OVR_VOLUME_H
