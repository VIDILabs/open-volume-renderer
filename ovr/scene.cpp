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

#include "scene.h"

#include <colormap.h>

namespace vidi {
enum VoxelType {
  VOXEL_UINT8   = ovr::VALUE_TYPE_UINT8,
  VOXEL_INT8    = ovr::VALUE_TYPE_INT8,
  VOXEL_UINT16  = ovr::VALUE_TYPE_UINT16,
  VOXEL_INT16   = ovr::VALUE_TYPE_INT16,
  VOXEL_UINT32  = ovr::VALUE_TYPE_UINT32,
  VOXEL_INT32   = ovr::VALUE_TYPE_INT32,
  VOXEL_FLOAT   = ovr::VALUE_TYPE_FLOAT,
  VOXEL_DOUBLE  = ovr::VALUE_TYPE_DOUBLE,
  VOXEL_FLOAT2  = ovr::VALUE_TYPE_FLOAT2,
  VOXEL_FLOAT3  = ovr::VALUE_TYPE_FLOAT3,
  VOXEL_FLOAT4  = ovr::VALUE_TYPE_FLOAT4,
  VOXEL_DOUBLE2 = ovr::VALUE_TYPE_DOUBLE2,
  VOXEL_DOUBLE3 = ovr::VALUE_TYPE_DOUBLE3,
  VOXEL_DOUBLE4 = ovr::VALUE_TYPE_DOUBLE4,
  VOXEL_VOID    = ovr::VALUE_TYPE_VOID,
};
} // namespace vidi
#define VIDI_VOLUME_EXTERNAL_TYPE_ENUM
#include <vidi_volume_reader.h>

namespace ovr {

template<typename T>
array_1d_scalar_t
CreateArray1DScalar(const std::vector<T>& input, bool shared)
{
  array_1d_scalar_t output = std::make_shared<Array<1>>();

  output->type = value_type<T>();
  output->dims = (int)input.size();
  if (shared)
    output->set_data((void*)input.data());
  else
    output->allocate((void*)input.data());

  return output;
}
#define instantiate_create_array1dscalar_vector(T) \
  template array_1d_scalar_t CreateArray1DScalar<T>(const std::vector<T>& input, bool shared);
instantiate_create_array1dscalar_vector(uint8_t);
instantiate_create_array1dscalar_vector(int8_t);
instantiate_create_array1dscalar_vector(uint32_t);
instantiate_create_array1dscalar_vector(int32_t);
instantiate_create_array1dscalar_vector(float);
instantiate_create_array1dscalar_vector(double);
#undef instantiate_create_array1dscalar_vector

template<typename T>
array_1d_scalar_t
CreateArray1DScalar(const T* input, size_t len, bool shared)
{
  array_1d_scalar_t output = std::make_shared<Array<1>>();

  output->type = value_type<T>();
  output->dims = (int)len;
  if (shared)
    output->set_data((void*)input);
  else
    output->allocate((void*)input);

  return output;
}
#define instantiate_create_array1dscalar_pointer(T) \
  template array_1d_scalar_t CreateArray1DScalar<T>(const T* input, size_t len, bool shared);
instantiate_create_array1dscalar_pointer(uint8_t);
instantiate_create_array1dscalar_pointer(int8_t);
instantiate_create_array1dscalar_pointer(uint32_t);
instantiate_create_array1dscalar_pointer(int32_t);
instantiate_create_array1dscalar_pointer(float);
instantiate_create_array1dscalar_pointer(double);
#undef instantiate_create_array1dscalar_pointer

array_1d_float2_t CreateArray1DFloat2(const std::vector<vec2f>& input, bool shared) { return CreateArray1DScalar(input, shared); }
array_1d_float2_t CreateArray1DFloat2(const vec2f* input, size_t len,  bool shared) { return CreateArray1DScalar(input, len, shared); }

array_1d_float3_t CreateArray1DFloat3(const std::vector<vec3f>& input, bool shared) { return CreateArray1DScalar(input, shared); }
array_1d_float3_t CreateArray1DFloat3(const vec3f* input, size_t len,  bool shared) { return CreateArray1DScalar(input, len, shared); }

array_1d_float4_t CreateArray1DFloat4(const std::vector<vec4f>& input, bool shared) { return CreateArray1DScalar(input, shared); }
array_1d_float4_t CreateArray1DFloat4(const vec4f* input, size_t len,  bool shared) { return CreateArray1DScalar(input, len, shared); }

array_1d_float4_t
CreateColorMap(const std::string& name)
{
  if (colormap::has(name)) {
    const std::vector<vec4f>& arr = (const std::vector<vec4f>&)colormap::get(name);
    return CreateArray1DFloat4(arr, false);
  }
  else {
    throw std::runtime_error("Unexpected colormap name: " + name);
  }
}

array_3d_scalar_t
CreateArray3DScalarFromFile(const std::string& filename, vec3i dims, ValueType type, size_t offset, bool is_big_endian)
{
  // data geometry
  assert(dims.x > 0 && dims.y > 0 && dims.z > 0);

  // load data from file
  std::shared_ptr<char[]> data_buffer;
  {
    vidi::VolumeFileDesc desc;
    desc.dims.x = dims.x;
    desc.dims.y = dims.y;
    desc.dims.z = dims.z;
    desc.type = (vidi::VoxelType)type;
    desc.offset = offset;
    desc.is_big_endian = is_big_endian;
    data_buffer = vidi::read_volume_structured_regular(filename, desc);
  }

  // finalize
  array_3d_scalar_t output = std::make_shared<Array<3>>();
  output->dims = dims;
  output->type = type;
  output->acquire_data(std::move(data_buffer));

  return output;
}

} // namespace ovr
