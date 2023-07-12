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

#include <fstream>

namespace ovr {

namespace {

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

template<size_t Size>
inline void
swap_bytes(void* data)
{
  char* p = reinterpret_cast<char*>(data);
  char* q = p + Size - 1;
  while (p < q)
    std::swap(*(p++), *(q--));
}

template<>
inline void
swap_bytes<1>(void*)
{
}

template<>
inline void
swap_bytes<2>(void* data)
{
  char* p = reinterpret_cast<char*>(data);
  std::swap(p[0], p[1]);
}

template<>
inline void
swap_bytes<4>(void* data)
{
  char* p = reinterpret_cast<char*>(data);
  std::swap(p[0], p[3]);
  std::swap(p[1], p[2]);
}

template<>
inline void
swap_bytes<8>(void* data)
{
  char* p = reinterpret_cast<char*>(data);
  std::swap(p[0], p[7]);
  std::swap(p[1], p[6]);
  std::swap(p[2], p[5]);
  std::swap(p[3], p[4]);
}

template<typename T>
inline void
swap_bytes(T* data)
{
  swap_bytes<sizeof(T)>(reinterpret_cast<void*>(data));
}

inline void
reverse_byte_order(char* data, size_t elemCount, size_t elemSize)
{
  switch (elemSize) {
  case 1: break;
  case 2:
    for (size_t i = 0; i < elemCount; ++i)
      swap_bytes<2>(&data[i * elemSize]);
    break;
  case 4:
    for (size_t i = 0; i < elemCount; ++i)
      swap_bytes<4>(&data[i * elemSize]);
    break;
  case 8:
    for (size_t i = 0; i < elemCount; ++i)
      swap_bytes<8>(&data[i * elemSize]);
    break;
  default: assert(false);
  }
}

} // namespace

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
  if (colormap::data.count(name) > 0) {
    // auto& _arr = *colormap::data.at(name);
    // std::vector<vec4f> arr;
    // for (int i = 0; i < _arr.size(); ++i) {
    //   arr.push_back(vec4f(_arr[i].r, _arr[i].g, _arr[i].b, _arr[i].a));
    // }
    std::vector<vec4f>& arr = *((std::vector<vec4f>*)colormap::data.at(name));
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

  size_t elem_count = (size_t)dims.x * dims.y * dims.z;

  size_t elem_size = // clang-format off
    (type == VALUE_TYPE_UINT8  || type == VALUE_TYPE_INT8 ) ? sizeof(uint8_t)  :
    (type == VALUE_TYPE_UINT16 || type == VALUE_TYPE_INT16) ? sizeof(uint16_t) :
    (type == VALUE_TYPE_UINT32 || type == VALUE_TYPE_INT32 || type == VALUE_TYPE_FLOAT) ? 
    sizeof(uint32_t) : sizeof(double);
  // clang-format on

  size_t data_size = elem_count * elem_size;

  // load the data
  std::shared_ptr<char[]> data_buffer;

  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  if (ifs.fail()) // cannot open the file
  {
    throw std::runtime_error("Cannot open the file: " + filename);
  }

  ifs.seekg(0, std::ios::end);
  size_t file_size = ifs.tellg();
  if (file_size < offset + data_size) // file size does not match data size
  {
    throw std::runtime_error("File size does not match data size");
  }
  ifs.seekg(offset, std::ios::beg);

  try {
    data_buffer.reset(new char[data_size]);
  }
  catch (std::bad_alloc&) // memory allocation failed
  {
    throw std::runtime_error("Cannot allocate memory for the data");
  }

  // read data
  ifs.read(data_buffer.get(), data_size);
  if (ifs.fail()) // reading data failed
  {
    throw std::runtime_error("Cannot read the file");
  }

  // reverse byte-order if necessary
  const bool reverse = (is_big_endian && elem_size > 1);
  if (reverse) {
    reverse_byte_order(data_buffer.get(), elem_count, elem_size);
  }

  ifs.close();

  // finalize
  array_3d_scalar_t output = std::make_shared<Array<3>>();
  output->dims = dims;
  output->type = type;
  output->acquire_data(std::move(data_buffer));

  return output;
}

} // namespace ovr
