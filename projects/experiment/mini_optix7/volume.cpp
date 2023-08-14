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

#include "volume.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <fstream>

namespace ovr {
namespace host {

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

template<size_t Size>
inline void
swapBytes(void* data)
{
  char* p = reinterpret_cast<char*>(data);
  char* q = p + Size - 1;
  while (p < q)
    std::swap(*(p++), *(q--));
}

template<>
inline void
swapBytes<1>(void*)
{
}

template<>
inline void
swapBytes<2>(void* data)
{
  char* p = reinterpret_cast<char*>(data);
  std::swap(p[0], p[1]);
}

template<>
inline void
swapBytes<4>(void* data)
{
  char* p = reinterpret_cast<char*>(data);
  std::swap(p[0], p[3]);
  std::swap(p[1], p[2]);
}

template<>
inline void
swapBytes<8>(void* data)
{
  char* p = reinterpret_cast<char*>(data);
  std::swap(p[0], p[7]);
  std::swap(p[1], p[6]);
  std::swap(p[2], p[5]);
  std::swap(p[3], p[4]);
}

template<typename T>
inline void
swapBytes(T* data)
{
  swapBytes<sizeof(T)>(reinterpret_cast<void*>(data));
}

inline void
reverseByteOrder(char* data, size_t elemCount, size_t elemSize)
{
  switch (elemSize) {
  case 1: break;
  case 2:
    for (size_t i = 0; i < elemCount; ++i)
      swapBytes<2>(&data[i * elemSize]);
    break;
  case 4:
    for (size_t i = 0; i < elemCount; ++i)
      swapBytes<4>(&data[i * elemSize]);
    break;
  case 8:
    for (size_t i = 0; i < elemCount; ++i)
      swapBytes<8>(&data[i * elemSize]);
    break;
  default: assert(false);
  }
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

template<typename IType, typename OType>
std::shared_ptr<char[]>
convert_volume(std::shared_ptr<char[]> idata, size_t size)
{
  std::shared_ptr<char[]> odata;
  odata.reset(new char[size * sizeof(OType)]);

  tbb::parallel_for(size_t(0), size, [&](size_t idx) {
    auto* i = (IType*)&idata[idx * sizeof(IType)];
    auto* o = (OType*)&odata[idx * sizeof(OType)];
    *o = static_cast<OType>(*i);
  });

  return odata;
}

template<typename T>
std::pair<T, T>
compute_scalar_minmax(void* _array, size_t count, size_t stride)
{
  static_assert(std::is_scalar<T>::value, "expecting a scalar type");

  if (stride == 0)
    stride = sizeof(T);

  T* array = (T*)_array;
  auto value = [array, stride](size_t index) -> T {
    const auto begin = (const uint8_t*)array;
    const auto curr = (T*)(begin + index * stride);
    return static_cast<T>(*curr);
  };

  T init;

  init = std::numeric_limits<T>::lowest();
  T actual_max = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, count), init,
    [value](const tbb::blocked_range<size_t>& r, T v) -> T {
      for (auto i = r.begin(); i != r.end(); ++i)
        v = std::max(v, value(i));
      return v;
    },
    [](T x, T y) -> T { return std::max(x, y); });

  init = std::numeric_limits<T>::max();
  T actual_min = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, count), init,
    [value](const tbb::blocked_range<size_t>& r, T v) -> T {
      for (auto i = r.begin(); i != r.end(); ++i)
        v = std::min(v, value(i));
      return v;
    },
    [](T x, T y) -> T { return std::min(x, y); });

  return std::make_pair(actual_min, actual_max);
}

template std::pair<float, float>
compute_scalar_minmax(void* _array, size_t count, size_t stride);

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

void
StructuredRegularVolume::load_from_file(const std::string& filename,
                                        vec3i data_dims,
                                        ValueType data_type,
                                        float data_value_min,
                                        float data_value_max,
                                        size_t data_offset,
                                        bool data_is_big_endian)
{
  // data geometry
  assert(data_dims.x > 0 && data_dims.y > 0 && data_dims.z > 0);

  size_t elem_count = (size_t)data_dims.x * data_dims.y * data_dims.z;

  size_t elem_size = // clang-format off
    (data_type == VALUE_TYPE_UINT8  || data_type == VALUE_TYPE_INT8 ) ? sizeof(uint8_t)  :
    (data_type == VALUE_TYPE_UINT16 || data_type == VALUE_TYPE_INT16) ? sizeof(uint16_t) :
    (data_type == VALUE_TYPE_UINT32 || data_type == VALUE_TYPE_INT32 || data_type == VALUE_TYPE_FLOAT) ? 
    sizeof(uint32_t) : sizeof(double);
  // clang-format on

  size_t data_size = elem_count * elem_size;

  // load the data
  std::shared_ptr<char[]> data_buffer;

  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  if (ifs.fail()) // cannot open the file
  {
    throw std::runtime_error("Cannot open the file");
  }

  ifs.seekg(0, std::ios::end);
  size_t file_size = ifs.tellg();
  if (file_size < data_offset + data_size) // file size does not match data size
  {
    throw std::runtime_error("File size does not match data size");
  }
  ifs.seekg(data_offset, std::ios::beg);

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

  const bool reverse = (data_is_big_endian && elem_size > 1);

  // reverse byte-order if necessary
  if (reverse) {
    reverseByteOrder(data_buffer.get(), elem_count, elem_size);
  }

  ifs.close();

  // create volume
  Array3D& output = self.volume;

  switch (data_type) {
  case VALUE_TYPE_UINT8: output = CreateArray3D<uint8_t>(data_buffer.get(), data_dims); break;
  case VALUE_TYPE_INT8: output = CreateArray3D<int8_t>(data_buffer.get(), data_dims); break;
  case VALUE_TYPE_UINT32: output = CreateArray3D<uint32_t>(data_buffer.get(), data_dims); break;
  case VALUE_TYPE_INT32: output = CreateArray3D<int32_t>(data_buffer.get(), data_dims); break;
  case VALUE_TYPE_FLOAT: output = CreateArray3D<float>(data_buffer.get(), data_dims); break;
  // TODO cannot handle the following correctly, so converting them into floats
  case VALUE_TYPE_UINT16:
    data_buffer = convert_volume<uint16_t, float>(data_buffer, elem_count);
    output = CreateArray3D<float>(data_buffer.get(), data_dims);
    break;
  case VALUE_TYPE_INT16:
    data_buffer = convert_volume<int16_t, float>(data_buffer, elem_count);
    output = CreateArray3D<float>(data_buffer.get(), data_dims);
    break;
  case VALUE_TYPE_DOUBLE:
    data_buffer = convert_volume<double, float>(data_buffer, elem_count);
    output = CreateArray3D<float>(data_buffer.get(), data_dims);
    break;
  default: throw std::runtime_error("#osc: unexpected volume type ...");
  }

  if (data_value_max >= data_value_min) {
    output.upper = min(output.upper, data_value_max);
    output.lower = max(output.lower, data_value_min);
  }

  output.scale = 1.f / (output.upper - output.lower);
}

} // namespace host
} // namespace ovr
