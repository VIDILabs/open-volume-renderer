#ifndef VIDI_VOLUME_READER_H
#define VIDI_VOLUME_READER_H

#include <cassert>
#include <fstream>
#include <memory>
#include <vector>

namespace vidi {

#ifndef VIDI_VOLUME_EXTERNAL_TYPE_ENUM
enum VoxelType {
  VOXEL_UINT8 = 100,
  VOXEL_INT8,
  VOXEL_UINT16 = 200,
  VOXEL_INT16,
  VOXEL_UINT32 = 300,
  VOXEL_INT32,
  VOXEL_FLOAT = 400,
  VOXEL_FLOAT2,
  VOXEL_FLOAT3,
  VOXEL_FLOAT4,
  VOXEL_DOUBLE = 500,
  VOXEL_DOUBLE2,
  VOXEL_DOUBLE3,
  VOXEL_DOUBLE4,
  VOXEL_VOID = 1000,
};
#endif // VIDI_VOLUME_READER_EXTERNAL_ENUM

inline int
sizeof_voxel_type(VoxelType type)
{
  switch (type) {
  case VOXEL_UINT8:
  case VOXEL_INT8: return sizeof(char);
  case VOXEL_UINT16:
  case VOXEL_INT16: return sizeof(short);
  case VOXEL_UINT32:
  case VOXEL_INT32: return sizeof(int);
  case VOXEL_FLOAT:   return sizeof(float);
  case VOXEL_FLOAT2:  return sizeof(float)*2;
  case VOXEL_FLOAT3:  return sizeof(float)*3;
  case VOXEL_FLOAT4:  return sizeof(float)*4;
  case VOXEL_DOUBLE:  return sizeof(double);
  case VOXEL_DOUBLE2: return sizeof(double)*2;
  case VOXEL_DOUBLE3: return sizeof(double)*3;
  case VOXEL_DOUBLE4: return sizeof(double)*4;
  default: return 0;
  }
}

template<typename T>
VoxelType
voxel_type();

#define _instantiate_voxel_type_function(TYPE, type) \
  template<>                                         \
  inline VoxelType voxel_type<type>()                \
  {                                                  \
    return TYPE;                                     \
  }
_instantiate_voxel_type_function(VOXEL_UINT8,  uint8_t);
_instantiate_voxel_type_function(VOXEL_INT8,   int8_t);
_instantiate_voxel_type_function(VOXEL_UINT16, uint16_t);
_instantiate_voxel_type_function(VOXEL_INT16,  int16_t);
_instantiate_voxel_type_function(VOXEL_UINT32, uint32_t);
_instantiate_voxel_type_function(VOXEL_INT32,  int32_t);
_instantiate_voxel_type_function(VOXEL_FLOAT,  float);
_instantiate_voxel_type_function(VOXEL_DOUBLE, double);
#undef _instantiate_voxel_type_function

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

struct VolumeFileDesc {
  VoxelType type;
  struct {
    int x, y, z;
  } dims;

  size_t offset;
  bool is_big_endian;
};

[[deprecated]] typedef VolumeFileDesc StructuredRegularVolumeDesc;

inline std::shared_ptr<char[]>
read_volume_structured_regular(const std::string& filename, VolumeFileDesc desc)
{
  // data geometry
  assert(desc.dims.x > 0 && desc.dims.y > 0 && desc.dims.z > 0);

  size_t elem_count = (size_t)desc.dims.x * desc.dims.y * desc.dims.z;

  size_t elem_size = sizeof_voxel_type(desc.type);

  size_t data_size = elem_count * elem_size;

  // load the data
  std::shared_ptr<char[]> data_buffer;

  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  if (ifs.fail()) // cannot open the file
    throw std::runtime_error("Cannot open the file: " + filename);

  ifs.seekg(0, std::ios::end);

  size_t file_size = ifs.tellg();
  if (file_size < desc.offset + data_size) // file size does not match data size
    throw std::runtime_error("File size does not match data size");

  ifs.seekg(desc.offset, std::ios::beg);

  try {
    data_buffer.reset(new char[data_size]);
  }
  catch (std::bad_alloc&) { // memory allocation failed
    throw std::runtime_error("Cannot allocate memory for the data");
  }

  // read data
  ifs.read(data_buffer.get(), data_size);
  if (ifs.fail()) // reading data failed
    throw std::runtime_error("Cannot read the file");

  // reverse byte-order if necessary
  const bool reverse = (desc.is_big_endian && elem_size > 1);
  if (reverse)
    reverse_byte_order(data_buffer.get(), elem_count, elem_size);
  ifs.close();

  return data_buffer;
}

inline void
read_volume_structured_regular(const std::string& filename, VolumeFileDesc desc, void* dst)
{
  // data geometry
  assert(desc.dims.x > 0 && desc.dims.y > 0 && desc.dims.z > 0);

  size_t elem_count = (size_t)desc.dims.x * desc.dims.y * desc.dims.z;

  size_t elem_size = sizeof_voxel_type(desc.type);

  size_t data_size = elem_count * elem_size;

  // // load the data
  // std::shared_ptr<char[]> data_buffer;

  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  if (ifs.fail()) // cannot open the file
    throw std::runtime_error("Cannot open the file");

  ifs.seekg(0, std::ios::end);

  size_t file_size = ifs.tellg();
  if (file_size < desc.offset + data_size) // file size does not match data size
    throw std::runtime_error("File size does not match data size");

  ifs.seekg(desc.offset, std::ios::beg);

  // try {
  //   data_buffer.reset(new char[data_size]);
  // }
  // catch (std::bad_alloc&) { // memory allocation failed
  //   throw std::runtime_error("Cannot allocate memory for the data");
  // }

  // read data
  ifs.read((char*)dst, data_size);
  if (ifs.fail()) // reading data failed
    throw std::runtime_error("Cannot read the file");

  // reverse byte-order if necessary
  const bool reverse = (desc.is_big_endian && elem_size > 1);
  if (reverse)
    reverse_byte_order((char*)dst, elem_count, elem_size);
  ifs.close();

  // return data_buffer;
}

} // namespace vidi

#endif // VIDI_VOLUME_READER_H
