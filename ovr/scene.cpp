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

#include <vidi_parallel_algorithm.h>

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
CreateColorMap(const char* name)
{
  if (colormap::has(name)) {
    const std::vector<vec4f>& arr = (const std::vector<vec4f>&)colormap::get(name);
    return CreateArray1DFloat4(arr, false);
  }
  else {
    throw std::runtime_error("Unexpected colormap name: " + std::string(name));
  }
}

array_3d_scalar_t
CreateArray3DScalarFromFile(const char* filename, vec3i dims, ValueType type, size_t offset, bool is_big_endian)
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
    data_buffer = vidi::read_volume_structured_regular(std::string(filename), desc);
  }
  // finalize
  array_3d_scalar_t output = std::make_shared<Array<3>>();
  output->dims = dims;
  output->type = type;
  output->acquire_data(std::move(data_buffer));
  return output;
}

#define INDENT std::string(indent, ' ')

namespace scene {

template<typename T>
static void 
print_array(std::ostream& os, const array_1d_t& array, size_t max_n_items)
{
  if (!array) {
    os << "nullptr\n";
    return;
  }

  os << "{ ";
  for (int i = 0; i < std::min(array->size(), max_n_items); i++) {
    os << array->data_typed<T>()[i] << " ";
  }
  if (array->size() > max_n_items) {
    os << "... skipped " << array->size() - max_n_items << " items ";
  }
  os << "}\n";
}

void 
TransferFunction::print(std::ostream& os, int indent) const
{
  os << INDENT << "transfer_function\n";
  os << INDENT << "+-> value_range = " << value_range << "\n";
  os << INDENT << "+-> color = ";
  print_array<vec4f>(os, color, 5);
  os << INDENT << "+-> opacity = ";
  print_array<float>(os, opacity, 5);
}

box3f 
Volume::get_bounds(const Scene&) const 
{
  box3f bounds;
  if (type == STRUCTURED_REGULAR_VOLUME) {
    auto& sr = structured_regular;
    auto& origin  = sr.grid_origin;
    auto& spacing = sr.grid_spacing;
    auto dims = vec3f(sr.data->dims);
    bounds = { origin, origin+dims*spacing };
  }
  return bounds;
}

void
Volume::print(const Scene& self, std::ostream& os, int indent) const
{
  if (type == STRUCTURED_REGULAR_VOLUME) {
    auto& sr = structured_regular;
    os << INDENT << "volume structured_regular\n";
    os << INDENT << "+-> grid_origin = " << sr.grid_origin << "\n";
    os << INDENT << "+-> grid_spacing = " << sr.grid_spacing << "\n";
    os << INDENT << "+-> data = " << sr.data << "\n";
  }
}

box3f 
Geometry::get_bounds(const Scene& self) const 
{
  box3f bounds;
  if (type == TRIANGLES_GEOMETRY) {
    auto& g = triangles;
    for (int i = 0; i < g.position->size(); i++) {  // TODO optimize this
      bounds.extend(g.position->data_typed<vec3f>()[i]);
    }
  }
  else if (type == SPHERES_GEOMETRY) {
    auto& g = spheres;
    for (int i = 0; i < g.sphere.position->size(); i++) {  // TODO optimize this
      bounds.extend(g.sphere.position->data_typed<vec3f>()[i]);
    }
  }
  else if (type == ISOSURFACE_GEOMETRY) {
    auto& g = isosurfaces;
    bounds = self.textures[g.volume_texture].volume.volume.get_bounds(self);
  }
  return bounds;
}

void
Geometry::print(const Scene& self, std::ostream& os, int indent) const
{
  if (type == TRIANGLES_GEOMETRY) {
    auto& g = triangles;
    os << INDENT << "geometry triangles\n";
    os << INDENT << "+-> position = ";
    print_array<vec3f>(os, g.position, 5);
    os << INDENT << "+-> index = ";
    print_array<int>(os, g.index, 5);
    os << INDENT << "+-> verts.normal = ";
    print_array<vec3f>(os, g.verts.normal, 5);
    os << INDENT << "+-> verts.texcoord = ";
    print_array<vec2f>(os, g.verts.texcoord, 5);
    os << INDENT << "+-> faces.normal = ";
    print_array<vec3f>(os, g.faces.normal, 5);
    os << INDENT << "+-> faces.texcoord = ";
    print_array<vec2f>(os, g.faces.texcoord, 5);
  }
  else if (type == SPHERES_GEOMETRY) {
    auto& g = spheres;
    os << INDENT << "geometry spheres\n";
    os << INDENT << "+-> sphere.position = ";
    print_array<vec3f>(os, g.sphere.position, 5);
    os << INDENT << "+-> sphere.radius = ";
    print_array<float>(os, g.sphere.radius, 5);
    os << INDENT << "+-> radius = " << g.radius << "\n";
  }
  else if (type == ISOSURFACE_GEOMETRY) {
    auto& g = isosurfaces;
    os << INDENT << "geometry isosurfaces\n";
    os << INDENT << "+-> isovalues = ";
    print_array<float>(os, g.isovalues, 5);
    os << INDENT << "+-> volume_texture = " << g.volume_texture << "\n";
    self.textures[g.volume_texture].volume.volume.print(self, os, indent + 4);
  }
}

box3f 
Model::get_bounds(const Scene& self) const 
{
  box3f bounds;
  if (type == GEOMETRIC_MODEL) {
    bounds = geometry_model.geometry.get_bounds(self);
  } else if (type == VOLUMETRIC_MODEL) {
    bounds = self.textures[volume_model.volume_texture].volume.volume.get_bounds(self);
  }
  return bounds;
}

void
Model::print(const Scene& self, std::ostream& os, int indent) const
{
  if (type == GEOMETRIC_MODEL) {
    os << INDENT << "model geometric\n";
    os << INDENT << "+-> mtl = " << geometry_model.mtl << "\n";
    os << INDENT << "+-> mtls = { ";
    for (auto& mtl : geometry_model.mtls) {
      os << mtl << " ";
    }
    os << "}\n";
    geometry_model.geometry.print(self, os, indent + 4);
  } else if (type == VOLUMETRIC_MODEL) {
    os << INDENT << "model volumetric\n";
    os << INDENT << "+-> transfer_function = ";
    volume_model.transfer_function.print(os, indent + 4);
    os << INDENT << "+-> volume_texture = " << volume_model.volume_texture << "\n";
    self.textures[volume_model.volume_texture].volume.volume.print(self, os, indent + 4);
  }
}

box3f 
Instance::get_bounds(const Scene& self) const 
{
  box3f bounds;
  for (auto& model : models) {
    auto b = model.get_bounds(self);
    b.lower = xfmPoint(transform, b.lower);
    b.upper = xfmPoint(transform, b.upper);
    bounds.extend(b);
  }
  return bounds;
}

void
Instance::print(const Scene& self, std::ostream& os, int indent) const
{
  os << INDENT << "instance\n";
  os << INDENT << "+-> transform: " << transform << "\n";
  for (auto& model : models) {
    model.print(self, os, indent + 4);
  }
}

box3f
Scene::get_bounds() const
{
  box3f bounds;
  for (auto& instance : instances) {
    bounds.extend(instance.get_bounds(*this));
  }
  return bounds;
}

void
Scene::print() const
{
  std::ostream& os = std::cout;
  os << "scene:\n";
  for (auto& instance : instances) {
    instance.print(*this, os, 2);
  }
}

template<typename T> T 
lerp(float t, T a, T b) { return a + t * (b - a); }

static vec4f 
access_transfer_function(const ovr::scene::TransferFunction& self, float value) 
{
  using namespace ovr;

  // remap to [0.0, 1.0]
  value = (value - self.value_range.x) / (self.value_range.y - self.value_range.x);
  // clamp to [0.0, 1.0)
  const float nextBefore1 = 0x1.fffffep-1f;
  value = clamp(value, 0.0f, nextBefore1);

  const int maxIdxC = self.color->size() - 1;
  const float idxCf = value * maxIdxC;
  float intC;
  const float fracC = std::modf(idxCf, &intC);
  const int idxC = idxCf;

  vec4f* cdata = (vec4f*)self.color->data();
  const vec4f col = lerp(fracC, cdata[idxC], cdata[std::min(maxIdxC, idxC + 1)]);

  const int maxIdxO = self.opacity->size() - 1;
  const float idxOf = value * maxIdxO;
  float intO;
  const float fracO = std::modf(idxOf, &intO);
  const int idxO = idxOf;

  float* odata = (float*)self.opacity->data();
  const float opacity = lerp(fracO, odata[idxO], odata[std::min(maxIdxO, idxO + 1)]);

  return vec4f(col.xyz(), opacity);
}

std::vector<uint32_t> 
Scene::add_materials_for_isosurfaces(std::vector<float> isovalues, 
  const ovr::scene::TransferFunction& transfer_function)
{
  std::vector<uint32_t> mtls;
  for (int i = 0; i < isovalues.size(); ++i) {
    ovr::scene::Material volume_mtl;
    volume_mtl.type = ovr::scene::Material::OBJ_MATERIAL;
    volume_mtl.obj.kd = access_transfer_function(transfer_function, isovalues[i]).xyz();
    this->materials.push_back(volume_mtl);
    mtls.push_back(this->materials.size() - 1);
  }
  return mtls;
}

} // namespace scene
} // namespace ovr
