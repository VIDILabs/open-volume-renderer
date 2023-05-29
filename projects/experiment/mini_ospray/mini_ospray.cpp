// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/* This is a small example tutorial how to use OSPRay in an application.
 *
 * On Linux build it in the build_directory with
 *   gcc -std=c99 ../apps/ospTutorial/ospTutorial.c \
 *       -I ../ospray/include -L . -lospray -Wl,-rpath,. -o ospTutorial
 * On Windows build it in the build_directory\$Configuration with
 *   cl ..\..\apps\ospTutorial\ospTutorial.c -I ..\..\ospray\include ^
 *       -I ..\.. ospray.lib
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#ifdef _WIN32
#include <conio.h>
#include <malloc.h>
#include <windows.h>
#else
#include <alloca.h>
#endif
#include "ospray/ospray_util.h"

#define STB_IMAGE_IMPLEMENTATION
#include <3rdparty/stb_image.h>
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <3rdparty/stb_image_write.h>

#include <string>
#include <iostream>
#include <fstream>

#include "diva_serializable.h"
#include "common_host.h"

#include <3rdparty/json.hpp>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

using namespace diva;

// helper function to write the rendered image as PPM file
void writePPM(
    const char *fileName, int size_x, int size_y, const uint32_t *pixel)
{
  FILE *file = fopen(fileName, "wb");
  if (!file)
  {
    fprintf(stderr, "fopen('%s', 'wb') failed: %d", fileName, errno);
    return;
  }
  fprintf(file, "P6\n%i %i\n255\n", size_x, size_y);
  unsigned char *out = (unsigned char *)alloca(3 * size_x);
  for (int y = 0; y < size_y; y++)
  {
    const unsigned char *in =
        (const unsigned char *)&pixel[(size_y - 1 - y) * size_x];
    for (int x = 0; x < size_x; x++)
    {
      out[3 * x + 0] = in[4 * x + 0];
      out[3 * x + 1] = in[4 * x + 1];
      out[3 * x + 2] = in[4 * x + 2];
    }
    fwrite(out, 3 * size_x, sizeof(char), file);
  }
  fprintf(file, "\n");
  fclose(file);
}

void writePNG(
    std::string filename, int width, int height, const uint32_t *pixels)
{
  uint8_t *_chars = new uint8_t[width * height * 4];
  int index = 0;
  for (int j = height - 1; j >= 0; --j)
  {
    // std::cout << j << std::endl;
    for (int i = 0; i < width; ++i)
    {
      // std::cout << i << std::endl;
      const uint8_t *pixel = (const uint8_t *)&pixels[i + j * width];
      _chars[index++] = pixel[0];
      _chars[index++] = pixel[1];
      _chars[index++] = pixel[2];
      _chars[index++] = pixel[3];
    }
  }
  // filename += ".jpg";
  // stbi_write_jpg(filename.c_str(), width, height, 3, _chars, 100);
  // filename += ".png";
  // stbi_write_png(filename.c_str(), width, height, 3, _chars, width * 3);
  filename += ".png";
  stbi_write_png(filename.c_str(), width, height, 4, _chars, width * 4);
  std::cout << "saving " << filename << std::endl;
  delete[] _chars;
}

template <typename T>
vec2f compute_minmax(void *_array, size_t count)
{
  T *array = (T *)_array;

  float actual_max = tbb::parallel_reduce(
      tbb::blocked_range<T *>(array, array + count), -std::numeric_limits<float>::max(),
      [](const tbb::blocked_range<T *> &r, float init) -> float
      {
        for (T *a = r.begin(); a != r.end(); ++a)
          init = std::max(init, static_cast<float>(*a));
        return init;
      },
      [](float x, float y) -> float
      { return std::max(x, y); });

  float actual_min = tbb::parallel_reduce(
      tbb::blocked_range<T *>(array, array + count), std::numeric_limits<float>::max(),
      [](const tbb::blocked_range<T *> &r, float init) -> float
      {
        for (T *a = r.begin(); a != r.end(); ++a)
          init = std::min(init, static_cast<float>(*a));
        return init;
      },
      [](float x, float y) -> float
      { return std::min(x, y); });

  return vec2f(actual_min, actual_max);
}

NLOHMANN_JSON_SERIALIZE_ENUM(V3D_TYPE,
                             {{V3D_VOID, "VOID"},
                              {V3D_BYTE, "BYTE"},
                              {V3D_UNSIGNED_BYTE, "UNSIGNED_BYTE"},
                              {V3D_SHORT, "SHORT"},
                              {V3D_UNSIGNED_SHORT, "UNSIGNED_SHORT"},
                              {V3D_INT, "INT"},
                              {V3D_UNSIGNED_INT, "UNSIGNED_INT"},
                              {V3D_FLOAT, "FLOAT"},
                              {V3D_DOUBLE, "DOUBLE"}});

OSPData
load_volume(const std::string &filename,
            /* loading parameters */
            VoxelType data_type,
            vec3i data_dimensions,
            size_t data_offset,
            bool data_is_big_endian)
{
  assert(data_dimensions.x > 0 && data_dimensions.y > 0 && data_dimensions.z > 0);

  const size_t elem_count = (size_t)data_dimensions.x * data_dimensions.y * data_dimensions.z;

  const size_t elem_size = // clang-format off
      (data_type == VoxelType::TYPE_UINT8  || 
       data_type == VoxelType::TYPE_INT8)  ? sizeof(uint8_t)  :
      (data_type == VoxelType::TYPE_UINT16 || 
       data_type == VoxelType::TYPE_INT16) ? sizeof(uint16_t) :
      (data_type == VoxelType::TYPE_UINT32 || 
       data_type == VoxelType::TYPE_INT32  || 
       data_type == VoxelType::TYPE_FLOAT) ? sizeof(uint32_t) : 
      sizeof(double); // clang-format on

  const size_t num_of_bytes = elem_count * elem_size;

  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  if (ifs.fail()) // cannot open the file
  {
    throw std::runtime_error("Cannot open the file");
  }

  ifs.seekg(0, std::ios::end);
  size_t file_size = ifs.tellg();
  if (file_size < data_offset + num_of_bytes) // file size does not match data size
  {
    throw std::runtime_error("File size does not match data size");
  }
  ifs.seekg(data_offset, std::ios::beg);

  std::shared_ptr<char[]> volume;
  try
  {
    volume.reset(new char[num_of_bytes]);
  }
  catch (std::bad_alloc &) // memory allocation failed
  {
    throw std::runtime_error("Cannot allocate memory for the data");
  }

  // read data
  ifs.read(volume.get(), num_of_bytes);
  if (ifs.fail()) // reading data failed
  {
    throw std::runtime_error("Cannot read the file");
  }

  const bool reverse = (data_is_big_endian && elem_size > 1);

  // reverse byte-order if necessary
  if (reverse)
  {
    reverseByteOrder(volume.get(), elem_count, elem_size);
  }

  ifs.close();

  // set volume value range
  vec2f actual_range;

  // create OSPRay Data
  OSPData shared_data;
  OSPData copied_data;

  // OSPData volume;
  switch (data_type)
  {
  case VoxelType::TYPE_UINT8:
    actual_range = compute_minmax<uint8_t>(volume.get(), elem_count);

    shared_data = ospNewSharedData3D(volume.get(), OSP_UCHAR, data_dimensions.x, data_dimensions.y, data_dimensions.z);
    copied_data = ospNewData(OSP_UCHAR, data_dimensions.x, data_dimensions.y, data_dimensions.z);

    break;
  case VoxelType::TYPE_INT8:
    actual_range = compute_minmax<int8_t>(volume.get(), elem_count);

    shared_data = ospNewSharedData3D(volume.get(), OSP_CHAR, data_dimensions.x, data_dimensions.y, data_dimensions.z);
    copied_data = ospNewData(OSP_CHAR, data_dimensions.x, data_dimensions.y, data_dimensions.z);

    break;
  case VoxelType::TYPE_UINT16:
    actual_range = compute_minmax<uint16_t>(volume.get(), elem_count);

    shared_data = ospNewSharedData3D(volume.get(), OSP_USHORT, data_dimensions.x, data_dimensions.y, data_dimensions.z);
    copied_data = ospNewData(OSP_USHORT, data_dimensions.x, data_dimensions.y, data_dimensions.z);

    break;
  case VoxelType::TYPE_INT16:
    actual_range = compute_minmax<int16_t>(volume.get(), elem_count);

    shared_data = ospNewSharedData3D(volume.get(), OSP_SHORT, data_dimensions.x, data_dimensions.y, data_dimensions.z);
    copied_data = ospNewData(OSP_SHORT, data_dimensions.x, data_dimensions.y, data_dimensions.z);

    break;
  case VoxelType::TYPE_UINT32:
    actual_range = compute_minmax<uint32_t>(volume.get(), elem_count);

    shared_data = ospNewSharedData3D(volume.get(), OSP_UINT, data_dimensions.x, data_dimensions.y, data_dimensions.z);
    copied_data = ospNewData(OSP_UINT, data_dimensions.x, data_dimensions.y, data_dimensions.z);

    break;
  case VoxelType::TYPE_INT32:
    actual_range = compute_minmax<int32_t>(volume.get(), elem_count);

    shared_data = ospNewSharedData3D(volume.get(), OSP_INT, data_dimensions.x, data_dimensions.y, data_dimensions.z);
    copied_data = ospNewData(OSP_INT, data_dimensions.x, data_dimensions.y, data_dimensions.z);

    break;
  case VoxelType::TYPE_FLOAT:
    actual_range = compute_minmax<float>(volume.get(), elem_count);

    shared_data = ospNewSharedData3D(volume.get(), OSP_FLOAT, data_dimensions.x, data_dimensions.y, data_dimensions.z);
    copied_data = ospNewData(OSP_FLOAT, data_dimensions.x, data_dimensions.y, data_dimensions.z);

    break;
  case VoxelType::TYPE_DOUBLE:
    actual_range = compute_minmax<double>(volume.get(), elem_count);

    shared_data = ospNewSharedData3D(volume.get(), OSP_DOUBLE, data_dimensions.x, data_dimensions.y, data_dimensions.z);
    copied_data = ospNewData(OSP_DOUBLE, data_dimensions.x, data_dimensions.y, data_dimensions.z);

    break;
  default:
    throw std::runtime_error("#osc: unexpected volume type ...");
  }

  ospCopyData1D(shared_data, copied_data, 0);

  return copied_data;
}

void load_v3djson(std::string input)
{
  using json = nlohmann::json;

  // read a JSON file
  std::ifstream in(input);
  json jsonfile;
  in >> jsonfile;

  auto jsondata = jsonfile["dataSource"][0];

  std::string filename = jsondata["fileName"].get<std::string>();
  vec3i dims;
  from_json(jsondata["dimensions"], dims);
  V3D_TYPE type;
  from_json(jsondata["type"], type);
  vec3f spacing(1.f);
  if (jsondata.contains("gridSpacing"))
    from_json(jsondata["gridSpacing"], spacing);

  json jsonview;
  if (jsonfile.contains("view"))
  {
    jsonview = jsonfile["view"];
  }
  else if (jsonfile.contains("scene"))
  {
    jsonview = jsonfile["scene"];
  }
  else
  {
    throw std::runtime_error("cannot find 'view' specification in JSON");
  }

  vec2f range;
  if (jsonview["volume"].contains("scalarMappingRange"))
  {
    range.x = jsonview["volume"]["scalarMappingRange"]["minimum"].get<float>();
    range.y = jsonview["volume"]["scalarMappingRange"]["maximum"].get<float>();
  }
  else if (jsonview["volume"].contains("scalarDomain"))
  {
    range.x = jsonview["volume"]["scalarDomain"]["minimum"].get<float>();
    range.y = jsonview["volume"]["scalarDomain"]["maximum"].get<float>();
  }
  else
  {
    throw std::runtime_error("cannot find 'scalarMappingRange' specification in JSON");
  }

  auto jsontfn = jsonview["volume"]["transferFunction"];
  std::vector<vec4f> tfn_c; /* p, rgb  */
  std::vector<vec4f> tex_c; /*    rgba */
  std::vector<float> tex_a; /*  alpha  */

  assert("BASE64" == jsontfn["alphaArray"]["encoding"].get<std::string>());
  auto alpha_array_base64 = jsontfn["alphaArray"]["data"].get<std::string>();
  auto resolution = int(size_base64(alpha_array_base64) / sizeof(float));
  assert(resolution == jsontfn["resolution"].get<int>());
  tex_a.resize(resolution);
  from_base64(alpha_array_base64, reinterpret_cast<char *>(tex_a.data()));

  if (jsontfn.contains("colorControls"))
  {
    const auto &color_controls = jsontfn["colorControls"];
    size_t count = color_controls.size();
    tfn_c.clear();
    for (int i = 0; i < count; ++i)
    {
      const auto &cc = color_controls[i];
      tfn_c.push_back(vec4f(cc["position"].get<float>(),     //
                            cc["color"]["r"].get<float>(),   //
                            cc["color"]["g"].get<float>(),   //
                            cc["color"]["b"].get<float>())); //
    }
  }

  auto lerp = [](const float l, const float r, const float pl, const float pr, const float p) -> float
  {
    const float dl = std::abs(pr - pl) > 0.0001f ? (p - pl) / (pr - pl) : 0.f;
    const float dr = 1.f - dl;
    return l * dr + r * dl;
  };

  tex_c.reserve(resolution);

  const float step = 1.0f / (float)(resolution - 1);
  for (int i = 0; i < resolution; ++i)
  {
    const float p = clamp(i * step, 0.0f, 1.0f);

    auto it_lower = std::lower_bound(tfn_c.begin(), tfn_c.end(), p, [](vec4f cc, float p)
                                     { return cc.x < p; });
    if (it_lower == tfn_c.end())
    {
      it_lower = tfn_c.end() - 1;
    }
    auto it_upper = it_lower + 1;
    if (it_upper == tfn_c.end())
    {
      it_upper = tfn_c.end() - 1;
    }

    const float r = lerp(it_lower->y, it_upper->y, it_lower->x, it_upper->x, p);
    const float g = lerp(it_lower->z, it_upper->z, it_lower->x, it_upper->x, p);
    const float b = lerp(it_lower->w, it_upper->w, it_lower->x, it_upper->x, p);
    tex_c.push_back(vec4f(r, g, b, 1.f));
  }

}

OSPGroup create_scene_volume()
{
  const std::string file = "volume/3.900E-04_H2O2.raw";
  const vec3i dims = vec3i(600, 750, 500);
  const vec3f spacing = vec3f(1.f, 1.f, 1.f);
  const vec2f range = vec2f(0.f, 0.0003f);

  OSPData data = load_volume(file, VoxelType::TYPE_FLOAT, dims, 0, false);

  // create and setup model and mesh
  OSPVolume volume = ospNewVolume("structuredRegular");
  ospSetParam(volume, "spacing", OSP_VEC3F, &spacing);
  ospSetObject(volume, "data", data);
  ospCommit(volume);

  static auto alphas = std::vector<float>{0.f, 0.05f, 0.1f, 0.1f, 0.1f, 0.05f, 0.f};
  static auto colors = std::vector<float>{1.f, 0.f, 0.f, 0.f, 0.f, 1.f};

  OSPData alphas_data = ospNewSharedData1D(alphas.data(), OSP_FLOAT, alphas.size());

  OSPData colors_data = ospNewSharedData1D(colors.data(), OSP_VEC3F, 2);

  OSPTransferFunction tfn = ospNewTransferFunction("piecewiseLinear");
  ospSetParam(tfn, "valueRange", OSP_VEC2F, &range);
  ospSetObject(tfn, "color", colors_data);
  ospSetObject(tfn, "opacity", alphas_data);
  ospCommit(tfn);

  // put the mesh into a model
  OSPVolumetricModel model = ospNewVolumetricModel(volume);
  ospSetObject(model, "transferFunction", tfn);
  ospCommit(model);
  ospRelease(volume);
  ospRelease(tfn);

  // put the model into a group (collection of models)
  OSPGroup group = ospNewGroup();
  ospSetObjectAsData(group, "volume", OSP_VOLUMETRIC_MODEL, model);
  ospCommit(group);
  ospRelease(model);

  return group;
}

OSPGroup create_scene_simple()
{
  // triangle mesh data
  static float vertex[] = {-1.0f,
                           -1.0f,
                           3.0f,
                           -1.0f,
                           1.0f,
                           3.0f,
                           1.0f,
                           -1.0f,
                           3.0f,
                           0.1f,
                           0.1f,
                           0.3f};
  static float color[] = {0.9f,
                          0.5f,
                          0.5f,
                          1.0f,
                          0.8f,
                          0.8f,
                          0.8f,
                          1.0f,
                          0.8f,
                          0.8f,
                          0.8f,
                          1.0f,
                          0.5f,
                          0.9f,
                          0.5f,
                          1.0f};
  static int32_t index[] = {0, 1, 2, 1, 2, 3};

  // create and setup model and mesh
  OSPGeometry mesh = ospNewGeometry("mesh");

  OSPData data = ospNewSharedData1D(vertex, OSP_VEC3F, 4);
  // alternatively with an OSPRay managed OSPData
  // OSPData managed = ospNewData1D(OSP_VEC3F, 4);
  // ospCopyData1D(data, managed, 0);

  ospCommit(data);
  ospSetObject(mesh, "vertex.position", data);
  ospRelease(data); // we are done using this handle

  data = ospNewSharedData1D(color, OSP_VEC4F, 4);
  ospCommit(data);
  ospSetObject(mesh, "vertex.color", data);
  ospRelease(data);

  data = ospNewSharedData1D(index, OSP_VEC3UI, 2);
  ospCommit(data);
  ospSetObject(mesh, "index", data);
  ospRelease(data);

  ospCommit(mesh);

  OSPMaterial mat = ospNewMaterial("pathtracer", "obj");
  ospCommit(mat);

  // put the mesh into a model
  OSPGeometricModel model = ospNewGeometricModel(mesh);
  ospSetObject(model, "material", mat);
  ospCommit(model);
  ospRelease(mesh);
  ospRelease(mat);

  // put the model into a group (collection of models)
  OSPGroup group = ospNewGroup();
  ospSetObjectAsData(group, "geometry", OSP_GEOMETRIC_MODEL, model);
  ospCommit(group);
  ospRelease(model);

  return group;
}

int main(int argc, const char **argv)
{
  // image size
  int imgSize_x = 1024; // width
  int imgSize_y = 768;  // height

  // camera
  float cam_pos[] = {300.f, 300.f, -600.f};
  float cam_up[] = {0.f, 1.f, 0.f};
  float cam_view[] = {0.1f, 0.f, 1.f};

#ifdef _WIN32
  int waitForKey = 0;
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi))
  {
    // detect standalone console: cursor at (0,0)?
    waitForKey = csbi.dwCursorPosition.X == 0 && csbi.dwCursorPosition.Y == 0;
  }
#endif

  printf("initialize OSPRay...");

  // initialize OSPRay; OSPRay parses (and removes) its commandline parameters,
  // e.g. "--osp:debug"
  OSPError init_error = ospInit(&argc, argv);
  if (init_error != OSP_NO_ERROR)
    return init_error;

  printf("done\n");

  // create and setup camera
  printf("setting up camera...");

  OSPCamera camera = ospNewCamera("perspective");
  {
    ospSetFloat(camera, "aspect", imgSize_x / (float)imgSize_y);
    ospSetParam(camera, "position", OSP_VEC3F, cam_pos);
    ospSetParam(camera, "direction", OSP_VEC3F, cam_view);
    ospSetParam(camera, "up", OSP_VEC3F, cam_up);
    ospCommit(camera); // commit each object to indicate modifications are done
  }

  printf("done\n");

  printf("setting up scene...");

  OSPWorld world = ospNewWorld();
  {
    // create and setup model and mesh
    // OSPGroup group = create_scene_simple();
    OSPGroup group = create_scene_volume();

    // put the group into an instance (give the group a world transform)
    OSPInstance instance = ospNewInstance(group);
    ospCommit(instance);
    ospRelease(group);

    // put the instance in the world
    ospSetObjectAsData(world, "instance", OSP_INSTANCE, instance);
    ospRelease(instance);

    // create and setup light for Ambient Occlusion
    OSPLight light = ospNewLight("ambient");
    ospCommit(light);
    ospSetObjectAsData(world, "light", OSP_LIGHT, light);
    ospRelease(light);
  }

  ospCommit(world);

  printf("done\n");

  // print out world bounds
  OSPBounds worldBounds = ospGetBounds(world);
  printf("world bounds: ({%f, %f, %f}, {%f, %f, %f}\n\n",
         worldBounds.lower[0],
         worldBounds.lower[1],
         worldBounds.lower[2],
         worldBounds.upper[0],
         worldBounds.upper[1],
         worldBounds.upper[2]);

  printf("setting up renderer...");

  // create renderer
  OSPRenderer renderer = ospNewRenderer("scivis"); // choose path tracing renderer

  // complete setup of renderer
  ospSetFloat(renderer, "backgroundColor", 1.0f); // white, transparent
  // ospSetBool(renderer, "shadow", true);
  // ospSetInt(renderer, "aoSamples", 1);
  ospCommit(renderer);

  // create and setup framebuffer
  OSPFrameBuffer framebuffer = ospNewFrameBuffer(imgSize_x,
                                                 imgSize_y,
                                                 OSP_FB_SRGBA,
                                                 OSP_FB_COLOR | /*OSP_FB_DEPTH |*/ OSP_FB_ACCUM);
  ospResetAccumulation(framebuffer);

  printf("rendering initial frame to firstFrame.ppm...");

  // render one frame
  ospRenderFrameBlocking(framebuffer, renderer, camera, world);

  // access framebuffer and write its content as PPM file
  const uint32_t *fb = (uint32_t *)ospMapFrameBuffer(framebuffer, OSP_FB_COLOR);
  writePNG("firstFrame", imgSize_x, imgSize_y, fb);
  ospUnmapFrameBuffer(fb, framebuffer);

  printf("done\n");
  printf("rendering 10 accumulated frames to accumulatedFrame.ppm...");

  // render 10 more frames, which are accumulated to result in a better
  // converged image
  for (int frames = 0; frames < 10; frames++)
    ospRenderFrameBlocking(framebuffer, renderer, camera, world);

  fb = (uint32_t *)ospMapFrameBuffer(framebuffer, OSP_FB_COLOR);
  writePNG("accumulatedFrame", imgSize_x, imgSize_y, fb);
  ospUnmapFrameBuffer(fb, framebuffer);

  printf("done\n\n");

  // final cleanups
  ospRelease(renderer);
  ospRelease(camera);
  ospRelease(framebuffer);
  ospRelease(world);

  printf("done\n");

  ospShutdown();

#ifdef _WIN32
  if (waitForKey)
  {
    printf("\n\tpress any key to exit");
    _getch();
  }
#endif

  return 0;
}
