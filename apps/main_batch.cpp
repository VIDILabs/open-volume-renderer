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

#include "cmdline.h"

#include "common/imageio.h"
#include "common/vidi_fps_counter.h"
#include "common/vidi_highperformance_timer.h"
#include "renderer.h"
#include "serializer/serializer.h"

#include <json/json.hpp>

#include <glfwapp/camera_frame.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

using namespace ovr::math;
using ovr::Camera;
using ovr::MainRenderer;


struct CmdArgs : CmdArgsBase {
public:
  args::ArgumentParser parser;
  args::HelpFlag help;
  args::Group required;
  args::Group group_camera;

  args::ValueFlag<std::string> m_scene;
  std::string scene() { return args::get(m_scene); }

  args::ValueFlag<vec3f, args_impl::Vec3fReader> m_camera_from; /*! camera position - *from* where we are looking */
  args::ValueFlag<vec3f, args_impl::Vec3fReader> m_camera_up;   /*! general up-vector */
  args::ValueFlag<vec3f, args_impl::Vec3fReader> m_camera_at;   /*! which point we are looking *at* */
  vec3f camera_from() { return (m_camera_from) ? args::get(m_camera_from) : vec3f(0.f, 0.f, -1000.f); }
  vec3f camera_at()   { return (m_camera_at)   ? args::get(m_camera_at)   : vec3f(0.f, 0.f, 0.f);     }
  vec3f camera_up()   { return (m_camera_up)   ? args::get(m_camera_up)   : vec3f(0.f, 1.f, 0.f);     }
  bool has_camera() { return (m_camera_from); }

  args::ValueFlag<vec2i, args_impl::Vec2iReader> m_fbsize; /*! camera position - *from* where we are looking */
  vec2i fbsize() { return (m_fbsize) ? args::get(m_fbsize) : vec2i(512, 512); }

  args::Flag m_path_tracing;
  bool path_tracing() { return (m_path_tracing); }

  args::ValueFlag<float> m_sampling_rate;
  float sampling_rate() { return (m_sampling_rate) ? args::get(m_sampling_rate) : 1.f; }

  args::ValueFlag<float> m_density_scale;
  float density_scale() { return (m_density_scale) ? args::get(m_density_scale) : 1.f; }

  args::ValueFlag<float> m_camera_speed;
  float camera_speed() { return (m_camera_speed) ? args::get(m_camera_speed) : 1.f; }

  args::ValueFlag<int> m_num_frames;
  int num_frames() { return args::get(m_num_frames); }

  args::ValueFlag<int> m_spp;
  int spp() { return (m_spp) ? args::get(m_spp) : 1; }

  args::ValueFlag<std::string> m_expname;
  std::string expname() { return (m_expname) ? args::get(m_expname) : "output"; }

  args::ValueFlag<std::string> m_device;
  std::string device() { return (m_device) ? args::get(m_device) : "ospray"; }

public:
  CmdArgs(const char *title, int argc, char **argv)
      : parser(title)
      , help(parser, "help", "display the help menu", {'h', "help"})
      , required(parser, "This group is all required:", args::Group::Validators::All)
      , group_camera(parser, "This group is all or none:", args::Group::Validators::AllOrNone)
      , m_scene(required, "string", "the scene to render", {"scene"})
      , m_camera_from(group_camera, "vec3f", "from where we are looking", {"camera-from"})
      , m_camera_at(group_camera, "vec3f", "which point we are looking at", {"camera-at"})
      , m_camera_up(group_camera, "vec3f", "general up-vector", {"camera-up"})
      , m_sampling_rate(parser, "float", "ray marching sampling rate", {"sampling-rate"})
      , m_density_scale(parser, "float", "path tracing density scale", {"density-scale"})
      , m_camera_speed(parser, "float", "camera moving speed", {"camera-speed"})
      , m_num_frames(required, "int", "number of frames to render", {"num-frames"})
      , m_expname(parser, "string", "experiment name", {"exp"})
      , m_path_tracing(parser, "bool", "whether to use path tracing", {"pt", "path-tracing"})
      , m_fbsize(parser, "vec2i", "framebuffer size", {"fbsize"})
      , m_device(parser, "string", "the rendering device to use", {"device"})
      , m_spp(parser, "int", "samples per pixel", {"spp"})
  {
    exec(parser, argc, argv);
  }
};

std::string timestamp(int i)
{
  std::ostringstream ss;
  ss << std::setw(6) << std::setfill('0') << i;
  return ss.str();
}


#if 0

ovr::Scene
create_vorts_scene(int i)
{
  using namespace ovr;

  static std::random_device rng_device;  // Will be used to obtain a seed for the random number engine
  static std::mt19937 rng(rng_device()); // Standard mersenne_twister_engine seeded with rng()
  static std::uniform_int_distribution<> index(0, colormap::name.size() - 1ULL);

  scene::TransferFunction tfn;

  /* 0.0070086f, 12.1394f */
  scene::Volume volume;
  volume.type = scene::Volume::STRUCTURED_REGULAR_VOLUME;
  volume.structured_regular.data = CreateArray3DScalarFromFile(
    std::string("../vortices/vorts") + std::to_string(i) + ".data", vec3i(128, 128, 128), VALUE_TYPE_FLOAT, 0, false);
  volume.structured_regular.grid_origin = -vec3i(128, 128, 128) * 2.f;
  volume.structured_regular.grid_spacing = 4.f;
  tfn.value_range = vec2f(0.0, 12.0);
  tfn.color = CreateColorMap("sequential2/summer");
  tfn.opacity = CreateArray1DScalar(std::vector<float>{ 0.f, 0.f, 0.01f, 0.2f, 0.01f, 0.f, 0.f });

  scene::Model model;
  model.type = scene::Model::VOLUMETRIC_MODEL;
  model.volume_model.volume = volume;
  model.volume_model.transfer_function = tfn;

  scene::Instance instance;
  instance.models.push_back(model);
  instance.transform = affine3f::translate(vec3f(0));

  scene::Scene scene;
  scene.instances.push_back(instance);

  return scene;
}

void
render_a_frame(std::shared_ptr<MainRenderer> ren_ospray,
               std::shared_ptr<MainRenderer> ren_optix7,
               vec2i frame_size,
               int frame_index,
               float max_variance,
               int max_spp,
               glfwapp::CameraFrame camera)
{
  MainRenderer::FrameBufferData pixels;
  float variance;

  ren_optix7->set_camera(camera.get_position(), camera.get_poi(), camera.get_accurate_up());
  ren_optix7->commit();
  ren_optix7->render();
  ren_optix7->mapframe(&pixels);
  ovr::save_image("output" + std::to_string(frame_index) + "grad.exr", (vec3f*)pixels.grad->to_cpu()->data(), /**/
                  frame_size.x, frame_size.y);

  ren_ospray->set_camera(camera.get_position(), camera.get_poi(), camera.get_accurate_up());

  ren_ospray->set_sample_per_pixel(1);
  ren_ospray->commit();
  ren_ospray->render();
  ren_ospray->mapframe(&pixels);
  ovr::save_image("input" + std::to_string(frame_index) + "spp1.exr", (vec4f*)pixels.rgba->to_cpu()->data(),
                  /**/ frame_size.x, frame_size.y);

  ren_ospray->set_sample_per_pixel(2);
  ren_ospray->commit();
  ren_ospray->render();
  ren_ospray->mapframe(&pixels);
  ovr::save_image("input" + std::to_string(frame_index) + "spp2.exr", (vec4f*)pixels.rgba->to_cpu()->data(),
                  /**/ frame_size.x, frame_size.y);

  ren_ospray->set_sample_per_pixel(4);
  ren_ospray->commit();
  ren_ospray->render();
  ren_ospray->mapframe(&pixels);
  ovr::save_image("input" + std::to_string(frame_index) + "spp4.exr", (vec4f*)pixels.rgba->to_cpu()->data(),
                  /**/ frame_size.x, frame_size.y);

  ren_ospray->set_sample_per_pixel(8);
  ren_ospray->commit();
  ren_ospray->render();
  ren_ospray->mapframe(&pixels);
  ovr::save_image("input" + std::to_string(frame_index) + "spp8.exr", (vec4f*)pixels.rgba->to_cpu()->data(),
                  /**/ frame_size.x, frame_size.y);

  ren_ospray->set_sample_per_pixel(max_spp);
  ren_ospray->commit();
  do {
    ren_ospray->render();
    variance = ren_ospray->unsafe_get_variance();
    std::cout << " - " << variance << std::endl;
  } while (variance > max_variance);

  ren_ospray->mapframe(&pixels);
  ovr::save_image("output" + std::to_string(frame_index) + ".var" + std::to_string(variance) + ".exr",
                  (vec4f*)pixels.rgba->to_cpu()->data(), frame_size.x, frame_size.y);
}

#endif

void
render_a_frame(std::shared_ptr<MainRenderer> ren, vec2i frame_size, int frame_index, glfwapp::CameraFrame camera, std::string expname)
{
  MainRenderer::FrameBufferData fbdata;

  ren->set_camera(camera.get_position(), camera.get_poi(), camera.get_accurate_up());
  ren->set_sparse_sampling(false);
  ren->commit();
  ren->render();
  ren->mapframe(&fbdata);

  auto* frame = (vec4f*)fbdata.rgba->to_cpu()->data();
  ovr::save_image(expname + timestamp(frame_index) + ".png", frame, frame_size.x, frame_size.y);
}

/*! main entry point to this example - initially optix, print hello world, then exit */
extern "C" int
main(int ac, char** av)
{
  CmdArgs args("Batch Renderer", ac, av);

  ovr::Scene scene = ovr::scene::create_json_scene(args.scene());

  Camera camera = args.has_camera() ? Camera{
    args.camera_from(), args.camera_at(), args.camera_up()
  } : scene.camera;

  auto camera_frame = glfwapp::CameraFrame(100.f);
  camera_frame.setOrientation(camera.eye, camera.at, camera.up);

  auto ren = create_renderer(args.device());
  ren->set_fbsize(args.fbsize());
  ren->set_frame_accumulation(true);
  ren->set_path_tracing(args.path_tracing());
  ren->set_sample_per_pixel(args.spp());
  ren->set_volume_sampling_rate(args.sampling_rate());
  ren->init(ac, (const char**)av, scene, camera);
  ren->commit();

#if 1
  /* generate secret number between 1 and 10: */
  auto R = camera_frame.get_focal_length();
  auto M = linear3f(camera_frame.get_frame_x(), camera_frame.get_frame_y(), camera_frame.get_frame_z());

  if (args.num_frames() == 1) {
    const auto& expname = args.expname();
    const auto& fbsize  = args.fbsize();

    MainRenderer::FrameBufferData fbdata;

    ren->set_camera(camera_frame.get_position(), camera_frame.get_poi(), camera_frame.get_accurate_up());
    ren->set_sparse_sampling(false);
    ren->commit();

    for (int i = 0; i <  5; ++i) ren->render();

    using Timer  = vidi::details::HighPerformanceTimer;
    {
      Timer timer;
      timer.reset();
      timer.start();
      for (int i = 0; i < 25; ++i) ren->render();
      timer.stop();
      const auto tot = timer.milliseconds() / 1000;
      printf("fps = %f\n", 25 / tot);
    }

    ren->swap();
    ren->mapframe(&fbdata);
    auto* frame = (vec4f*)fbdata.rgba->to_cpu()->data();
    ovr::save_image(expname + timestamp(0) + ".png", frame, fbsize.x, fbsize.y);
  }
  else {
  float t = 0;
  for (int idx = 0; idx < args.num_frames(); ++idx) {
    const vec3f poi = camera_frame.get_poi();
    float theta = sin(13.f * t) * (float)M_PI;
    float phi = cos(5.f * t) * (float)M_PI;
    float r = R * (0.6f + 0.1f * sin(6.f * t));
    float x = r * cos(phi) * sin(theta);
    float y = r * sin(phi) * sin(theta);
    float z = r * cos(theta);
    auto c = xfmVector(M, vec3f(x, y, z));

    printf("camera pos (%f,%f,%f) polar (%f,%f,%f)\n", c.x, c.y, c.z, phi, theta, r);

    camera_frame.setOrientation(c + poi, poi, camera_frame.get_accurate_up());
    t += (args.camera_speed() * (float)M_PI) / args.num_frames();

    render_a_frame(ren, args.fbsize(), idx, camera_frame, args.expname());
  }
  }
#endif

  return 0;
}
