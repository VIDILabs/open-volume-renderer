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

//. ======================================================================== //
//. Copyright 2018-2019 Ingo Wald                                            //
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

// This file creates a renderer that only does rendering.

// clang-format off
#include "glfwapp/GLFWApp.h"
#include <glad/glad.h>
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif// clang-format on
#include <implot.h>

#include "common/cross_device_buffer.h"
#include "common/vidi_async_loop.h"
#include "common/vidi_fps_counter.h"
#include "common/vidi_logger.h"
#include "common/vidi_screenshot.h"
#include "common/vidi_transactional_value.h"
#include "renderer.h"

// #define OVR_LOGGING

namespace tfn {
typedef ovr::math::vec2f vec2f;
typedef ovr::math::vec2i vec2i;
typedef ovr::math::vec3f vec3f;
typedef ovr::math::vec3i vec3i;
typedef ovr::math::vec4f vec4f;
typedef ovr::math::vec4i vec4i;
} // namespace tfn
#define TFN_MODULE_EXTERNAL_VECTOR_TYPES
#include <tfn/widget.h>
using tfn::TransferFunctionWidget;

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <mutex>
#include <thread>

using namespace ovr::math;
using ovr::Camera;
using ovr::MainRenderer;
// using ovr::AutomatedCameraPath;

using vidi::AsyncLoop;
using vidi::FPSCounter;
using vidi::HistoryFPSCounter;
using vidi::CsvLogger;
using vidi::TransactionalValue;
// using vidi::EyeMovement;

void help()
{
  std::cout << "renderapp <renderer> <scene-file>."  << std::endl
            << "\t available renderers: optix7-rm, optix7-pt, ospray-rm, osptay-pt, vidi3d, gradient"
            << std::endl;
}

struct MainWindow : public glfwapp::GLFCameraWindow {
public:
  enum FrameLayer {
    FRAME_RGBA = 0,
    FRAME_GRAD,
  };

private:
  std::shared_ptr<MainRenderer> renderer;
  MainRenderer::FrameBufferData renderer_output;

  struct FrameOutputs {
    vec2i size{ 0 };
    vec4f* rgba{ nullptr };
    vec3f* grad{ nullptr };
  };

  const FrameLayer frame_active_layer;
  TransactionalValue<FrameOutputs> frame_outputs; /* wrote by BG, consumed by GUI */
  GLuint frame_texture{ 0 };                        /* local to GUI thread */
  vec2i frame_size_local{ 0 };                      /* local to GUI thread */
  TransactionalValue<vec2i> frame_size_shared{ 0 }; /* wrote by GUI, consumed by BG */

  /* local to GUI thread */
  struct {
    vec2f focus{ 0.5f, 0.5f };
    float focus_scale{ 0.06f };
    float base_noise{ 0.07f };
    bool add_lights{ true };
    bool sparse_sampling{ false };
    bool frame_accumulation{ true };
    float volume_sampling_rate{ 1.f };
    float camera_path_speed{ 0.5f };
    bool global_illumination{ false };
    int spp{ 1 };

    float ambient{ .6f };
    float diffuse{ .9f };
    float specular{ .4f };
    float shininess{ 40.f };

    float radius{ 2415.8f };
    float phi{ 99.53f };
    float theta{ 112.2f };
    float intensity{ 1.f };

  } config;

  bool async_enabled{ true }; /* local to GUI thread */
  AsyncLoop async_rendering_loop;

  TransferFunctionWidget widget;

  std::atomic<float> variance{ 0 }; /* not critical */
  HistoryFPSCounter foreground_fps; /* thread safe */
  HistoryFPSCounter background_fps;
  CsvLogger logger;
  double frame_time = 0.0;

  bool gui_enabled{ true }; /* trigger GUI to show */
  bool gui_performance_enabled{ true }; /* trigger performance GUI to show */

public:
  MainWindow(const std::string& title,
             std::shared_ptr<MainRenderer> renderer,
             FrameLayer layer,
             const Camera& camera,
             const float scale,
             int width,
             int height,
             std::string default_tfn)
    : GLFCameraWindow(title, camera.from, camera.at, camera.up, scale, width, height)
    , async_rendering_loop(std::bind(&MainWindow::render_background, this))
    , widget(std::bind(&MainWindow::set_transfer_function,
                       this,
                       std::placeholders::_1,
                       std::placeholders::_2,
                       std::placeholders::_3))
    , renderer(renderer)
    , frame_active_layer(layer)
  {
    ImPlot::CreateContext();
#ifdef OVR_LOGGING
    logger.initialize({"frame", "fps", "frame_time", "render_time", "inference_time"});
#endif

    /* over write initial values defined internally by devices */
    renderer->set_focus(config.focus, config.focus_scale, config.base_noise);
    renderer->set_sparse_sampling(config.sparse_sampling);
    renderer->set_frame_accumulation(config.frame_accumulation);
    renderer->set_volume_sampling_rate(config.volume_sampling_rate);

    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glGenTextures(1, &frame_texture);

    /* initialize transfer function */
    const auto& tfn = renderer->unsafe_get_tfn();
    if (!tfn.tfn_colors.empty()) {
      std::vector<vec4f> color_controls;
      for (int i = 0; i < tfn.tfn_colors.size() / 3; ++i) {
        color_controls.push_back(vec4f(i / float(tfn.tfn_colors.size() / 3 - 1), /* control point position */
                                       tfn.tfn_colors.at(3 * i),                 //
                                       tfn.tfn_colors.at(3 * i + 1),             //
                                       tfn.tfn_colors.at(3 * i + 2)));           //
      }
      assert(!tfn.tfn_alphas.empty());
      std::vector<vec2f> alpha_controls;
      for (int i = 0; i < tfn.tfn_alphas.size() / 2; ++i) {
        alpha_controls.push_back(vec2f(tfn.tfn_alphas.at(2 * i), tfn.tfn_alphas.at(2 * i + 1)));
      }
      widget.add_tfn(color_controls, alpha_controls, "builtin");
    }
    if (tfn.tfn_value_range.y >= tfn.tfn_value_range.x) {
      widget.set_default_value_range(tfn.tfn_value_range.x, tfn.tfn_value_range.y);
    }

    if (!default_tfn.empty()) {
      widget.load(default_tfn);
    }

    resize(vec2i(0, 0));

    if (async_enabled) {
      render_background(); // warm up
      async_rendering_loop.start();
    }
  }

  ~MainWindow()
  {
    ImPlot::DestroyContext();
  }

  /* background thread */
  void render_background()
  {
    auto start = std::chrono::high_resolution_clock::now(); 

    if (frame_size_shared.update()) {
      frame_outputs.assign([&](FrameOutputs& d) { d.size = vec2i(0, 0); });
      renderer->set_fbsize(frame_size_shared.ref());
    }
    if (frame_size_shared.ref().long_product() == 0)
      return;

    renderer->commit();
    renderer->mapframe(&renderer_output);

    FrameOutputs output;
    {
      switch (frame_active_layer) {
      case FRAME_RGBA: output.rgba = (vec4f*)renderer_output.rgba->to_cpu()->data(); break;
      case FRAME_GRAD: output.grad = (vec3f*)renderer_output.grad->to_cpu()->data(); break;
      default: throw std::runtime_error("something is wrong");
      }
      output.size = frame_size_shared.get();
    }
    frame_outputs = output;

    renderer->swap();

    variance = renderer->unsafe_get_variance();

    double render_time = 0.0;
    renderer->render(); 
    render_time = renderer->render_time; 

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    frame_time += diff.count();

    if (background_fps.count()) {
#ifdef OVR_LOGGING
      logger.log_entry<double>({(double)background_fps.frame, background_fps.fps, frame_time / 10, render_time / 10, 0.0});
#endif
      background_fps.update_history((float)frame_time / 10.f, (float)render_time / 10.f, 0.0f);
      renderer->render_time = 0.0;
      frame_time = 0.0;
    }
  }

  /* GUI thread */
  void render() override
  {
    if (cameraFrame.modified) {
      renderer->set_camera(cameraFrame.get_position(), cameraFrame.get_poi(), cameraFrame.get_accurate_up());
      cameraFrame.modified = false;
    }

    if (!async_enabled) {
      render_background();
    }
  }

  /* GUI thread */
  virtual void key(int key, int mods) override
  {
    switch (key)
    {
      case 'f':
      case 'F':
        std::cout << "Entering 'fly' mode" << std::endl;
        if (flyModeManip)
          cameraFrameManip = flyModeManip;
        break;
      case 'i':
      case 'I':
        std::cout << "Entering 'inspect' mode" << std::endl;
        if (inspectModeManip)
          cameraFrameManip = inspectModeManip;
        break;
      case 'g':
      case 'G':
        std::cout << "Toggling GUI" << std::endl;
        gui_enabled = !gui_enabled;
        break;
      case 'p':
      case 'P':
        std::cout << "Toggling performance GUI" << std::endl;
        gui_performance_enabled = !gui_performance_enabled;
        break;
      case 's':
      case 'S':
        std::cout << "Saving screenshot" << std::endl;
        {
          const FrameOutputs& out = frame_outputs.get();
          switch (frame_active_layer) {
            case FRAME_RGBA: vidi::Screenshot::save(out.rgba, out.size); break;
            case FRAME_GRAD: vidi::Screenshot::save(out.grad, out.size); break;
            default: throw std::runtime_error("something is wrong");
          }
        }
        break;
      case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(GLFWindow::handle, GLFW_TRUE);
      default:
        if (cameraFrameManip)
          cameraFrameManip->key(key, mods);
    }
  }

  /* GUI thread */
  void set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const vec2f& r)
  {
    std::vector<float> cc(c.size() * 3);
    for (int i = 0; i < c.size(); ++i) {
      cc[3 * i + 0] = c[i].x;
      cc[3 * i + 1] = c[i].y;
      cc[3 * i + 2] = c[i].z;
    }
    std::vector<float> oo(o.size() * 2);
    for (int i = 0; i < o.size(); ++i) {
      oo[2 * i + 0] = o[i].x;
      oo[2 * i + 1] = o[i].y;
    }
    renderer->set_transfer_function(cc, oo, r);
  }

  /* GUI thread */
  void draw() override
  {
    glBindTexture(GL_TEXTURE_2D, frame_texture);

    frame_outputs.update([&](const FrameOutputs& out) 
    {
      switch (frame_active_layer) {
      case FRAME_RGBA: glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, out.size.x, out.size.y, 0, GL_RGBA, GL_FLOAT, out.rgba); break;
      case FRAME_GRAD: glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,  out.size.x, out.size.y, 0, GL_RGB, GL_FLOAT, out.grad); break;
      default: throw std::runtime_error("something is wrong");
      }
    });

    const auto& size = frame_size_local;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1, 1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, frame_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, size.x, size.y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float)size.x, 0.f, (float)size.y, -1.f, 1.f);
    glBegin(GL_QUADS);
    {
      glTexCoord2f(0.f, 0.f);
      glVertex3f(0.f, 0.f, 0.f);
      glTexCoord2f(0.f, 1.f);
      glVertex3f(0.f, (float)size.y, 0.f);
      glTexCoord2f(1.f, 1.f);
      glVertex3f((float)size.x, (float)size.y, 0.f);
      glTexCoord2f(1.f, 0.f);
      glVertex3f((float)size.x, 0.f, 0.f);
    }
    glEnd();

    if (gui_enabled) {
      ImGui::SetNextWindowSizeConstraints(ImVec2(450, 400), ImVec2(FLT_MAX, FLT_MAX));
      if (ImGui::Begin("Control Panel", NULL)) {

        bool updated_mat = false;
        updated_mat |= ImGui::SliderFloat("Mat: Ambient", &config.ambient, 0.f, 1.f, "%.3f");
        updated_mat |= ImGui::SliderFloat("Mat: Diffuse", &config.diffuse, 0.f, 1.f, "%.3f");
        updated_mat |= ImGui::SliderFloat("Mat: Specular", &config.specular, 0.f, 1.f, "%.3f");
        updated_mat |= ImGui::SliderFloat("Mat: Shininess", &config.shininess, 0.f, 100.f, "%.3f");

        bool updated_light = false;
        updated_light |= ImGui::SliderFloat("Light: Phi", &config.phi, 0.f, 360.f, "%.2f");
        updated_light |= ImGui::SliderFloat("Light: Theta", &config.theta, 0.f, 360.f, "%.2f");
        updated_light |= ImGui::SliderFloat("Light: Intensity", &config.intensity, 0.f, 2.f, "%.3f");

        bool updated = false;
        updated |= ImGui::SliderFloat("Focus Center X", &config.focus.x, 0.f, 1.f, "%.3f");
        updated |= ImGui::SliderFloat("Focus Center Y", &config.focus.y, 0.f, 1.f, "%.3f");
        updated |= ImGui::SliderFloat("Focus Scale", &config.focus_scale, 0.01f, 1.f, "%.3f");
        updated |= ImGui::SliderFloat("Base Noise", &config.base_noise, 0.01f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        if (updated) {
          renderer->set_focus(config.focus, config.focus_scale, config.base_noise);
        }

        if (updated_mat) {
          renderer->set_mat_ambient(config.ambient);
          renderer->set_mat_diffuse(config.diffuse);
          renderer->set_mat_specular(config.specular);
          renderer->set_mat_shininess(config.shininess);
        }

        if (updated_light) {
          // renderer->set_light_radius(config.radius);
          renderer->set_light_phi(config.phi);
          renderer->set_light_theta(config.theta);
          renderer->set_light_intensity(config.intensity);
        }

        static bool add_lights = config.add_lights;
        if (ImGui::Checkbox("Add Lights", &add_lights)) {
          config.add_lights = add_lights;
          renderer->set_add_lights(config.add_lights);
        }

        static bool sparse_sampling = config.sparse_sampling;
        if (ImGui::Checkbox("Sparse Sampling", &sparse_sampling)) {
          config.sparse_sampling = sparse_sampling;
          renderer->set_sparse_sampling(config.sparse_sampling);
        }

        static bool frame_accumulation = config.frame_accumulation;
        if (ImGui::Checkbox("Frame Accumulation", &frame_accumulation)) {
          config.frame_accumulation = frame_accumulation;
          renderer->set_frame_accumulation(config.frame_accumulation);
        }

        static bool global_illumination = config.global_illumination;
        if (ImGui::Checkbox("Global Illumination", &global_illumination)) {
          config.global_illumination = global_illumination;
          renderer->set_path_tracing(config.global_illumination);
        }

        static int spp = config.spp;
        if (ImGui::SliderInt("Sample Per Pixel", &spp, 1, 32)) {
          config.spp = spp;
          renderer->set_sample_per_pixel(config.spp);
        }

        static float sr = config.volume_sampling_rate;
        if (ImGui::SliderFloat("Sample Rate", &sr, 0.01f, 10.f, "%.3f")) {
          config.volume_sampling_rate = sr;
          renderer->set_volume_sampling_rate(config.volume_sampling_rate);
        }

        widget.build_gui();
      }
      ImGui::End();
      widget.render();
    }

    // Performance Graph
    if (gui_performance_enabled) {
      float padding = 2.0f;
      ImGui::SetNextWindowPos(ImVec2(padding, padding), ImGuiCond_Always, ImVec2(0.0f, 1.0f));
      ImGui::SetNextWindowSizeConstraints(ImVec2(300, 200), ImVec2(FLT_MAX, FLT_MAX));
      if (ImGui::Begin("Performance", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration)) {
        if (ImPlot::BeginPlot("##Performance Plot", ImVec2(500,150), ImPlotFlags_AntiAliased | ImPlotFlags_NoFrame)) {
          ImPlot::SetupAxes("frame history", "time [ms]", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
          ImPlot::PlotLine("frame time", background_fps.indices.data(), background_fps.frame_time_history.data(), (int)background_fps.frame_time_history.size());
          ImPlot::EndPlot();
        }
        ImGui::End();
      }
    }

    if (foreground_fps.count()) {
      std::stringstream title;
      title << std::fixed << std::setprecision(3) << " fg = " << foreground_fps.fps << " fps,";
      title << std::fixed << std::setprecision(3) << " bg = " << background_fps.fps << " fps,";
      title << std::fixed << std::setprecision(3) << " var = " << variance << ".";
      glfwSetWindowTitle(handle, title.str().c_str());
    }
  }

  /* GUI thread */
  void resize(const vec2i& size) override
  {
    frame_size_local = size;
    frame_size_shared = size;
  }

  /* GUI thread */
  void close() override
  {
    if (async_enabled)
      async_rendering_loop.stop();

    glDeleteTextures(1, &frame_texture);
  }
};

/*! main entry point to this example - initially optix, print hello world, then exit */
extern "C" int
main(int ac, const char** av)
{
  // -------------------------------------------------------
  // initialize camera
  // -------------------------------------------------------

  // something approximating the scale of the world, so the
  // camera knows how much to move for any given user interaction:
  const float worldScale = 100.f;

  ovr::Scene scene;
  if (ac < 2) {
    // scene = create_example_scene();
    // scene.camera = { /*from*/ vec3f(0.f, 0.f, -1200.f),
    //                  /* at */ vec3f(0.f, 0.f, 0.f),
    //                  /* up */ vec3f(0.f, 1.f, 0.f) };
    throw std::runtime_error("no scene file specified");
  }
  else {
    scene = ovr::scene::create_scene(std::string(av[1]));

    // TODO hack for testing isosurface rendering

    // const int32_t volume_raw_id = scene.instances[0].models[0].volume_model.volume_texture;
    // scene.instances[0].models[0].volume_model.volume_texture = volume_raw_id;

    // ovr::scene::Texture volume_tfn;
    // volume_tfn.type = ovr::scene::Texture::TRANSFER_FUNCTION_TEXTURE;
    // volume_tfn.transfer_function.transfer_function = scene.instances[0].models[0].volume_model.transfer_function;
    // volume_tfn.transfer_function.volume_texture = volume_raw_id;
    // scene.textures.push_back(volume_tfn);
    // const int32_t volume_tfn_id = scene.textures.size() - 1;

    // ovr::scene::Material volume_mtl;
    // volume_mtl.type = ovr::scene::Material::OBJ_MATERIAL;
    // volume_mtl.obj.map_kd = volume_tfn_id;
    // scene.materials.push_back(volume_mtl);
    // const int32_t volume_mtl_id = scene.materials.size() - 1;

    // ovr::scene::Model model;
    // model.type = ovr::scene::Model::GEOMETRIC_MODEL;
    // model.geometry_model.geometry.type = ovr::scene::Geometry::ISOSURFACE_GEOMETRY;
    // model.geometry_model.geometry.isosurfaces.volume_texture = volume_raw_id;
    // model.geometry_model.geometry.isosurfaces.isovalues = { 0.5f };
    // model.geometry_model.mtl = volume_mtl_id;
    // scene.instances[0].models[0] = model;
  }

  MainWindow::FrameLayer layer;  
  std::shared_ptr<ovr::MainRenderer> renderer;
  std::string device = "optix7";
  if (ac >= 3) {
    device = av[2];
  }

  std::string tfn = "";
  if (ac >= 4) {
    tfn = av[3];
  }

  if (device == "gradient") {
    renderer = create_renderer("optix7");
    renderer->set_path_tracing(false);
    layer = MainWindow::FRAME_GRAD;
  }
  else {
    renderer = create_renderer(device);
    layer = MainWindow::FRAME_RGBA;
  }

  renderer->init(ac, av, scene, scene.camera);

  // -------------------------------------------------------
  // initialize opengl window
  // -------------------------------------------------------
  MainWindow* window = new MainWindow("OVR", renderer, layer, scene.camera, worldScale, 2560, 1440, tfn);
  window->run();
  window->close();

  return 0;
}
