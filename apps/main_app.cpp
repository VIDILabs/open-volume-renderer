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
#include "serializer/serializer.h"

#include "imageop.h"

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
using ovr::ImageOp;

using vidi::AsyncLoop;
using vidi::FPSCounter;
using vidi::HistoryFPSCounter;
using vidi::CsvLogger;
using vidi::TransactionalValue;

void help()
{
  std::cout << "renderapp <renderer> <scene-file> <device> <transfer-function>."  << std::endl
            << "\t available renderers: optix7, ospray, gradient, etc."
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

  std::shared_ptr<ImageOp> denoiser;
  MainRenderer::FrameBufferData denoiser_output;

  struct FrameOutputs {
    vec2i size{ 0 };
    vec4f* rgba{ nullptr };
    vec3f* grad{ nullptr };
  };

  const FrameLayer frame_active_layer;
  TransactionalValue<FrameOutputs> frame_outputs; /* wrote by BG, consumed by GUI */
  GLuint frame_texture{ 0 };   /* local to GUI thread */
  vec2i frame_size_local{ 0 }; /* local to GUI thread */

  /* local to GUI thread */
  struct {
    bool global_illumination{ false };
    bool tonemapping { false };
    bool frame_accumulation{ true };
    float volume_sampling_rate{ 1.f };
    float volume_density_scale{ 1.f };
    // float camera_path_speed{ 0.5f };
    int spp{ 1 };
    std::atomic<bool> denoise{ false };
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
    : GLFCameraWindow(title, camera.eye, camera.at, camera.up, scale, width, height)
    , async_rendering_loop(std::bind(&MainWindow::render_background, this))
    , widget(std::bind(&MainWindow::set_transfer_function, this,
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
    renderer->set_focus(vec2f(0.5), 0.06f, 0.07f);
    renderer->set_sparse_sampling(false);
    renderer->set_frame_accumulation(config.frame_accumulation);
    renderer->set_volume_sampling_rate(config.volume_sampling_rate);
    renderer->set_volume_density_scale(config.volume_density_scale);

    denoiser = create_imageop("denoiser");
    denoiser->initialize(0, NULL); // TODO: add a more generic way to manage image ops

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
                                       tfn.tfn_colors.at(3 * i),       // R value
                                       tfn.tfn_colors.at(3 * i + 1),   // G value
                                       tfn.tfn_colors.at(3 * i + 2))); // B value
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

    // commit first to make sure framebuffer data are valid
    renderer->commit();

    // async rendering to the backbuffer
    double render_time = 0.0;
    renderer->render(); 
    render_time = renderer->render_time; 
    variance = renderer->unsafe_get_variance();

    // display the front buffer at the same time
    renderer->mapframe(&renderer_output);
    if (renderer_output.size.long_product() == 0) { return; }

    auto* output = &renderer_output;
    if (config.denoise) {
      denoiser_output.size = renderer_output.size;
      denoiser->resize(denoiser_output.size.x, denoiser_output.size.y);
      denoiser->process(renderer_output.rgba);
      denoiser->map(denoiser_output.rgba);
      output = &denoiser_output;
    }

    // copy to the GUI thread
    FrameOutputs out; 
    out.size = renderer_output.size;
    switch (frame_active_layer) {
    case FRAME_RGBA: out.rgba = (vec4f*)output->rgba->to_cpu()->data(); break;
    case FRAME_GRAD: out.grad = (vec3f*)output->grad->to_cpu()->data(); break;
    default: throw std::runtime_error("something is wrong");
    }
    frame_outputs = out;

    // swap front and back
    renderer->swap();

    // statistics
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
    frame_outputs.update([&](const FrameOutputs& out) {
      glBindTexture(GL_TEXTURE_2D, frame_texture);
      switch (frame_active_layer) {
      case FRAME_RGBA: if (out.rgba) glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, out.size.x, out.size.y, 0, GL_RGBA, GL_FLOAT, out.rgba); break;
      case FRAME_GRAD: if (out.grad) glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,  out.size.x, out.size.y, 0, GL_RGB,  GL_FLOAT, out.grad); break;
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

        static bool denoise = config.denoise;
        if (ImGui::Checkbox("OptiX Denoise", &denoise)) {
          config.denoise = denoise;
        }
        ImGui::SameLine();
        if (ImGui::Checkbox("Tonemapping", &config.tonemapping)) {
          renderer->set_tonemapping(config.tonemapping);
        }

        if (ImGui::Checkbox("Global Illumination", &config.global_illumination)) {
          renderer->set_path_tracing(config.global_illumination);
        }
        ImGui::SameLine();
        if (ImGui::Checkbox("Frame Accumulation", &config.frame_accumulation)) {
          renderer->set_frame_accumulation(config.frame_accumulation);
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

        static float ds = config.volume_density_scale;
        if (ImGui::SliderFloat("Density Scale", &ds, 0.01f, 10.f, "%.3f")) {
          config.volume_density_scale = ds;
          renderer->set_volume_density_scale(config.volume_density_scale);
        }

        widget.build_gui();
      }
      ImGui::End();
      widget.render();
      
      // Device Specific GUIs
      renderer->ui();
    }

    // Performance Graph
    if (gui_performance_enabled) {
      float padding = 2.0f;
      ImGui::SetNextWindowPos(ImVec2(padding, padding), ImGuiCond_Always, ImVec2(0.0f, 1.0f));
      ImGui::SetNextWindowSizeConstraints(ImVec2(300, 200), ImVec2(FLT_MAX, FLT_MAX));
      if (ImGui::Begin("Performance", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration)) {
        if (ImPlot::BeginPlot("##Performance Plot", ImVec2(500,150), ImPlotFlags_NoFrame)) {
          ImPlot::SetupAxes("frame history", "time [ms]", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
          ImPlot::PlotLine("frame time", background_fps.indices.data(), background_fps.frame_time_history.data(), (int)background_fps.frame_time_history.size());
          ImPlot::EndPlot();
        }
      }
      ImGui::End();
    }

    // FPS Counters
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
    renderer->set_fbsize(size);
  }

  /* GUI thread */
  void close() override
  {
    if (async_enabled) async_rendering_loop.stop();
    glDeleteTextures(1, &frame_texture);
  }
};

/*! main entry point to this example - initially optix, print hello world, then exit */
extern "C" int
main(int ac, const char** av)
{
  // -------------------------------------------------------
  // initialize 
  // -------------------------------------------------------

  // something approximating the scale of the world, so the
  // camera knows how much to move for any given user interaction:
  const float worldScale = 100.f;

  // -------------------------------------------------------
  // parse arguments
  // -------------------------------------------------------
  std::string scene_file;
  if (ac < 2) {
    help();
    throw std::runtime_error("no scene file specified");
  }
  scene_file = av[1];

  std::string device = "optix7";
  if (ac >= 3) {
    device = av[2];
  }
  
  std::string tfn = "";
  if (ac >= 4) {
    tfn = av[3];
  }

  // -------------------------------------------------------
  // parse device name
  // -------------------------------------------------------
  ovr::Scene scene;

  // Hack for testing isosurface rendering
  if (device == "isosurface") {
    scene = create_scene_device(scene_file, "ospray");

    const int32_t volume_raw_id = scene.instances[0].models[0].volume_model.volume_texture;
    scene.instances[0].models[0].volume_model.volume_texture = volume_raw_id;

    ovr::scene::Texture volume_tfn;
    volume_tfn.type = ovr::scene::Texture::TRANSFER_FUNCTION_TEXTURE;
    volume_tfn.transfer_function.transfer_function = scene.instances[0].models[0].volume_model.transfer_function;
    volume_tfn.transfer_function.volume_texture = volume_raw_id;
    scene.textures.push_back(volume_tfn);
    const int32_t volume_tfn_id = scene.textures.size() - 1;

    ovr::scene::Material volume_mtl;
    volume_mtl.type = ovr::scene::Material::OBJ_MATERIAL;
    volume_mtl.obj.map_kd = volume_tfn_id;
    scene.materials.push_back(volume_mtl);
    const int32_t volume_mtl_id = scene.materials.size() - 1;

    ovr::scene::Model model;
    model.type = ovr::scene::Model::GEOMETRIC_MODEL;
    model.geometry_model.geometry.type = ovr::scene::Geometry::ISOSURFACE_GEOMETRY;
    model.geometry_model.geometry.isosurfaces.volume_texture = volume_raw_id;
    model.geometry_model.geometry.isosurfaces.isovalues = { 0.5f };
    model.geometry_model.mtl = volume_mtl_id;
    scene.instances[0].models[0] = model;
  }
  else {
    scene = create_scene_device(scene_file, device);
  }

  // -------------------------------------------------------
  // create renderer
  // -------------------------------------------------------
  MainWindow::FrameLayer layer;  
  std::shared_ptr<ovr::MainRenderer> renderer;
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
  MainWindow* window = new MainWindow("OVR", renderer, layer, scene.camera, worldScale, 768, 768, tfn);
  window->run();
  window->close();

  return 0;
}
