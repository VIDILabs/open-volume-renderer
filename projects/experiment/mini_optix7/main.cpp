//. ======================================================================== //
//.                                                                          //
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

#include "renderer.h"

// our helper library for window handling
#include "glfwapp/GLFWApp.h"
#include <GL/gl.h>

namespace tfn {
typedef ovr::math::vec2f vec2f;
typedef ovr::math::vec2i vec2i;
typedef ovr::math::vec3f vec3f;
typedef ovr::math::vec3i vec3i;
typedef ovr::math::vec4f vec4f;
typedef ovr::math::vec4i vec4i;
} // namespace tfn
#define TFN_MODULE_EXTERNAL_VECTOR_TYPES
#include "tfn/widget.h"
using tfn::TransferFunctionWidget;

#define STB_IMAGE_IMPLEMENTATION
#include "3rdparty/stb_image.h"
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdparty/stb_image_write.h"

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

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
using ovr::host::Camera;
using ovr::host::MainRenderer;

struct MainWindow : public glfwapp::GLFCameraWindow {
public:
  GLuint fbTexture = 0;
  vec2i fbSize;
  std::vector<uint32_t> pixels;

  TransferFunctionWidget widget;

  MainRenderer renderer;

public:
  MainWindow(int ac, char** av, const std::string& title, const Camera& camera, const float worldScale)
    : GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale)
    , widget(std::bind(&MainRenderer::set_transfer_function,
                       &renderer,
                       std::placeholders::_1,
                       std::placeholders::_2,
                       std::placeholders::_3))
  {
    renderer.set_scene(ac, av);
    renderer.set_camera(camera);
    renderer.init();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  }

  void render() override
  {
    static bool first = true;

    if (first) {
      if (!renderer.tfn_colors.empty()) {
        widget.add_tfn(renderer.tfn_colors, renderer.tfn_alphas, "builtin");
      }
    }

    if (cameraFrame.modified) {
      renderer.set_camera(cameraFrame.get_position(), cameraFrame.get_poi(), cameraFrame.get_accurate_up());
      cameraFrame.modified = false;
    }

    renderer.render();

    first = false;
  }

  void update_framerate()
  {
    static auto start = std::chrono::high_resolution_clock::now();
    static size_t counter = 0;
    ++counter;
    if (counter == 10) {
      const auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end - start;
      start = end;
      std::stringstream title;
      title << std::fixed << std::setprecision(3) << counter / diff.count() << " fps";
      glfwSetWindowTitle(handle, title.str().c_str());
      counter = 0;
    }
  }

  void draw() override
  {
    renderer.download_pixels(pixels.data());

    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_LIGHTING);

    if (fbTexture == 0)
      glGenTextures(1, &fbTexture);

    glBindTexture(GL_TEXTURE_2D, fbTexture);
    GLenum texFormat = GL_RGBA;
    GLenum texelType = GL_UNSIGNED_BYTE;
    glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y,
                 /* border */ 0, GL_RGBA, texelType, pixels.data());

    glColor3f(1, 1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fbTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, fbSize.x, fbSize.y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);
    glBegin(GL_QUADS);
    {
      glTexCoord2f(0.f, 0.f);
      glVertex3f(0.f, 0.f, 0.f);
      glTexCoord2f(0.f, 1.f);
      glVertex3f(0.f, (float)fbSize.y, 0.f);
      glTexCoord2f(1.f, 1.f);
      glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);
      glTexCoord2f(1.f, 0.f);
      glVertex3f((float)fbSize.x, 0.f, 0.f);
    }
    glEnd();

    widget.build();

    update_framerate();
  }

  void resize(const vec2i& newSize) override
  {
    fbSize = newSize;
    renderer.resize(newSize);
    pixels.resize(newSize.x * newSize.y);
  }
};

/*! main entry point to this example - initially optix, print hello
  world, then exit */
extern "C" int
main(int ac, char** av)
{
  try {
    // -------------------------------------------------------
    // initialize camera
    // -------------------------------------------------------
    Camera camera = { /*from*/ vec3f(0.f, 0.f, -12.f),
                      /* at */ vec3f(0.f, 0.f, 0.f),
                      /* up */ vec3f(0.f, 1.f, 0.f) };
    // Camera camera = { /*from*/ vec3f(-4.50488, 1.47628, -4.94709),
    //                   /* at */ vec3f(0.f, 0.f, 0.f),
    //                   /* up */ vec3f(0.f, 1.f, 0.f) };
    // (C)urrent camera:
    // - from :(0.190742,0.182741,0.964468)
    // - poi  :(0,0,0)
    // - upVec:(0,1,0)
    // - frame:{
    //     vx = (0.980999,0,-0.194012),
    //     vy = (-0.0354544,0.983161,-0.179271),
    //     vz = (0.190742,0.182741,0.964468)
    //   }
    // something approximating the scale of the world, so the
    // camera knows how much to move for any given user interaction:
    const float worldScale = 10.f;

    // -------------------------------------------------------
    // initialize opengl window
    // -------------------------------------------------------
    auto* window = new MainWindow(ac, av, "Optix 7 Renderer", camera, worldScale);

    window->run();
    window->close();
  }
  catch (std::runtime_error& e) {
    std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what() << GDT_TERMINAL_DEFAULT << std::endl;
    exit(1);
  }
  return 0;
}
