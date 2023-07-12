//. ======================================================================= //
//.                                                                         //
//. Copyright 2019-2022 Qi Wu                                               //
//.                                                                         //
//. Licensed under the MIT License                                          //
//.                                                                         //
//. ======================================================================= //
// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <glad/glad.h>

#include "GLFWApp.h"

#include <imconfig.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>

#ifndef GLFWAPP_USE_OPENGL2
#include <imgui_impl_opengl3.h>
#else
#include <imgui_impl_opengl2.h>
#endif

#include <cassert>

/*! \namespace glfwapp */
namespace glfwapp
{
  using namespace gdt;

  static void glfw_error_callback(int error, const char *description)
  {
    fprintf(stderr, "Error: %s\n", description);
  }

  GLFWindow::~GLFWindow()
  {
#ifndef GLFWAPP_USE_OPENGL2
    ImGui_ImplOpenGL3_Shutdown();
#else
    ImGui_ImplOpenGL2_Shutdown();
#endif
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(handle);
    glfwTerminate();
  }

  GLFWindow::GLFWindow(const std::string &title, int w, int h)
  {
    glfwSetErrorCallback(glfw_error_callback);
#if __APPLE__
    glfwInitHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);
#endif

    if (!glfwInit())
      exit(EXIT_FAILURE);

#ifndef GLFWAPP_USE_OPENGL2
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
#endif

    handle = glfwCreateWindow(w, h, title.c_str(), NULL, NULL);
    if (!handle)
    {
      glfwTerminate();
      exit(EXIT_FAILURE);
    }

    glfwSetWindowUserPointer(handle, this);
    glfwMakeContextCurrent(handle);
    glfwSwapInterval(1);

    // Load GLAD symbols
    int err = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) == 0;
    if (err)
      throw std::runtime_error("Failed to initialize OpenGL loader!");

    /* Setup Dear ImGui context */
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    // Setup Dear ImGui style
    ImGui::StyleColorsDark(); // or ImGui::StyleColorsClassic();
    // Initialize Dear ImGui
    ImGui_ImplGlfw_InitForOpenGL(handle, true);

#ifndef GLFWAPP_USE_OPENGL2
    const char* glsl_version = "#version 150"; // GL 3.3 + GLSL 150
    ImGui_ImplOpenGL3_Init(glsl_version);
#else
    ImGui_ImplOpenGL2_Init();
#endif
  }

  /*! callback for a window resizing event */
  static void glfwindow_reshape_cb(GLFWwindow *window, int width, int height)
  {
    GLFWindow *gw = static_cast<GLFWindow *>(glfwGetWindowUserPointer(window));
    assert(gw);
    gw->resize(vec2i(width, height));
    // assert(GLFWindow::current);
    //   GLFWindow::current->resize(vec2i(width,height));
  }

  /*! callback for a key press */
  static void glfwindow_key_cb(GLFWwindow *window, int key, int scancode, int action, int mods)
  {
    /* action handled by ImGui */
    if (ImGui::GetIO().WantCaptureKeyboard)
      return;

    GLFWindow *gw = static_cast<GLFWindow *>(glfwGetWindowUserPointer(window));
    assert(gw);
    if (action == GLFW_PRESS)
    {
      gw->key(key, mods);
    }
  }

  /*! callback for _moving_ the mouse to a new position */
  static void glfwindow_mouseMotion_cb(GLFWwindow *window, double x, double y)
  {
    GLFWindow *gw = static_cast<GLFWindow *>(glfwGetWindowUserPointer(window));
    assert(gw);
    gw->mouseMotion(vec2i((int)x, (int)y));
  }

  /*! callback for pressing _or_ releasing a mouse button*/
  static void glfwindow_mouseButton_cb(GLFWwindow *window, int button, int action, int mods)
  {
    /* action handled by ImGui */
    if (ImGui::GetIO().WantCaptureMouse)
      return;

    GLFWindow *gw = static_cast<GLFWindow *>(glfwGetWindowUserPointer(window));
    assert(gw);
    gw->mouseButton(button, action, mods);
  }

  void GLFWindow::run()
  {
    int width, height;
    glfwGetFramebufferSize(handle, &width, &height);
    resize(vec2i(width, height));

    // glfwSetWindowUserPointer(window, GLFWindow::current);
    glfwSetFramebufferSizeCallback(handle, glfwindow_reshape_cb);
    glfwSetMouseButtonCallback(handle, glfwindow_mouseButton_cb);
    glfwSetKeyCallback(handle, glfwindow_key_cb);
    glfwSetCursorPosCallback(handle, glfwindow_mouseMotion_cb);

    while (!glfwWindowShouldClose(handle))
    {
      /* render OptiX */
      render();

#ifndef GLFWAPP_USE_OPENGL2
      ImGui_ImplOpenGL3_NewFrame();
#else
      ImGui_ImplOpenGL2_NewFrame();
#endif
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      /* render OpenGL */
      draw();

      ImGui::Render();
#ifndef GLFWAPP_USE_OPENGL2
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
#else
      ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
#endif

      glfwSwapBuffers(handle);
      glfwPollEvents();
    }
  }

  // GLFWindow *GLFWindow::current = nullptr;

} // namespace glfwapp
