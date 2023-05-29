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

#pragma once

// common gdt helper tools
#include "gdt/math/mat.h"
// glfw framework
#include <glad/glad.h>
// it is necessary to include glad before glfw
#include <GLFW/glfw3.h>

#include "camera_frame.h"

/*! \namespace glfwapp */
namespace glfwapp
{
  using namespace gdt;

  struct GLFWindow
  {
    GLFWindow(const std::string &title, int w, int h);
    ~GLFWindow();

    void disableResizing()
    {
      disable_resizing = true;
    }

    /*! put pixels on the screen ... */
    virtual void draw()
    { /* empty - to be subclassed by user */
    }

    /*! callback that window got resized */
    virtual void resize(const vec2i &newSize)
    { /* empty - to be subclassed by user */
    }

    virtual void key(int key, int mods)
    {
    }

    /*! callback that window got resized */
    virtual void mouseMotion(const vec2i &newPos)
    {
    }

    /*! callback that window got resized */
    virtual void mouseButton(int button, int action, int mods)
    {
    }

    inline vec2i getMousePos() const
    {
      double x, y;
      glfwGetCursorPos(handle, &x, &y);
      return vec2i((int)x, (int)y);
    }

    /*! re-render the frame - typically part of draw(), but we keep
      this a separate function so render() can focus on optix
      rendering, and now have to deal with opengl pixel copies
      etc */
    virtual void render()
    { /* empty - to be subclassed by user */
    }

    /*! properly close the window */
    virtual void close()
    { /* empty - to be subclassed by user */
    }

    /*! opens the actual window, and runs the window's events to
      completion. This function will only return once the window
      gets closed */
    void run();

    /*! the glfw window handle */
    GLFWwindow *handle{nullptr};

    /*! a flag to stop window from being automatically resized */
    bool disable_resizing{false};
  };

  // ------------------------------------------------------------------
  /*! abstract base class that allows to manipulate a renderable
    camera */
  struct CameraFrameManip
  {
    CameraFrameManip(CameraFrame *cameraFrame)
        : cameraFrame(cameraFrame)
    {
    }

    /*! this gets called when the user presses a key on the keyboard ... */
    virtual void key(int key, int mods)
    {
      CameraFrame &fc = *cameraFrame;

      switch (key)
      {
      case '+':
      case '=':
        fc.motionSpeed *= 1.5f;
        std::cout << "# viewer: new motion speed is " << fc.motionSpeed << std::endl;
        break;
      case '-':
      case '_':
        fc.motionSpeed /= 1.5f;
        std::cout << "# viewer: new motion speed is " << fc.motionSpeed << std::endl;
        break;
      case 'C':
        std::cout << "(C)urrent camera:" << std::endl;
        std::cout << "- from :" << fc.get_position() << std::endl;
        std::cout << "- poi  :" << fc.get_poi() << std::endl;
        std::cout << "- upVec:" << fc.get_up() << std::endl;
        std::cout << "- frame:" << fc.get_frame() << std::endl;
        break;
      case 'x':
      case 'X':
        fc.set_up(fc.get_up() == vec3f(1, 0, 0) ? vec3f(-1, 0, 0) : vec3f(1, 0, 0));
        break;
      case 'y':
      case 'Y':
        fc.set_up(fc.get_up() == vec3f(0, 1, 0) ? vec3f(0, -1, 0) : vec3f(0, 1, 0));
        break;
      case 'z':
      case 'Z':
        fc.set_up(fc.get_up() == vec3f(0, 0, 1) ? vec3f(0, 0, -1) : vec3f(0, 0, 1));
        break;
      default:
        break;
      }
    }

    virtual void strafe(const vec3f &howMuch)
    {
      cameraFrame->set_position(cameraFrame->get_position() + howMuch);
      cameraFrame->modified = true;
    }

    /*! strafe, in screen space */
    virtual void strafe(const vec2f &howMuch)
    {
      strafe(+howMuch.x * cameraFrame->get_frame_x() - howMuch.y * cameraFrame->get_frame_y());
    }

    virtual void move(const float step) = 0;
    virtual void rotate(const vec2f &curr, const vec2f &prev, const float &speed) = 0;

    // /*! this gets called when the user presses a key on the keyboard ... */
    // virtual void special(int key, const vec2i &where) { };

    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    virtual void mouseDragLeft(const vec2f &curr, const vec2f &prev)
    {
      rotate(curr, prev, degrees_per_drag_fraction);
    }

    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    virtual void mouseDragMiddle(const vec2f &curr, const vec2f &prev)
    {
      const vec2f delta = curr - prev;
      strafe(delta * pixels_per_move * cameraFrame->motionSpeed);
    }

    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    virtual void mouseDragRight(const vec2f &curr, const vec2f &prev)
    {
      const vec2f delta = curr - prev;
      move(delta.y * pixels_per_move * cameraFrame->motionSpeed);
    }

    // /*! mouse button got either pressed or released at given location */
    // virtual void mouseButtonLeft  (const vec2i &where, bool pressed) {}

    // /*! mouse button got either pressed or released at given location */
    // virtual void mouseButtonMiddle(const vec2i &where, bool pressed) {}

    // /*! mouse button got either pressed or released at given location */
    // virtual void mouseButtonRight (const vec2i &where, bool pressed) {}

  protected:
    CameraFrame *cameraFrame;
    const float kbd_rotate_degrees{10.f};
    const float degrees_per_drag_fraction{150.f};
    const float pixels_per_move{10.f};
  };

  struct GLFCameraWindow : public GLFWindow
  {
    GLFCameraWindow(const std::string &title,
                    const vec3f &camera_from,
                    const vec3f &camera_at,
                    const vec3f &camera_up,
                    const float worldScale,
                    int w = 800, int h = 800)
        : GLFWindow(title, w, h),
          cameraFrame(worldScale)
    {
      cameraFrame.setOrientation(camera_from, camera_at, camera_up);
      enableFlyMode();
      enableInspectMode();
    }

    void enableFlyMode();
    void enableInspectMode();

    // /*! put pixels on the screen ... */
    // virtual void draw()
    // { /* empty - to be subclassed by user */ }

    // /*! callback that window got resized */
    // virtual void resize(const vec2i &newSize)
    // { /* empty - to be subclassed by user */ }

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
      case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(GLFWindow::handle, GLFW_TRUE);
      default:
        if (cameraFrameManip)
          cameraFrameManip->key(key, mods);
      }
    }

    /*! callback that window got resized */
    virtual void mouseMotion(const vec2i &currMousePos) override
    {
      vec2i windowSize;
      glfwGetWindowSize(handle, &windowSize.x, &windowSize.y);

      const vec2f curr = vec2f(currMousePos) / vec2f(windowSize);
      const vec2f prev = vec2f(lastMousePos) / vec2f(windowSize);

      if (isPressed.leftButton && cameraFrameManip)
        cameraFrameManip->mouseDragLeft(curr, prev);
      if (isPressed.rightButton && cameraFrameManip)
        cameraFrameManip->mouseDragRight(curr, prev);
      if (isPressed.middleButton && cameraFrameManip)
        cameraFrameManip->mouseDragMiddle(curr, prev);
      lastMousePos = currMousePos;
      /* empty - to be subclassed by user */
    }

    /*! callback that window got resized */
    virtual void mouseButton(int button, int action, int mods) override
    {
      const bool pressed = (action == GLFW_PRESS);
      switch (button)
      {
      case GLFW_MOUSE_BUTTON_LEFT:
        isPressed.leftButton = pressed;
        break;
      case GLFW_MOUSE_BUTTON_MIDDLE:
        isPressed.middleButton = pressed;
        break;
      case GLFW_MOUSE_BUTTON_RIGHT:
        isPressed.rightButton = pressed;
        break;
      }
      lastMousePos = getMousePos();
    }

    // /*! mouse got dragged with left button pressedn, by 'delta'
    //   pixels, at last position where */
    // virtual void mouseDragLeft  (const vec2i &where, const vec2i &delta) {}

    // /*! mouse got dragged with left button pressedn, by 'delta'
    //   pixels, at last position where */
    // virtual void mouseDragRight (const vec2i &where, const vec2i &delta) {}

    // /*! mouse got dragged with left button pressedn, by 'delta'
    //   pixels, at last position where */
    // virtual void mouseDragMiddle(const vec2i &where, const vec2i &delta) {}

    /*! a (global) pointer to the currently active window, so we can
      route glfw callbacks to the right GLFWindow instance (in this
      simplified library we only allow on window at any time) */
    // static GLFWindow *current;

    struct
    {
      bool leftButton{false}, middleButton{false}, rightButton{false};
    } isPressed;
    vec2i lastMousePos = {-1, -1};

    friend struct CameraFrameManip;

    CameraFrame cameraFrame;
    std::shared_ptr<CameraFrameManip> cameraFrameManip;
    std::shared_ptr<CameraFrameManip> inspectModeManip;
    std::shared_ptr<CameraFrameManip> flyModeManip;
  };

  // ------------------------------------------------------------------
  /*! camera manipulator with the following traits
      
    - there is a "point of interest" (POI) that the camera rotates around. 
      (we track this via poi_distance, the point is then thta distance along the fame's z axis)
      
    - we can restrict the minimum and maximum distance camera can be away from this point
      
    - we can specify a max bounding box that this poi can never exceed (validPoiBounds).
      
    - we can restrict whether that point can be moved (by using a single-point valid poi bounds box
      
    - left drag rotates around the object

    - right drag moved forward, backward (within min/max distance bounds)

    - middle drag strafes left/right/up/down (within valid poi bounds)
      
   */
  struct InspectModeManip : public CameraFrameManip
  {

    InspectModeManip(CameraFrame *cameraFrame)
        : CameraFrameManip(cameraFrame)
    {
    }

  private:
    /*! helper function: rotate camera frame by given degrees, then
      make sure the frame, poidistance etc are all properly set,
      the widget gets notified, etc */
    virtual void rotate(const vec2f &curr, const vec2f &prev, const float &speed = 1.f) override
    {
      CameraFrame &fc = *cameraFrame;
      const vec3f poi = fc.get_poi();
      fc.rotate_frame(curr, prev, speed);
      fc.set_position(poi + fc.get_focal_length() * fc.get_frame_z());
      fc.modified = true;
    }

    /*! helper function: move forward/backwards by given multiple of
      motion speed, then make sure the frame, poidistance etc are
      all properly set, the widget gets notified, etc */
    virtual void move(const float step) override
    {
      const vec3f poi = cameraFrame->get_poi();
      // inspectmode can't get 'beyond' the look-at point:
      const float minReqDistance = 0.1f * cameraFrame->motionSpeed;
      cameraFrame->set_focal_length(max(minReqDistance, cameraFrame->get_focal_length() - step));
      cameraFrame->set_position(poi + cameraFrame->get_focal_length() * cameraFrame->get_frame_z());
      cameraFrame->modified = true;
    }
  };

  // ------------------------------------------------------------------
  /*! camera manipulator with the following traits
    - left button rotates the camera around the viewer position
    - middle button strafes in camera plane
    - right buttton moves forward/backwards
   */
  struct FlyModeManip : public CameraFrameManip
  {

    FlyModeManip(CameraFrame *cameraFrame)
        : CameraFrameManip(cameraFrame)
    {
    }

  private:
    /*! helper function: rotate camera frame by given degrees, then
      make sure the frame, poidistance etc are all properly set,
      the widget gets notified, etc */
    virtual void rotate(const vec2f &curr, const vec2f &prev, const float &speed = 1.f) override
    {
      CameraFrame &fc = *cameraFrame;
      fc.rotate_frame(curr, prev, speed);
      fc.modified = true;
    }

    /*! helper function: move forward/backwards by given multiple of
      motion speed, then make sure the frame, poidistance etc are
      all properly set, the widget gets notified, etc */
    virtual void move(const float step) override
    {
      cameraFrame->set_position(cameraFrame->get_position() + step * cameraFrame->get_frame_z());
      cameraFrame->modified = true;
    }
  };

  inline void GLFCameraWindow::enableFlyMode()
  {
    flyModeManip = std::make_shared<FlyModeManip>(&cameraFrame);
    cameraFrameManip = flyModeManip;
  }

  inline void GLFCameraWindow::enableInspectMode()
  {
    inspectModeManip = std::make_shared<InspectModeManip>(&cameraFrame);
    cameraFrameManip = inspectModeManip;
  }

} // namespace glfwapp
