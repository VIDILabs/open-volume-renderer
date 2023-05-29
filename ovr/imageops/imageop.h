#pragma once

#include "ovr/renderer.h"

#include <cross_device_buffer.h>
#include <math_def.h>

namespace ovr {

struct ImageOp {
  virtual void initialize(int ac, const char** av) = 0;
  virtual void process(MainRenderer::FrameBufferData* fb) = 0;
  virtual void resize(int width, int height) = 0;
  virtual void map(std::shared_ptr<CrossDeviceBuffer>& output) const = 0;
};

}
