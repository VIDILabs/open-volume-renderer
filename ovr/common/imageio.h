#pragma once

#include "math_def.h"
#include <string>

namespace ovr {

template<typename T>
void
save_image(std::string filename, const T* pixels, int width, int height, int channel, int channel_stride);

void
save_image(std::string filename, const uint32_t* pixels /* RGBA8 */, int width, int height);

void
save_image(std::string filename, const vec3f* pixels /* RGB32F */, int width, int height);

void
save_image(std::string filename, const vec4f* pixels /* RGBA32F*/, int width, int height);

}
