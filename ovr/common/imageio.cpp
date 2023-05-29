#include "imageio.h"

#define STB_IMAGE_IMPLEMENTATION
// #include <3rdparty/stb_image.h>
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #include <3rdparty/stb_image_write.h>
#include <stbi/stb_image.h>
#include <stbi/stb_image_write.h>

#define TINYEXR_IMPLEMENTATION
// #include <3rdparty/tinyexr/tinyexr.h>
#include <tinyexr/tinyexr.h>

void
save_exr(const float* data, int width, int height, int nChannels, int channelStride, const char* outfilename)
{
  EXRHeader header;
  InitEXRHeader(&header);

  EXRImage image;
  InitEXRImage(&image);

  image.num_channels = nChannels;

  std::vector<std::vector<float>> images(nChannels);
  std::vector<float*> image_ptr(nChannels);
  for (int i = 0; i < nChannels; ++i) {
    images[i].resize((size_t)width * (size_t)height);
  }

  for (int i = 0; i < nChannels; ++i) {
    image_ptr[i] = images[nChannels - i - 1].data();
  }

  for (size_t i = 0; i < (size_t)width * height; i++) {
    for (int c = 0; c < nChannels; ++c) {
      images[c][i] = data[channelStride * i + c];
    }
  }

  image.images = (unsigned char**)image_ptr.data();
  image.width = width;
  image.height = height;

  header.num_channels = nChannels;
  header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
  // Must be BGR(A) order, since most of EXR viewers expect this channel order.
  strncpy(header.channels[0].name, "B", 255);
  header.channels[0].name[strlen("B")] = '\0';
  if (nChannels > 1) {
    strncpy(header.channels[1].name, "G", 255);
    header.channels[1].name[strlen("G")] = '\0';
  }
  if (nChannels > 2) {
    strncpy(header.channels[2].name, "R", 255);
    header.channels[2].name[strlen("R")] = '\0';
  }
  if (nChannels > 3) {
    strncpy(header.channels[3].name, "A", 255);
    header.channels[3].name[strlen("A")] = '\0';
  }

  header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
  header.requested_pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
  for (int i = 0; i < header.num_channels; i++) {
    header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;          // pixel type of input image
    header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
  }

  const char* err = NULL; // or nullptr in C++11 or later.
  int ret = SaveEXRImageToFile(&image, &header, outfilename, &err);
  if (ret != TINYEXR_SUCCESS) {
    std::string error_message = std::string("Failed to save EXR image: ") + err;
    FreeEXRErrorMessage(err); // free's buffer for an error message
    throw std::runtime_error(error_message);
  }
  printf("Saved exr file. [ %s ] \n", outfilename);

  free(header.channels);
  free(header.pixel_types);
  free(header.requested_pixel_types);
}

void
load_exr(float** data, int* width, int* height, const char* filename)
{
  const char* err = nullptr;

  int ret = LoadEXR(data, width, height, filename, &err);

  if (ret != TINYEXR_SUCCESS) {
    if (err) {
      std::string error_message = std::string("Failed to load EXR image: ") + err;
      FreeEXRErrorMessage(err);
      throw std::runtime_error(error_message);
    }
    else {
      throw std::runtime_error("Failed to load EXR image");
    }
  }
}

std::shared_ptr<uint32_t>
image_to_rgba8(const uint8_t* input, int width, int height, int ch, int ch_stride, bool flip_vertical)
{
  if (ch < 1)
    throw std::runtime_error("less than 1 channel requested, why is that?");
  if (ch > 4)
    throw std::runtime_error("more than 4 channel requested, why is that?");

  if (flip_vertical) {
    uint32_t* output = new uint32_t[(size_t)width * height];
    size_t index = 0;
    for (int j = height - 1; j >= 0; --j) {
      for (int i = 0; i < width; ++i) {
        auto* in = &input[ch_stride * (i + (size_t)j * width)];
        auto* out = (uint8_t*)&output[index++];
        out[0] = in[0];
        out[1] = ch >= 2 ? in[1] : uint8_t(255);
        out[2] = ch >= 3 ? in[2] : uint8_t(255);
        out[3] = ch >= 4 ? in[3] : uint8_t(255);
      }
    }
    return std::shared_ptr<uint32_t>(output, std::default_delete<uint32_t[]>());
  }
  else {
    if (ch == 4 && ch_stride == 4) {
      return std::shared_ptr<uint32_t>((uint32_t*)input, [](uint32_t*) {});
    }
    else {
      uint32_t* output = new uint32_t[(size_t)width * height];
      for (size_t i = 0; i < (size_t)width * height; ++i) {
        auto* in = &input[ch_stride * i];
        auto* out = (uint8_t*)&output[i];
        out[0] = in[0];
        out[1] = ch >= 2 ? in[1] : uint8_t(255);
        out[2] = ch >= 3 ? in[2] : uint8_t(255);
        out[3] = ch >= 4 ? in[3] : uint8_t(255);
      }
      return std::shared_ptr<uint32_t>(new uint32_t[(size_t)width * height], std::default_delete<uint32_t[]>());
    }
  }
}

std::shared_ptr<uint32_t>
image_to_rgba8(const float* input, int width, int height, int ch, int ch_stride, bool flip_vertical)
{
  if (ch < 1)
    throw std::runtime_error("less than 1 channel requested, why is that?");
  if (ch > 4)
    throw std::runtime_error("more than 4 channel requested, why is that?");

  uint32_t* output = new uint32_t[(size_t)width * height];

  if (flip_vertical) {
    size_t index = 0;
    for (int j = height - 1; j >= 0; --j) {
      for (int i = 0; i < width; ++i) {
        auto* in = &input[ch_stride * (i + (size_t)j * width)];
        auto* out = (uint8_t*)&output[index++];
        out[0] = uint8_t(std::clamp(in[0], 0.f, 1.f) * 255.f);
        out[1] = ch >= 2 ? uint8_t(std::clamp(in[1], 0.f, 1.f) * 255.f) : uint8_t(255);
        out[2] = ch >= 3 ? uint8_t(std::clamp(in[2], 0.f, 1.f) * 255.f) : uint8_t(255);
        out[3] = ch >= 4 ? uint8_t(std::clamp(in[3], 0.f, 1.f) * 255.f) : uint8_t(255);
      }
    }
  }
  else {
    for (size_t i = 0; i < (size_t)width * height; ++i) {
      auto* in = &input[ch_stride * i];
      auto* out = (uint8_t*)&output[i];
      out[0] = uint8_t(std::clamp(in[0], 0.f, 1.f) * 255.f);
      out[1] = ch >= 2 ? uint8_t(std::clamp(in[1], 0.f, 1.f) * 255.f) : uint8_t(255);
      out[2] = ch >= 3 ? uint8_t(std::clamp(in[2], 0.f, 1.f) * 255.f) : uint8_t(255);
      out[3] = ch >= 4 ? uint8_t(std::clamp(in[3], 0.f, 1.f) * 255.f) : uint8_t(255);
    }
  }

  return std::shared_ptr<uint32_t>(output, std::default_delete<uint32_t[]>());
}

std::shared_ptr<float>
image_to_rgba32f(const uint8_t* input, int width, int height, int ch, int ch_stride, bool flip_vertical)
{
  if (ch < 1)
    throw std::runtime_error("less than 1 channel requested, why is that?");
  if (ch > 4)
    throw std::runtime_error("more than 4 channel requested, why is that?");

  float* output = new float[4 * (size_t)width * height];

  if (flip_vertical) {
    size_t index = 0;
    for (int j = height - 1; j >= 0; --j) {
      for (int i = 0; i < width; ++i) {
        auto* in = &input[ch_stride * (i + (size_t)j * width)];
        auto* out = (float*)&output[4 * (index++)];
        out[0] = in[0] / 255.f;
        out[1] = ch >= 2 ? in[1] / 255.f : 1.f;
        out[2] = ch >= 3 ? in[2] / 255.f : 1.f;
        out[3] = ch >= 4 ? in[3] / 255.f : 1.f;
      }
    }
  }
  else {
    for (size_t i = 0; i < (size_t)width * height; ++i) {
      auto* in = &input[ch_stride * i];
      auto* out = (float*)&output[4 * i];
      out[0] = in[0] / 255.f;
      out[1] = ch >= 2 ? in[1] / 255.f : 1.f;
      out[2] = ch >= 3 ? in[2] / 255.f : 1.f;
      out[3] = ch >= 4 ? in[3] / 255.f : 1.f;
    }
  }

  return std::shared_ptr<float>(output, std::default_delete<float[]>());
}

std::shared_ptr<float>
image_to_rgba32f(const float* input, int width, int height, int ch, int ch_stride, bool flip_vertical)
{
  if (ch < 1)
    throw std::runtime_error("less than 1 channel requested, why is that?");
  if (ch > 4)
    throw std::runtime_error("more than 4 channel requested, why is that?");

  if (flip_vertical) {
    float* output = new float[4 * (size_t)width * height];
    size_t index = 0;
    for (int j = height - 1; j >= 0; --j) {
      for (int i = 0; i < width; ++i) {
        auto* in = &input[ch_stride * (i + (size_t)j * width)];
        auto* out = (float*)&output[4 * (index++)];
        out[0] = in[0];
        out[1] = ch >= 2 ? in[1] : 1.f;
        out[2] = ch >= 3 ? in[2] : 1.f;
        out[3] = ch >= 4 ? in[3] : 1.f;
      }
    }

    return std::shared_ptr<float>(output, std::default_delete<float[]>());
  }
  else {
    if (ch == 4 && ch_stride == 4) {
      return std::shared_ptr<float>((float*)input, [](float*) {});
    }
    else {
      float* output = new float[4 * (size_t)width * height];
      for (size_t i = 0; i < (size_t)width * height; ++i) {
        auto* in = &input[ch_stride * i];
        auto* out = (float*)&output[4 * i];
        out[0] = in[0];
        out[1] = ch >= 2 ? in[1] : 1.f;
        out[2] = ch >= 3 ? in[2] : 1.f;
        out[3] = ch >= 4 ? in[3] : 1.f;
      }
      return std::shared_ptr<float>(output, std::default_delete<float[]>());
    }
  }
}

namespace ovr {

template<typename T>
void
save_image(std::string filename, const T* pixels, int width, int height, int channel, int channel_stride)
{
  std::string ext = filename.substr(filename.find_last_of(".") + 1);
  if (ext == "exr") {
    auto data = image_to_rgba32f(pixels, width, height, channel, channel_stride, true);
    save_exr(data.get(), width, height, 4, 4, filename.c_str());
  }
  else {
    stbi_flip_vertically_on_write(true);
    auto data = image_to_rgba8(pixels, width, height, channel, channel_stride, false);
    if (ext == "png") {
      stbi_write_png(filename.c_str(), width, height, 4, data.get(), width * 4);
    }
    else if (ext == "jpg") {
      stbi_write_jpg(filename.c_str(), width, height, 4, data.get(), 100);
    }
  }
  std::cout << "saving " << filename << std::endl;
}

template void
save_image<uint8_t>(std::string filename,
                    const uint8_t* image_data,
                    int width,
                    int height,
                    int channel,
                    int channel_stride);

template void
save_image<float>(std::string filename,
                  const float* image_data,
                  int width,
                  int height,
                  int channel,
                  int channel_stride);

void
save_image(std::string filename, const uint32_t* pixels /* RGBA8 */, int width, int height)
{
  save_image(filename, (uint8_t*)pixels, width, height, 4, 4);
}

void
save_image(std::string filename, const vec3f* pixels /* RGB32F */, int width, int height)
{
  save_image(filename, (float*)pixels, width, height, 3, 3);
}

void
save_image(std::string filename, const vec4f* pixels /* RGBA32F*/, int width, int height)
{
  save_image(filename, (float*)pixels, width, height, 4, 4);
}

} // namespace ovr

#include "tinyexr/miniz.c"
