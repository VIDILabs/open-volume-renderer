#pragma once

#include "imageop.h"

struct OptixDenoiser_t;
struct OptixDeviceContext_t;

namespace ovr::optix7 {

struct Optix7Denoiser : ImageOp
{
private:
  CUDABuffer denoiserScratch;
  CUDABuffer denoiserState;
  CUDABuffer outputBuffer;

  OptixDenoiser_t* denoiser{};
  OptixDeviceContext_t* optixContext{};
  
  uint32_t frameID = 25; // => 1.f / blendFactor;
  vec2i frameSize;

  cudaStream_t stream{};

public:
  void initialize(int ac, const char** av) override;
  void resize(int width, int height) override;
  void process(std::shared_ptr<CrossDeviceBuffer>& input) override;
  void map(std::shared_ptr<CrossDeviceBuffer>& output) const override;
  // void reset() override { frameID = 0; }

private:
  void process(vec4f* input);
};

}
