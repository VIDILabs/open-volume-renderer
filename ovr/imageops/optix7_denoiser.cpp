#include "optix7_denoiser.h"

#include <optix.h>
#include <optix_stubs.h>
// #include <optix_function_table.h>

// #ifdef __cplusplus
// extern "C" {
// #endif
// extern OptixFunctionTable g_optixFunctionTable;
// #ifdef __cplusplus
// }
// #endif

namespace ovr::optix7 {

static void
context_log_cb(unsigned int level, const char* tag, const char* message, void*)
{
#ifndef NDEBUG
  fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
#endif
}

void 
Optix7Denoiser::initialize(int ac, const char** av)
{
  // -------------------------------------------------------
  // check for available optix7 capable devices
  // -------------------------------------------------------
  cudaFree(0);
  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0)
    throw std::runtime_error("[optix7] no CUDA capable devices found!");
  // std::cout << "[optix7] found " << numDevices << " CUDA devices" << std::endl;

  // if (!g_optixFunctionTable.optixDeviceContextCreate) 
  {
    OPTIX_CHECK(optixInit());
  }

  CUcontext cudaContext{};
  CUresult res = cuCtxGetCurrent(&cudaContext);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "[optix7] Error querying current context: error code %d\n", res);
    throw std::runtime_error("[optix7] Error querying current context");
  }
  OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
  OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
}

void 
Optix7Denoiser::process(std::shared_ptr<CrossDeviceBuffer>& fb) 
{
  process((vec4f*)fb->to_cuda()->data());
}

void 
Optix7Denoiser::map(std::shared_ptr<CrossDeviceBuffer>& fb) const 
{
  fb->set_data((vec4f*)outputBuffer.d_pointer(), outputBuffer.sizeInBytes, CrossDeviceBuffer::DEVICE_CUDA);
}

void 
Optix7Denoiser::process(vec4f* input)
{
  if (frameSize.x == 0 || frameSize.y == 0) { return; }

  OptixDenoiserParams denoiserParams;
  denoiserParams.denoiseAlpha = 1;
  denoiserParams.hdrIntensity = (CUdeviceptr)0;
  if (frameID > 0)
    denoiserParams.blendFactor = 1.f / (frameID);
  else
    denoiserParams.blendFactor = 0.0f;
  // ++frameID;

  // -------------------------------------------------------
  OptixImage2D inputLayer;
  inputLayer.data = (CUdeviceptr)input; // inputBuffer.d_pointer();
  /// Width of the image (in pixels)
  inputLayer.width = frameSize.x;
  /// Height of the image (in pixels)
  inputLayer.height = frameSize.y;
  /// Stride between subsequent rows of the image (in bytes).
  inputLayer.rowStrideInBytes = frameSize.x * sizeof(float4);
  /// Stride between subsequent pixels of the image (in bytes).
  /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
  inputLayer.pixelStrideInBytes = sizeof(float4);
  /// Pixel format.
  inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

  // -------------------------------------------------------
  OptixImage2D outputLayer;
  outputLayer.data = outputBuffer.d_pointer();
  /// Width of the image (in pixels)
  outputLayer.width = frameSize.x;
  /// Height of the image (in pixels)
  outputLayer.height = frameSize.y;
  /// Stride between subsequent rows of the image (in bytes).
  outputLayer.rowStrideInBytes = frameSize.x * sizeof(float4);
  /// Stride between subsequent pixels of the image (in bytes).
  /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
  outputLayer.pixelStrideInBytes = sizeof(float4);
  /// Pixel format.
  outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

#if OPTIX_VERSION >= 70300
  OptixDenoiserGuideLayer denoiserGuideLayer = {};
  OptixDenoiserLayer denoiserLayer = {};
  denoiserLayer.input = inputLayer;
  denoiserLayer.output = outputLayer;
  OPTIX_CHECK(optixDenoiserInvoke(denoiser,
                    /*stream*/ stream,
                    &denoiserParams,
                    denoiserState.d_pointer(),
                    denoiserState.sizeInBytes,
                    &denoiserGuideLayer,
                    &denoiserLayer,
                    1,
                    /*inputOffsetX*/ 0,
                    /*inputOffsetY*/ 0,
                    denoiserScratch.d_pointer(),
                    denoiserScratch.sizeInBytes));
#else
  OPTIX_CHECK(optixDenoiserInvoke(denoiser,
                    /*stream*/ stream,
                    &denoiserParams,
                    denoiserState.d_pointer(),
                    denoiserState.sizeInBytes,
                    &inputLayer,
                    1,
                    /*inputOffsetX*/ 0,
                    /*inputOffsetY*/ 0,
                    &outputLayer,
                    denoiserScratch.d_pointer(),
                    denoiserScratch.sizeInBytes));
#endif
}

void 
Optix7Denoiser::resize(int width, int height)
{
  const vec2i& newSize = vec2i(width, height);

  if (frameSize == newSize) return;
  frameSize = newSize;

  if (denoiser) {
    OPTIX_CHECK(optixDenoiserDestroy(denoiser));
  };

  // inputBuffer.resize(newSize.x * newSize.y * sizeof(vec4f), stream);
  outputBuffer.resize(newSize.x * newSize.y * sizeof(vec4f), stream);

  // ------------------------------------------------------------------
  // create the denoiser:
  OptixDenoiserOptions denoiserOptions = {};

#if OPTIX_VERSION >= 70300
  OPTIX_CHECK(optixDenoiserCreate(optixContext, OPTIX_DENOISER_MODEL_KIND_LDR, &denoiserOptions, &denoiser));
#else
  denoiserOptions.inputKind = OPTIX_DENOISER_INPUT_RGB;

#if OPTIX_VERSION < 70100
  // these only exist in 7.0, not 7.1
  denoiserOptions.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
#endif
  OPTIX_CHECK(optixDenoiserCreate(optixContext, &denoiserOptions, &denoiser));
  OPTIX_CHECK(optixDenoiserSetModel(denoiser, OPTIX_DENOISER_MODEL_KIND_LDR, NULL, 0));
#endif

  // .. then compute and allocate memory resources for the denoiser
  OptixDenoiserSizes denoiserReturnSizes;
  OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, newSize.x, newSize.y, &denoiserReturnSizes));

#if OPTIX_VERSION < 70100
  denoiserScratch.resize(denoiserReturnSizes.recommendedScratchSizeInBytes, stream);
#else
  denoiserScratch.resize(std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes, denoiserReturnSizes.withoutOverlapScratchSizeInBytes), stream);
#endif
  denoiserState.resize(denoiserReturnSizes.stateSizeInBytes, stream);

  // ------------------------------------------------------------------
  OPTIX_CHECK(optixDenoiserSetup(denoiser, stream, newSize.x, newSize.y,
                                 denoiserState.d_pointer(),
                                 denoiserState.sizeInBytes,
                                 denoiserScratch.d_pointer(),
                                 denoiserScratch.sizeInBytes));
}

}