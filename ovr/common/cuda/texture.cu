//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
#include "cuda_misc.h"
#include "cuda_math.h"
#include "texture.h"

#include <assert.h>
#include <math.h>

////////////////////
// Kernel helpers //
////////////////////

#ifdef __NVCC__
#define UTIL_CUDA_HOST_DEVICE __host__ __device__
#else
#define UTIL_CUDA_HOST_DEVICE
#endif

// A key benefit of using the new surface objects is that we don't need any global
// binding points anymore. We can directly pass them as function arguments.

__global__ void
generate_mipmaps_device(uint32_t imageW, uint32_t imageH, cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput)
{
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  float px = 1.0 / float(imageW);
  float py = 1.0 / float(imageH);

  if ((x < imageW) && (y < imageH)) {
    // take the average of 4 samples

    // we are using the normalized access to make sure non-power-of-two textures
    // behave well when downsized.
    float4 color = (tex2D<float4>(mipInput, (x + 0) * px, (y + 0) * py)) + (tex2D<float4>(mipInput, (x + 1) * px, (y + 0) * py)) +
                   (tex2D<float4>(mipInput, (x + 1) * px, (y + 1) * py)) + (tex2D<float4>(mipInput, (x + 0) * px, (y + 1) * py));

    color /= 4.0;

    surf2Dwrite(color, mipOutput, x * sizeof(float4), y);
  }
}

void
generate_mipmaps(cudaMipmappedArray_t mipmapArray, cudaExtent size)
{
  size_t width = size.width;
  size_t height = size.height;

  uint32_t level = 0;

  while (width != 1 || height != 1) {
    width /= 2;
    width = MAX((size_t)1, width);
    height /= 2;
    height = MAX((size_t)1, height);

    cudaArray_t levelFrom;
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));
    cudaArray_t levelTo;
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelTo, mipmapArray, level + 1));

    cudaExtent levelToSize;
    CUDA_CHECK(cudaArrayGetInfo(NULL, &levelToSize, NULL, levelTo));
    assert(levelToSize.width == width);
    assert(levelToSize.height == height);
    assert(levelToSize.depth == 0);

    // generate texture object for reading
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = levelFrom;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.normalizedCoords = true;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;

    cudaTextureObject_t texInput;
    CUDA_CHECK(cudaCreateTextureObject(&texInput, &texRes, &texDesc, NULL));

    // generate surface object for writing
    cudaResourceDesc surfRes;
    memset(&surfRes, 0, sizeof(cudaResourceDesc));
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = levelTo;

    cudaSurfaceObject_t surfOutput;
    CUDA_CHECK(cudaCreateSurfaceObject(&surfOutput, &surfRes));

    // run mipmap kernel
    util::bilinear_kernel(generate_mipmaps_device, 0, /*stream=*/0, (uint32_t)width, (uint32_t)height, surfOutput, texInput);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDestroySurfaceObject(surfOutput));
    CUDA_CHECK(cudaDestroyTextureObject(texInput));

    level++;
  }
}

uint32_t
get_mipmap_levels(cudaExtent size)
{
  size_t sz = MAX(MAX(size.width, size.height), size.depth);
  uint32_t levels = 0;
  while (sz) {
    sz /= 2;
    levels++;
  }
  return levels;
}

cudaTextureObject_t
create_mipmap_rgba32f_texture(void* data, int width, int height)
{
  // how many mipmaps we need
  cudaExtent extent;
  extent.width = width;
  extent.height = height;
  extent.depth = 0;
  uint32_t levels = get_mipmap_levels(extent);

  cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaMipmappedArray_t mipmapArray;
  CUDA_CHECK(cudaMallocMipmappedArray(&mipmapArray, &desc, extent, levels));

  // upload level 0
  cudaArray_t level0;
  CUDA_CHECK(cudaGetMipmappedArrayLevel(&level0, mipmapArray, 0));

  cudaMemcpy3DParms copyParams = { 0 };
  copyParams.srcPtr = make_cudaPitchedPtr(data, width * 4 * sizeof(float), width, height);
  copyParams.dstArray = level0;
  copyParams.extent = extent;
  copyParams.extent.depth = 1;
  copyParams.kind = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpy3D(&copyParams));

  // compute rest of mipmaps based on level 0
  generate_mipmaps(mipmapArray, extent);

  // generate bindless texture object
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(cudaResourceDesc));
  resDesc.resType = cudaResourceTypeMipmappedArray;
  resDesc.res.mipmap.mipmap = mipmapArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(cudaTextureDesc));
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.mipmapFilterMode = cudaFilterModeLinear;
  texDesc.normalizedCoords = true;
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.addressMode[2] = cudaAddressModeClamp;
  texDesc.maxMipmapLevelClamp = float(levels - 1);

  cudaTextureObject_t texture;
  CUDA_CHECK(cudaCreateTextureObject(&texture, &resDesc, &texDesc, NULL));
  return texture;
}

cudaTextureObject_t
create_pitch2d_rgba32f_texture(void* data, int width, int height)
{
  // Second step: create a cuda texture out of this image. It'll be used to generate training 
  // data efficiently on the fly
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = data;
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  resDesc.res.pitch2D.width = width;
  resDesc.res.pitch2D.height = height;
  resDesc.res.pitch2D.pitchInBytes = width * 4 * sizeof(float);

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.normalizedCoords = true;
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.addressMode[2] = cudaAddressModeClamp;

  cudaResourceViewDesc viewDesc;
  memset(&viewDesc, 0, sizeof(viewDesc));
  viewDesc.format = cudaResViewFormatFloat4;
  viewDesc.width = width;
  viewDesc.height = height;

  cudaTextureObject_t texture;
  CUDA_CHECK(cudaCreateTextureObject(&texture, &resDesc, &texDesc, &viewDesc));
  return texture;
}
