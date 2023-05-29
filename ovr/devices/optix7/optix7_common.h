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

#ifndef OVR_OPTIX7_OPTIX7_COMMON_H
#define OVR_OPTIX7_OPTIX7_COMMON_H

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <math_def.h>

#include <optix.h>
#include <optix_stubs.h>

// ------------------------------------------------------------------
//
// Host Functions
//
// ------------------------------------------------------------------
#ifdef __cplusplus

#include <cuda_buffer.h>

#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace ovr {
namespace optix7 {

// ------------------------------------------------------------------
// CUDA Texture Setup
// ------------------------------------------------------------------

// template<typename Type>
// inline cudaTextureObject_t
// createCudaTexture(const cudaArray_t& dataArr,
//                   bool normalizedCoords,
//                   cudaTextureReadMode readMode,
//                   cudaTextureFilterMode filterMode = cudaFilterModeLinear,
//                   cudaTextureFilterMode mipmapFilterMode = cudaFilterModeLinear,
//                   cudaTextureAddressMode addressMode = cudaAddressModeClamp,
//                   bool sRGB = false,
//                   float minMipmapLevelClamp = 0,
//                   float maxMipmapLevelClamp = 99,
//                   int maxAnisotropy = 1)
// {
//   cudaTextureObject_t dataTex{};
// 
//   cudaResourceDesc res_desc{};
//   memset(&res_desc, 0, sizeof(cudaResourceDesc));
// 
//   res_desc.resType = cudaResourceTypeArray;
//   res_desc.res.array.array = dataArr;
// 
//   cudaTextureDesc tex_desc{};
//   memset(&tex_desc, 0, sizeof(cudaTextureDesc));
// 
//   tex_desc.addressMode[0] = addressMode;
//   tex_desc.addressMode[1] = addressMode;
//   tex_desc.addressMode[2] = addressMode;
//   tex_desc.filterMode = filterMode;
//   tex_desc.readMode = readMode;
//   tex_desc.normalizedCoords = normalizedCoords ? 1 : 0;
//   tex_desc.maxAnisotropy = maxAnisotropy;
//   tex_desc.maxMipmapLevelClamp = maxMipmapLevelClamp;
//   tex_desc.minMipmapLevelClamp = minMipmapLevelClamp;
//   tex_desc.mipmapFilterMode = mipmapFilterMode;
//   tex_desc.sRGB = sRGB ? 1 : 0;
// 
//   CUDA_CHECK(cudaCreateTextureObject(&dataTex, &res_desc, &tex_desc, nullptr));
// 
//   return dataTex;
// }

// template<typename Type>
// inline cudaArray_t
// createCudaArray3D(void* dataPtr, const int3& dims)
// {
//   cudaArray_t dataArr{};
// 
//   // allocate 3D CUDA array
//   cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<Type>();
//   CUDA_CHECK(cudaMalloc3DArray(&dataArr, &channel_desc, make_cudaExtent(dims.x, dims.y, dims.z)));
// 
//   // copy data to the CUDA array
//   cudaMemcpy3DParms param = { 0 };
//   param.srcPos = make_cudaPos(0, 0, 0);
//   param.dstPos = make_cudaPos(0, 0, 0);
//   param.srcPtr = make_cudaPitchedPtr(dataPtr, dims.x * sizeof(Type), dims.x, dims.y);
//   param.dstArray = dataArr;
//   param.extent = make_cudaExtent(dims.x, dims.y, dims.z);
//   param.kind = cudaMemcpyHostToDevice;
//   CUDA_CHECK(cudaMemcpy3D(&param));
// 
//   return dataArr; // need to then bind the CUDA array to the texture object
// }

// template<typename Type>
// inline cudaArray_t
// createCudaArray1D(const void* dataPtr, const size_t& size)
// {
//   cudaArray_t dataArr{};
// 
//   // Allocate actually a 2D CUDA array of shape N x 1
//   cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<Type>();
//   CUDA_CHECK(cudaMallocArray(&dataArr, &channel_desc, size, 1));
// 
//   // Copy data to the CUDA array
//   const size_t nByte = size * sizeof(Type);
//   CUDA_CHECK(cudaMemcpy2DToArray(dataArr, 0, 0, dataPtr, nByte, nByte, 1, cudaMemcpyHostToDevice));
//   return dataArr;
// 
//   // Need to bind the CUDA array to the texture object
// }

// template<typename Type>
// inline cudaArray_t
// createCudaArray2D(const void* dataPtr, const int width, const int height)
// {
//   cudaArray_t dataArr{};
// 
//   // Allocate actually a 2D CUDA array of shape width x height
//   cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<Type>();
//   CUDA_CHECK(cudaMallocArray(&dataArr, &channel_desc, width, height));
// 
//   // Copy data to the CUDA array
//   CUDA_CHECK(cudaMemcpy2DToArray(dataArr, 0, 0, dataPtr, width * sizeof(float), height, 1, cudaMemcpyHostToDevice));
//   return dataArr;
// 
//   // Need to bind the CUDA array to the texture object
// }

// ------------------------------------------------------------------
// OptiX Helper Functions and Classes
// ------------------------------------------------------------------

#define ALIGN_SBT __align__(OPTIX_SBT_RECORD_ALIGNMENT)

/*! SBT record for a raygen program */
struct ALIGN_SBT RaygenRecord {
  ALIGN_SBT char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
  // just a dummy value - later examples will use more interesting data here
  void* data{};
};

/*! SBT record for a miss program */
struct ALIGN_SBT MissRecord {
  ALIGN_SBT char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
  // just a dummy value - later examples will use more interesting data here
  void* data{};
};

/*! SBT record for a hitgroup program */
struct ALIGN_SBT HitgroupRecord {
  ALIGN_SBT char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
  void* data{};
};

struct ISingleRB {
  virtual ~ISingleRB() = default;
  virtual void create() = 0;
  virtual void resize(size_t& count) = 0;
  virtual void download_async(cudaStream_t stream) = 0;
  virtual void* d_pointer() const = 0;
  virtual void* h_pointer() const = 0;
  virtual void deepcopy(void* dst) = 0;
  virtual void reset_buffer(cudaStream_t stream) = 0;
};

template<typename T>
struct SingleRenderBuffer : ISingleRB {
protected:
  CUDABuffer device_buffer;
  std::vector<T> host_buffer;

public:
  ~SingleRenderBuffer() override
  {
    if (device_buffer.d_pointer())
      device_buffer.free();
  }

  void create() override
  {
    // CUDA_CHECK(cudaStreamCreate(&stream));
  }

  void resize(size_t& count) override
  {
    device_buffer.resize(count * sizeof(T));
    host_buffer.resize(count);
  }

  void download_async(cudaStream_t stream) override
  {
    device_buffer.download_async(host_buffer.data(), host_buffer.size(), stream);
  }

  void* d_pointer() const override
  {
    return (void*)device_buffer.d_pointer();
  }

  void* h_pointer() const override
  {
    return (void*)host_buffer.data();
  }

  void deepcopy(void* dst) override
  {
    std::memcpy(dst, host_buffer.data(), host_buffer.size() * sizeof(T));
  }

  void reset_buffer(cudaStream_t stream) override
  {
    CUDA_CHECK(cudaMemsetAsync((void*)device_buffer.d_pointer(), 0, device_buffer.sizeInBytes, stream));
  }
};

template<typename... Args>
struct MultipleRenderBuffers {
  cudaStream_t stream{};
  std::vector<std::shared_ptr<ISingleRB>> buffers;

  MultipleRenderBuffers()
  {
    buffers = std::vector<std::shared_ptr<ISingleRB>>{ std::make_shared<SingleRenderBuffer<Args>>()... };
  }

  ~MultipleRenderBuffers()
  {
    for (auto& b : buffers)
      b.reset();
  }

  void create()
  {
    CUDA_CHECK(cudaStreamCreate(&stream));
    for (auto& b : buffers)
      b->create();
  }

  void resize(size_t& count)
  {
    for (auto& b : buffers)
      b->resize(count);
  }

  void download_async()
  {
    for (auto& b : buffers)
      b->download_async(stream);
  }

  void* d_pointer(int layout) const
  {
    const int n = sizeof...(Args);
    if (layout >= n)
      throw std::runtime_error(std::string("cannot access the ") + std::to_string(layout) +
                               std::string("-th render buffer, as there are only ") + std::to_string(n) +
                               std::string(" render buffer(s) available."));
    return (void*)buffers[layout]->d_pointer();
  }

  void* h_pointer(int layout) const
  {
    const int n = sizeof...(Args);
    if (layout >= n)
      throw std::runtime_error(std::string("cannot access the ") + std::to_string(layout) +
                               std::string("-th render buffer, as there are only ") + std::to_string(n) +
                               std::string(" render buffer(s) available."));
    return (void*)buffers[layout]->h_pointer();
  }

  void deepcopy(int layout, void* dst)
  {
    const int n = sizeof...(Args);
    if (layout >= n)
      throw std::runtime_error(std::string("cannot access the ") + std::to_string(layout) +
                               std::string("-th render buffer, as there are only ") + std::to_string(n) +
                               std::string(" render buffer(s) available."));
    buffers[layout]->deepcopy(dst);
  }

  void reset()
  {
    for (auto& b : buffers)
      b->reset_buffer(stream);
  }
};

template<typename... Args>
struct DoubleBufferObject {
private:
  using BufferObject = MultipleRenderBuffers<Args...>;

  BufferObject& current_buffer()
  {
    return buffers[current_buffer_index];
  }

  const BufferObject& current_buffer() const
  {
    return buffers[current_buffer_index];
  }

  BufferObject buffers[2];
  int current_buffer_index{ 0 };

  size_t fb_pixel_count{ 0 };
  vec2i fb_size;

public:
  ~DoubleBufferObject() {}

  void create()
  {
    buffers[0].create();
    buffers[1].create();
  }

  void resize(vec2i s)
  {
    fb_size = s;
    fb_pixel_count = (size_t)fb_size.x * fb_size.y;
    {
      buffers[0].resize(fb_pixel_count);
      buffers[1].resize(fb_pixel_count);
    }
  }

  void safe_swap()
  {
    // CUDA_CHECK(cudaStreamSynchronize(current_buffer().stream));
    current_buffer_index = (current_buffer_index + 1) % 2;
  }

  bool empty() const
  {
    return fb_pixel_count == 0;
  }

  const vec2i& size() const
  {
    return fb_size;
  }

  cudaStream_t current_stream()
  {
    return current_buffer().stream;
  }

  void download_async()
  {
    current_buffer().download_async();
  }

  void* device_pointer(int layout) const
  {
    return (void*)current_buffer().d_pointer(layout);
  }

  void* host_pointer(int layout) const
  {
    return (void*)current_buffer().h_pointer(layout);
  }

  void deepcopy(int layout, void* dst)
  {
    current_buffer().deepcopy(layout, dst);
  }

  void reset()
  {
    buffers[0].reset();
    buffers[1].reset();
  }
};

template<typename T>
struct HasSbtEquivalent {
private:
  CUDABuffer sbt_buffer;
  mutable char* sbt_data{ NULL };
  mutable size_t sbt_size{ 0 };

protected:
  T self;

public:
  ~HasSbtEquivalent()
  {
    sbt_buffer.free();
  }

  void UpdateSbtData(cudaStream_t stream)
  {
    sbt_buffer.upload_async(sbt_data, sbt_size, stream);
  }

  void* CreateSbtPtr(cudaStream_t stream)
  {
    sbt_data = (char*)&self;
    sbt_size = sizeof(T);

    /* create and upload to GPU */
    sbt_buffer.alloc_and_upload_async(&self, 1, stream);
    return (void*)sbt_buffer.d_pointer();
  }

  void* GetSbtPtr() const
  {
    return (void*)sbt_buffer.d_pointer();
  }
};

inline OptixTraversableHandle
buildas_exec(OptixDeviceContext optixContext, std::vector<OptixBuildInput>& input, CUDABuffer& asBuffer)
{
  OptixTraversableHandle asHandle{ 0 };

  // ==================================================================
  // BLAS setup
  // ==================================================================

  OptixAccelBuildOptions accelOptions = {};
  accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes blasBufferSizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions, input.data(),
                                           (int)input.size(), // num_build_inputs
                                           &blasBufferSizes));

  // ==================================================================
  // prepare compaction
  // ==================================================================

  CUDABuffer compactedSizeBuffer;
  compactedSizeBuffer.alloc(sizeof(uint64_t));

  OptixAccelEmitDesc emitDesc;
  emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitDesc.result = compactedSizeBuffer.d_pointer();

  // ==================================================================
  // execute build (main stage)
  // ==================================================================

  CUDABuffer tempBuffer;
  tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

  CUDABuffer outputBuffer;
  outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

  OPTIX_CHECK(optixAccelBuild(optixContext, 0 /* stream */, &accelOptions, input.data(), (int)input.size(),
                              tempBuffer.d_pointer(), tempBuffer.sizeInBytes, outputBuffer.d_pointer(),
                              outputBuffer.sizeInBytes, &asHandle, &emitDesc, 1));
  CUDA_SYNC_CHECK();

  // ==================================================================
  // perform compaction
  // ==================================================================

  uint64_t compactedSize;
  compactedSizeBuffer.download_async(&compactedSize, 1, 0 /* stream */);

  asBuffer.alloc(compactedSize);
  OPTIX_CHECK(
    optixAccelCompact(optixContext, 0 /* stream */, asHandle, asBuffer.d_pointer(), asBuffer.sizeInBytes, &asHandle));
  CUDA_SYNC_CHECK();

  // ==================================================================
  // aaaaaand .... clean up
  // ==================================================================

  outputBuffer.free(); // << the UNcompacted, temporary output buffer
  tempBuffer.free();
  compactedSizeBuffer.free();

  return asHandle;
}

} // namespace optix7
} // namespace ovr

#endif // #ifdef __cplusplus
#endif // OVR_OPTIX7_OPTIX7_COMMON_H
