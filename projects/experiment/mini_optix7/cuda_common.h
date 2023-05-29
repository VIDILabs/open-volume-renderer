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

#pragma once

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <optix.h>
#include <optix_stubs.h>

// ------------------------------------------------------------------
// Math Functions
// ------------------------------------------------------------------
#include <gdt/math/mat.h>
#include <gdt/math/vec.h>
namespace ovr {
namespace math = gdt;
using vec2f = math::vec2f;
using vec2i = math::vec2i;
using vec3f = math::vec3f;
using vec3i = math::vec3i;
using vec4f = math::vec4f;
using vec4i = math::vec4i;
using affine3f = math::affine3f;
using math::clamp;
using math::max;
using math::min;
using math::xfmNormal;
using math::xfmPoint;
} // namespace ovr

// ------------------------------------------------------------------
//
// Host Functions
//
// ------------------------------------------------------------------
#ifdef __cplusplus

// #define GLFW_INCLUDE_NONE
// #include <GLFW/glfw3.h> // Needs to be included before gl_interop

// #include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

namespace ovr {
namespace host {

// ------------------------------------------------------------------
// CUDA Related Macro
// ------------------------------------------------------------------

#define CUDA_CHECK(call)                                                                            \
  {                                                                                                 \
    cudaError_t rc = cuda##call;                                                                    \
    if (rc != cudaSuccess) {                                                                        \
      std::stringstream txt;                                                                        \
      cudaError_t err = rc; /*cudaGetLastError();*/                                                 \
      txt << "CUDA Error " << cudaGetErrorName(err) << " (msg: " << cudaGetErrorString(err) << ")"; \
      throw std::runtime_error(txt.str());                                                          \
    }                                                                                               \
  }

#define CUDA_CHECK_NOEXCEPT(call) \
  {                               \
    cuda##call;                   \
  }

#define CUDA_SYNC_CHECK()                                                                          \
  {                                                                                                \
    cudaDeviceSynchronize();                                                                       \
    cudaError_t error = cudaGetLastError();                                                        \
    if (error != cudaSuccess) {                                                                    \
      fprintf(stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
      exit(2);                                                                                     \
    }                                                                                              \
  }

#define OPTIX_CHECK(call)                                                                  \
  {                                                                                        \
    OptixResult res = call;                                                                \
    if (res != OPTIX_SUCCESS) {                                                            \
      fprintf(stderr, "OptiX call (%s) failed with %d (line %d)\n", #call, res, __LINE__); \
      exit(2);                                                                             \
    }                                                                                      \
  }

// ------------------------------------------------------------------
// CUDA Buffer
// ------------------------------------------------------------------

/*! simple wrapper for creating, and managing a device-side CUDA buffer */
struct CUDABuffer {
  // access raw pointer
  inline const CUdeviceptr& d_pointer() const
  {
    return (const CUdeviceptr&)d_ptr;
  }

  // re-size buffer to given number of bytes
  void resize(size_t size)
  {
    if (d_ptr)
      free();
    alloc(size);
  }

  // allocate to given number of bytes
  void alloc(size_t size)
  {
    assert(d_ptr == nullptr);
    this->sizeInBytes = size;
    CUDA_CHECK(Malloc((void**)&d_ptr, sizeInBytes));
  }

  // free allocated memory
  void free()
  {
    CUDA_CHECK(Free(d_ptr));
    d_ptr = nullptr;
    sizeInBytes = 0;
  }

  // template<typename T> void alloc_and_upload(const std::vector<T>& vt)
  // {
  //   resize(vt.size() * sizeof(T));
  //   upload((const T*)vt.data(), vt.size());
  // }

  // template<typename T> void alloc_and_upload(const T* ptr, size_t size)
  // {
  //   resize(size * sizeof(T));
  //   upload((const T*)ptr, size);
  // }

  // template<typename T> void upload(const T* t, size_t count)
  // {
  //   assert(d_ptr != nullptr);
  //   assert(sizeInBytes == count * sizeof(T));
  //   CUDA_CHECK(Memcpy(d_ptr, (void*)t, count * sizeof(T), cudaMemcpyHostToDevice));
  // }

  // template<typename T> void download(T* t, size_t count)
  // {
  //   assert(d_ptr != nullptr);
  //   assert(sizeInBytes == count * sizeof(T));
  //   CUDA_CHECK(Memcpy((void*)t, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
  // }

  template<typename T>
  void alloc_and_upload_async(const std::vector<T>& vt, cudaStream_t stream)
  {
    resize(vt.size() * sizeof(T));
    upload_async((const T*)vt.data(), vt.size(), stream);
  }

  template<typename T>
  void alloc_and_upload_async(const T* ptr, size_t size, cudaStream_t stream)
  {
    resize(size * sizeof(T));
    upload_async((const T*)ptr, size, stream);
  }

  template<typename T>
  void upload_async(const T* t, size_t count, cudaStream_t stream)
  {
    assert(d_ptr != nullptr);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(MemcpyAsync(d_ptr, (void*)t, count * sizeof(T), cudaMemcpyHostToDevice, stream));
  }

  template<typename T>
  void download_async(T* t, size_t count, cudaStream_t stream)
  {
    assert(d_ptr != nullptr);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(MemcpyAsync((void*)t, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
  }

  size_t sizeInBytes = 0;
  void* d_ptr = nullptr;
};

// ------------------------------------------------------------------
// CUDA Texture Setup
// ------------------------------------------------------------------

template<typename Type>
inline cudaTextureObject_t
createCudaTexture(const cudaArray_t& dataArr,
                  bool normalizedCoords = true,
                  cudaTextureFilterMode filterMode = cudaFilterModeLinear,
                  cudaTextureFilterMode mipmapFilterMode = cudaFilterModePoint,
                  cudaTextureAddressMode addressMode = cudaAddressModeClamp,
                  bool sRGB = false,
                  int minMipmapLevelClamp = 0,
                  int maxMipmapLevelClamp = 99,
                  int maxAnisotropy = 1)
{
  cudaTextureObject_t dataTex{};

  cudaResourceDesc res_desc{};
  memset(&res_desc, 0, sizeof(cudaResourceDesc));

  res_desc.resType = cudaResourceTypeArray;
  res_desc.res.array.array = dataArr;

  cudaTextureDesc tex_desc{};
  memset(&tex_desc, 0, sizeof(cudaTextureDesc));

  tex_desc.addressMode[0] = addressMode;
  tex_desc.addressMode[1] = addressMode;
  tex_desc.addressMode[2] = addressMode;
  tex_desc.filterMode = filterMode;
  tex_desc.readMode = sizeof(Type) >= 4 ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
  tex_desc.normalizedCoords = normalizedCoords ? 1 : 0;
  tex_desc.maxAnisotropy = maxAnisotropy;
  tex_desc.maxMipmapLevelClamp = maxMipmapLevelClamp;
  tex_desc.minMipmapLevelClamp = minMipmapLevelClamp;
  tex_desc.mipmapFilterMode = mipmapFilterMode;
  tex_desc.sRGB = sRGB ? 1 : 0;

  CUDA_CHECK(CreateTextureObject(&dataTex, &res_desc, &tex_desc, nullptr));

  return dataTex;
}

template<typename Type>
inline cudaArray_t
createCudaArray3D(void* dataPtr, const vec3i& dims)
{
  cudaArray_t dataArr{};

  // allocate 3D CUDA array
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<Type>();
  CUDA_CHECK(Malloc3DArray(&dataArr, &channel_desc, make_cudaExtent(dims.x, dims.y, dims.z)));

  // copy data to the CUDA array
  cudaMemcpy3DParms param = { 0 };
  param.srcPos = make_cudaPos(0, 0, 0);
  param.dstPos = make_cudaPos(0, 0, 0);
  param.srcPtr = make_cudaPitchedPtr(dataPtr, dims.x * sizeof(Type), dims.x, dims.y);
  param.dstArray = dataArr;
  param.extent = make_cudaExtent(dims.x, dims.y, dims.z);
  param.kind = cudaMemcpyHostToDevice;
  CUDA_CHECK(Memcpy3D(&param));

  return dataArr; // need to then bind the CUDA array to the texture object
}

template<typename Type>
inline cudaArray_t
createCudaArray1D(const void* dataPtr, const size_t& size)
{
  cudaArray_t dataArr{};

  // Allocate actually a 2D CUDA array of shape N x 1
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<Type>();
  CUDA_CHECK(MallocArray(&dataArr, &channel_desc, size, 1));

  // Copy data to the CUDA array
  const size_t nByte = size * sizeof(Type);
  CUDA_CHECK(Memcpy2DToArray(dataArr, 0, 0, dataPtr, nByte, nByte, 1, cudaMemcpyHostToDevice));
  return dataArr;

  // Need to bind the CUDA array to the texture object
}

#if 0

enum class CUDAOutputBufferType {
  CUDA_DEVICE = 0, // not preferred, typically slower than ZERO_COPY
  GL_INTEROP = 1,  // single device only, preferred for single device
  ZERO_COPY = 2,   // general case, preferred for multi-gpu if not fully nvlink connected
  CUDA_P2P = 3     // fully connected only, preferred for fully nvlink connected
};

template<typename PIXEL_FORMAT> class CUDAOutputBuffer {
public:
  CUDAOutputBuffer(CUDAOutputBufferType type, int32_t width, int32_t height);
  ~CUDAOutputBuffer();

  void setDevice(int32_t device_idx)
  {
    m_device_idx = device_idx;
  }
  void setStream(CUstream stream)
  {
    m_stream = stream;
  }

  void resize(int32_t width, int32_t height);

  // Allocate or update device pointer as necessary for CUDA access
  PIXEL_FORMAT* map();
  void unmap();

  int32_t width() const
  {
    return m_width;
  }
  int32_t height() const
  {
    return m_height;
  }

  // Get output buffer
  GLuint getPBO();
  void deletePBO();
  PIXEL_FORMAT* getHostPointer();

private:
  void makeCurrent()
  {
    CUDA_CHECK(SetDevice(m_device_idx));
  }

  CUDAOutputBufferType m_type;

  int32_t m_width = 0u;
  int32_t m_height = 0u;

  cudaGraphicsResource* m_cuda_gfx_resource = nullptr;
  GLuint m_pbo = 0u;
  PIXEL_FORMAT* m_device_pixels = nullptr;
  PIXEL_FORMAT* m_host_zcopy_pixels = nullptr;
  std::vector<PIXEL_FORMAT> m_host_pixels;

  CUstream m_stream = 0u;
  int32_t m_device_idx = 0;
};

template<typename PIXEL_FORMAT>
CUDAOutputBuffer<PIXEL_FORMAT>::CUDAOutputBuffer(CUDAOutputBufferType type, int32_t width, int32_t height)
  : m_type(type)
{
  // Output dimensions must be at least 1 in both x and y to avoid an error
  // with cudaMalloc.
#if 0
    if( width < 1 || height < 1 )
    {
        throw sutil::Exception( "CUDAOutputBuffer dimensions must be at least 1 in both x and y." );
    }
#else
  ensureMinimumSize(width, height);
#endif

  // If using GL Interop, expect that the active device is also the display device.
  if (type == CUDAOutputBufferType::GL_INTEROP) {
    int current_device, is_display_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    CUDA_CHECK(cudaDeviceGetAttribute(&is_display_device, cudaDevAttrKernelExecTimeout, current_device));
    if (!is_display_device) {
      throw sutil::Exception("GL interop is only available on display device, please use display device for optimal "
                             "performance.  Alternatively you can disable GL interop with --no-gl-interop and run with "
                             "degraded performance.");
    }
  }
  resize(width, height);
}

template<typename PIXEL_FORMAT> CUDAOutputBuffer<PIXEL_FORMAT>::~CUDAOutputBuffer()
{
  try {
    makeCurrent();
    if (m_type == CUDAOutputBufferType::CUDA_DEVICE || m_type == CUDAOutputBufferType::CUDA_P2P) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_device_pixels)));
    }
    else if (m_type == CUDAOutputBufferType::ZERO_COPY) {
      CUDA_CHECK(cudaFreeHost(reinterpret_cast<void*>(m_host_zcopy_pixels)));
    }
    else if (m_type == CUDAOutputBufferType::GL_INTEROP) {
      // nothing needed
    }

    if (m_pbo != 0u) {
      GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
      GL_CHECK(glDeleteBuffers(1, &m_pbo));
    }
  }
  catch (std::exception& e) {
    std::cerr << "CUDAOutputBuffer destructor caught exception: " << e.what() << std::endl;
  }
}

template<typename PIXEL_FORMAT>
void
CUDAOutputBuffer<PIXEL_FORMAT>::resize(int32_t width, int32_t height)
{
  // Output dimensions must be at least 1 in both x and y to avoid an error
  // with cudaMalloc.
  ensureMinimumSize(width, height);

  if (m_width == width && m_height == height)
    return;

  m_width = width;
  m_height = height;

  makeCurrent();

  if (m_type == CUDAOutputBufferType::CUDA_DEVICE || m_type == CUDAOutputBufferType::CUDA_P2P) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_device_pixels)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_device_pixels), m_width * m_height * sizeof(PIXEL_FORMAT)));
  }

  if (m_type == CUDAOutputBufferType::GL_INTEROP || m_type == CUDAOutputBufferType::CUDA_P2P) {
    // GL buffer gets resized below
    GL_CHECK(glGenBuffers(1, &m_pbo));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_pbo));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(PIXEL_FORMAT) * m_width * m_height, nullptr, GL_STREAM_DRAW));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0u));

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cuda_gfx_resource, m_pbo, cudaGraphicsMapFlagsWriteDiscard));
  }

  if (m_type == CUDAOutputBufferType::ZERO_COPY) {
    CUDA_CHECK(cudaFreeHost(reinterpret_cast<void*>(m_host_zcopy_pixels)));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&m_host_zcopy_pixels), m_width * m_height * sizeof(PIXEL_FORMAT),
                             cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&m_device_pixels),
                                        reinterpret_cast<void*>(m_host_zcopy_pixels), 0 /*flags*/
                                        ));
  }

  if (m_type != CUDAOutputBufferType::GL_INTEROP && m_type != CUDAOutputBufferType::CUDA_P2P && m_pbo != 0u) {
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_pbo));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(PIXEL_FORMAT) * m_width * m_height, nullptr, GL_STREAM_DRAW));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0u));
  }

  if (!m_host_pixels.empty())
    m_host_pixels.resize(m_width * m_height);
}

template<typename PIXEL_FORMAT>
PIXEL_FORMAT*
CUDAOutputBuffer<PIXEL_FORMAT>::map()
{
  if (m_type == CUDAOutputBufferType::CUDA_DEVICE || m_type == CUDAOutputBufferType::CUDA_P2P) {
    // nothing needed
  }
  else if (m_type == CUDAOutputBufferType::GL_INTEROP) {
    makeCurrent();

    size_t buffer_size = 0u;
    CUDA_CHECK(cudaGraphicsMapResources(1, &m_cuda_gfx_resource, m_stream));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&m_device_pixels), &buffer_size,
                                                    m_cuda_gfx_resource));
  }
  else // m_type == CUDAOutputBufferType::ZERO_COPY
  {
    // nothing needed
  }

  return m_device_pixels;
}

template<typename PIXEL_FORMAT>
void
CUDAOutputBuffer<PIXEL_FORMAT>::unmap()
{
  makeCurrent();

  if (m_type == CUDAOutputBufferType::CUDA_DEVICE || m_type == CUDAOutputBufferType::CUDA_P2P) {
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
  }
  else if (m_type == CUDAOutputBufferType::GL_INTEROP) {
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cuda_gfx_resource, m_stream));
  }
  else // m_type == CUDAOutputBufferType::ZERO_COPY
  {
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
  }
}

template<typename PIXEL_FORMAT>
GLuint
CUDAOutputBuffer<PIXEL_FORMAT>::getPBO()
{
  if (m_pbo == 0u)
    GL_CHECK(glGenBuffers(1, &m_pbo));

  const size_t buffer_size = m_width * m_height * sizeof(PIXEL_FORMAT);

  if (m_type == CUDAOutputBufferType::CUDA_DEVICE) {
    // We need a host buffer to act as a way-station
    if (m_host_pixels.empty())
      m_host_pixels.resize(m_width * m_height);

    makeCurrent();
    CUDA_CHECK(
      cudaMemcpy(static_cast<void*>(m_host_pixels.data()), m_device_pixels, buffer_size, cudaMemcpyDeviceToHost));

    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_pbo));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, buffer_size, static_cast<void*>(m_host_pixels.data()), GL_STREAM_DRAW));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
  }
  else if (m_type == CUDAOutputBufferType::GL_INTEROP) {
    // Nothing needed
  }
  else if (m_type == CUDAOutputBufferType::CUDA_P2P) {
    makeCurrent();
    void* pbo_buff = nullptr;
    size_t dummy_size = 0;

    CUDA_CHECK(cudaGraphicsMapResources(1, &m_cuda_gfx_resource, m_stream));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&pbo_buff, &dummy_size, m_cuda_gfx_resource));
    CUDA_CHECK(cudaMemcpy(pbo_buff, m_device_pixels, buffer_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cuda_gfx_resource, m_stream));
  }
  else // m_type == CUDAOutputBufferType::ZERO_COPY
  {
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_pbo));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, buffer_size, static_cast<void*>(m_host_zcopy_pixels), GL_STREAM_DRAW));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
  }

  return m_pbo;
}

template<typename PIXEL_FORMAT>
void
CUDAOutputBuffer<PIXEL_FORMAT>::deletePBO()
{
  GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
  GL_CHECK(glDeleteBuffers(1, &m_pbo));
  m_pbo = 0;
}

template<typename PIXEL_FORMAT>
PIXEL_FORMAT*
CUDAOutputBuffer<PIXEL_FORMAT>::getHostPointer()
{
  if (m_type == CUDAOutputBufferType::CUDA_DEVICE || m_type == CUDAOutputBufferType::CUDA_P2P ||
      m_type == CUDAOutputBufferType::GL_INTEROP) {
    m_host_pixels.resize(m_width * m_height);

    makeCurrent();
    CUDA_CHECK(cudaMemcpy(static_cast<void*>(m_host_pixels.data()), map(), m_width * m_height * sizeof(PIXEL_FORMAT),
                          cudaMemcpyDeviceToHost));
    unmap();

    return m_host_pixels.data();
  }
  else // m_type == CUDAOutputBufferType::ZERO_COPY
  {
    return m_host_zcopy_pixels;
  }
}

#endif

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

} // namespace host
} // namespace ovr

#endif // #ifdef __cplusplus
