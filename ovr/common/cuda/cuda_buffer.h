//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
#pragma once
#ifndef HELPER_CUDA_BUFFER_H
#define HELPER_CUDA_BUFFER_H

#include "cuda_utils.h"

#include <cuda.h> // for CUdeviceptr
#include <cuda_runtime.h>

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <vector>

// #define CUDA_BUFFER_VERBOSE_MEMORY_ALLOCS

// ------------------------------------------------------------------
// CUDA Buffer
// ------------------------------------------------------------------

/*! simple wrapper for creating, and managing a device-side CUDA buffer */
struct CUDABuffer {
  size_t sizeInBytes = 0;
  void* d_ptr = nullptr;
  bool owned_data = false;

public:
	CUDABuffer() {}

	CUDABuffer& operator=(CUDABuffer&& other) 
  {
		std::swap(sizeInBytes, other.sizeInBytes);
		std::swap(d_ptr, other.d_ptr);
		return *this;
	}

	CUDABuffer(CUDABuffer&& other) 
  {
		*this = std::move(other);
	}

  CUDABuffer(const CUDABuffer &other) : owned_data{false}, sizeInBytes{other.sizeInBytes}, d_ptr{other.d_ptr} {}

	// Frees memory again
	~CUDABuffer() 
  {
		try {
			free(0);
		} catch (std::runtime_error error) {
			// Don't need to report on memory-free problems when the driver is shutting down.
			if (std::string{error.what()}.find("driver shutting down") == std::string::npos) {
				fprintf(stderr, "Could not free memory: %s\n", error.what());
			}
		}
	}

  void set_external(CUDABuffer& other)
  {
    free(0);
    sizeInBytes = other.sizeInBytes;
    d_ptr = other.d_ptr;
    owned_data = false;
  }

  // access raw pointer
  inline const CUdeviceptr& d_pointer() const { return (const CUdeviceptr&)d_ptr; }

  // re-size buffer to given number of bytes
  void resize(size_t size, cudaStream_t stream = 0)
  {
    if (size == sizeInBytes) return;
    free(stream); alloc(size, stream);
  }

  // set memory value
  void memset(int value, cudaStream_t stream = 0)
  {
    CUDA_CHECK(cudaMemsetAsync((void*)d_ptr, value, sizeInBytes, stream));
  }

  // allocate to given number of bytes
  void alloc(size_t size, cudaStream_t stream = 0)
  {
    assert(d_ptr == nullptr);
    this->sizeInBytes = size;
    
    CUDA_CHECK(cudaTrackedMallocAsync((void**)&d_ptr, sizeInBytes, stream));

    owned_data = true;
  }

  // free allocated memory
  void free(cudaStream_t stream = 0)
  {
    if (owned_data && d_ptr) {
      CUDA_CHECK(cudaTrackedFreeAsync(d_ptr, sizeInBytes, stream));
    }
    d_ptr = nullptr;
    sizeInBytes = 0;
  }

  void nullify(cudaStream_t stream)
  {
    if (d_ptr) CUDA_CHECK(cudaMemsetAsync((void*)d_ptr, 0, sizeInBytes, stream));
  }

  template<typename T> void alloc_and_upload(const std::vector<T>& vt)
  {
    resize(vt.size() * sizeof(T));
    upload((const T*)vt.data(), vt.size());
  }

  template<typename T> void alloc_and_upload(const T* ptr, size_t size)
  {
    resize(size * sizeof(T));
    upload((const T*)ptr, size);
  }

  template<typename T>
  void alloc_and_upload_async(const std::vector<T>& vt, cudaStream_t stream = 0)
  {
    resize(vt.size() * sizeof(T), stream);
    upload_async((const T*)vt.data(), vt.size(), stream);
  }

  template<typename T>
  void alloc_and_upload_async(const T* ptr, size_t size, cudaStream_t stream = 0)
  {
    resize(size * sizeof(T), stream);
    upload_async((const T*)ptr, size, stream);
  }

  template<typename T>
  void upload(const T* t, size_t count)
  {
    assert(d_ptr != nullptr);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(cudaMemcpy(d_ptr, (void*)t, count * sizeof(T), cudaMemcpyHostToDevice));
  }

  template<typename T>
  void upload_async(const T* t, size_t count, cudaStream_t stream = 0)
  {
    assert(d_ptr != nullptr);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(cudaMemcpyAsync(d_ptr, (void*)t, count * sizeof(T), cudaMemcpyHostToDevice, stream));
  }

  template<typename T>
  void download(T* t, size_t count)
  {
    assert(d_ptr != nullptr);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(cudaMemcpy((void*)t, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
  }

  template<typename T>
  void download_async(T* t, size_t count, cudaStream_t stream = 0)
  {
    assert(d_ptr != nullptr);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(cudaMemcpyAsync((void*)t, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
  }
};


template<typename T>
struct CUDABufferTyped : private CUDABuffer {
public:
  using CUDABuffer::memset;
  // using CUDABuffer::nullify;

  size_t size() const {
    return sizeInBytes / sizeof(T);
  }

  T* release() {
    sizeInBytes = 0;
    owned_data = false;
    return d_pointer();
  }

  void set_external(CUDABufferTyped<T>& other) {
    CUDABuffer::set_external(other);
  }

  inline const T* d_pointer() const { return (const T*)CUDABuffer::d_pointer(); }
  inline       T* d_pointer()       { return (      T*)CUDABuffer::d_pointer(); }

  void resize(size_t count, cudaStream_t stream = 0) {
    CUDABuffer::resize(count * sizeof(T), stream);
  }

  void alloc(size_t count, cudaStream_t stream = 0) {
    CUDABuffer::alloc(count * sizeof(T), stream);
  }

  void free(cudaStream_t stream = 0) {
    CUDABuffer::free(stream);
  }

  void alloc_and_upload(const std::vector<T>& vt, cudaStream_t stream = 0) {
    CUDABuffer::alloc_and_upload_async(vt, stream);
  }

  void alloc_and_upload(const T* ptr, size_t count, cudaStream_t stream = 0) {
    CUDABuffer::alloc_and_upload_async(ptr, count, stream);
  }

  void upload(const T* ptr, size_t count, cudaStream_t stream = 0) {
    CUDABuffer::upload_async(ptr, count, stream);
  }
  
  void download(T* ptr, size_t count, cudaStream_t stream = 0) {
    CUDABuffer::download_async(ptr, count, stream);
  }

  void download(std::vector<T>& vec, cudaStream_t stream = 0) {
    vec.resize(size());
    CUDABuffer::download_async(vec.data(), vec.size(), stream);
  }
};


// ------------------------------------------------------------------
// CUDA Texture Setup
// ------------------------------------------------------------------

/*! simple wrapper for creating, and managing a device-side CUDA buffer */
struct CUDAArray {
  cudaArray_t handler = nullptr;
  bool owned_data = false;
  int3 shape = make_int3(0, 0, 0);
  int elem_size = 0;
  int n_dims = 0;
  cudaChannelFormatDesc channel_desc{};

public:
	CUDAArray() {}

	CUDAArray& operator=(CUDAArray&& other) {
		std::swap(handler, other.handler);
		std::swap(owned_data, other.owned_data);
    std::swap(shape, other.shape);
    std::swap(elem_size, other.elem_size);
    std::swap(n_dims, other.n_dims);
    std::swap(channel_desc, other.channel_desc);
		return *this;
	}

	CUDAArray(CUDAArray&& other) {
		*this = std::move(other);
	}

  CUDAArray(const CUDAArray &other) 
    : handler{other.handler}
    , owned_data{false}
    , shape{other.shape}
    , elem_size{other.elem_size}
    , n_dims{other.n_dims}
    , channel_desc{other.channel_desc} 
  {}

	// Frees memory again
	~CUDAArray() {
		try {
			free();
		} catch (std::runtime_error error) {
			// Don't need to report on memory-free problems when the driver is shutting down.
			if (std::string{error.what()}.find("driver shutting down") == std::string::npos) {
				fprintf(stderr, "Could not free memory: %s\n", error.what());
			}
		}
	}

  // free allocated memory
  void free() {
    if (owned_data && handler) {
      CUDA_CHECK_NOEXCEPT(cudaTrackedFreeArray(handler));
    }
  }

  // in c++11 using std::enable_if
  template<typename Type>
  void alloc(int length) {
    n_dims = 1;
    elem_size = sizeof(Type);
    shape.x = length;
    channel_desc = cudaCreateChannelDesc<Type>();
    owned_data = true; assert(handler == nullptr);
    CUDA_CHECK(cudaTrackedMallocArray(&handler, &channel_desc, length));
  }

  template<typename Type>
  void alloc(int2 dims) { 
    n_dims = 2;
    elem_size = sizeof(Type);
    shape.x = dims.x; shape.y = dims.y;
    channel_desc = cudaCreateChannelDesc<Type>();
    owned_data = true; assert(handler == nullptr);
    CUDA_CHECK(cudaTrackedMallocArray(&handler, &channel_desc, dims.x, dims.y));
  }

  template<typename Type>
  void alloc(int3 dims) { 
    n_dims = 3;
    elem_size = sizeof(Type);
    shape.x = dims.x; shape.y = dims.y; shape.z = dims.z;
    channel_desc = cudaCreateChannelDesc<Type>();
    owned_data = true; assert(handler == nullptr);
    CUDA_CHECK(cudaTrackedMalloc3DArray(&handler, &channel_desc, make_cudaExtent(dims.x, dims.y, dims.z)));
  }

  void fill(const void* dptr, bool device_pointer = false) { // Copy data to the CUDA array
    const auto kind = device_pointer ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    if (n_dims == 1) {
      const size_t nByte = (size_t)shape.x * elem_size;
      CUDA_CHECK(cudaMemcpy2DToArray(handler, 0, 0, dptr, nByte, nByte, 1, kind));
    }
    else if (n_dims == 2) {
      const size_t nByte = (size_t)shape.x * (size_t)shape.y * elem_size;
      CUDA_CHECK(cudaMemcpy2DToArray(handler, 0, 0, dptr, nByte, nByte, shape.y, kind));
    }
    else if (n_dims == 3) {
      cudaMemcpy3DParms param = { 0 };
      param.srcPos = make_cudaPos(0, 0, 0);
      param.dstPos = make_cudaPos(0, 0, 0);
      param.srcPtr = make_cudaPitchedPtr((void*)dptr, shape.x * elem_size, shape.x, shape.y);
      param.dstArray = handler;
      param.extent = make_cudaExtent(shape.x, shape.y, shape.z);
      param.kind = kind;
      CUDA_CHECK(cudaMemcpy3D(&param));
    }
  }
};

struct CUDATexture {
  cudaTextureObject_t handler = 0;

public:
  CUDATexture() {}

  ~CUDATexture() {
    if (handler) CUDA_CHECK_NOEXCEPT(cudaDestroyTextureObject(handler));
  }

  void bind(const CUDAArray& array, bool trilinear = true, bool normalized_coords = true) {
    const auto filter_mode = trilinear 
      ? cudaFilterModeLinear 
      : cudaFilterModePoint;
    const auto read_mode = (array.channel_desc.x < 32) 
      ? cudaReadModeNormalizedFloat
      : cudaReadModeElementType;

    cudaResourceDesc res_desc{};
    memset(&res_desc, 0, sizeof(cudaResourceDesc));

    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array.handler;

    cudaTextureDesc tex_desc{};
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));

    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.addressMode[2] = cudaAddressModeClamp;
    tex_desc.filterMode = filter_mode;
    tex_desc.readMode = read_mode;
    tex_desc.normalizedCoords = normalized_coords ? 1 : 0;

    CUDA_CHECK(cudaCreateTextureObject(&handler, &res_desc, &tex_desc, nullptr));
  }
};

// TODO support stream ordered operation //

template<typename Type>
inline cudaTextureObject_t
createCudaTexture(const cudaArray_t& dataArr,
                  cudaTextureReadMode readMode,
                  cudaTextureFilterMode filterMode = cudaFilterModeLinear,
                  cudaTextureFilterMode mipmapFilterMode = cudaFilterModeLinear,
                  cudaTextureAddressMode addressMode = cudaAddressModeClamp,
                  bool normalizedCoords = true,
                  bool sRGB = false,
                  int minMipmapLevelClamp = 0,
                  int maxMipmapLevelClamp = 99,
                  int maxAnisotropy = 1)
{
  cudaTextureObject_t tex{};

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
  tex_desc.readMode = readMode;
  tex_desc.normalizedCoords = normalizedCoords ? 1 : 0;
  tex_desc.maxAnisotropy = maxAnisotropy;
  tex_desc.maxMipmapLevelClamp = (float)maxMipmapLevelClamp;
  tex_desc.minMipmapLevelClamp = (float)minMipmapLevelClamp;
  tex_desc.mipmapFilterMode = mipmapFilterMode;
  tex_desc.sRGB = sRGB ? 1 : 0;

  CUDA_CHECK(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr));

  return tex;
}

template<typename Type>
inline cudaSurfaceObject_t
createCudaSurface(const cudaArray_t& dataArr)
{
  cudaSurfaceObject_t surf{};

  cudaResourceDesc res_desc{};
  memset(&res_desc, 0, sizeof(cudaResourceDesc));

  res_desc.resType = cudaResourceTypeArray;
  res_desc.res.array.array = dataArr;

  CUDA_CHECK(cudaCreateSurfaceObject(&surf, &res_desc));

  return surf;
}

template<typename Type>
inline cudaArray_t
createCudaArray3D(void* dataPtr, const int3& dims)
{
  cudaArray_t dataArr{};

  // allocate 3D CUDA array
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<Type>();
  CUDA_CHECK(cudaTrackedMalloc3DArray(&dataArr, &channel_desc, make_cudaExtent(dims.x, dims.y, dims.z)));

  // copy data to the CUDA array
  cudaMemcpy3DParms param = { 0 };
  param.srcPos = make_cudaPos(0, 0, 0);
  param.dstPos = make_cudaPos(0, 0, 0);
  param.srcPtr = make_cudaPitchedPtr(dataPtr, dims.x * sizeof(Type), dims.x, dims.y);
  param.dstArray = dataArr;
  param.extent = make_cudaExtent(dims.x, dims.y, dims.z);
  param.kind = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpy3D(&param));

  return dataArr; // need to then bind the CUDA array to the texture object
}

template<typename Type>
inline cudaArray_t
createCudaArray1D(const void* dataPtr, const size_t& size)
{
  cudaArray_t dataArr{};

  // Allocate actually a 2D CUDA array of shape N x 1
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<Type>();
  CUDA_CHECK(cudaTrackedMallocArray(&dataArr, &channel_desc, size, 1));

  // Copy data to the CUDA array
  const size_t nByte = size * sizeof(Type);
  CUDA_CHECK(cudaMemcpy2DToArray(dataArr, 0, 0, dataPtr, nByte, nByte, 1, cudaMemcpyHostToDevice));
  return dataArr;

  // Need to bind the CUDA array to the texture object
}

template<typename Type>
inline void
fillCudaArray1D(cudaArray_t dataArr, const void* dataPtr, const size_t& size)
{
  // Copy data to the CUDA array
  const size_t nByte = size * sizeof(Type);
  CUDA_CHECK(cudaMemcpy2DToArray(dataArr, 0, 0, dataPtr, nByte, nByte, 1, cudaMemcpyHostToDevice));
}

template<typename Type>
inline cudaArray_t
allocateCudaArray1D(const void* dataPtr, const size_t& size)
{
  cudaArray_t dataArr{};

  // Allocate a 1D CUDA array of size N
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<Type>();
  CUDA_CHECK(cudaTrackedMallocArray(&dataArr, &channel_desc, size));

  // Copy data to the CUDA array
  fillCudaArray1D<Type>(dataArr, dataPtr, size);

  // Need to bind the CUDA array to the texture object
  return dataArr;
}

#endif // HELPER_CUDA_BUFFER_H
