//. ======================================================================== //
//. Copyright 2018-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under MIT                                                       //
//. ======================================================================== //

#pragma once

#ifdef OVR_BUILD_CUDA_DEVICES
#include <cuda_buffer.h>
#endif

#include "math_def.h"

#include <vector>

namespace ovr {

struct CrossDeviceBuffer {

public:
  enum Device {
    DEVICE_INVALID,
    DEVICE_CPU,
#ifdef OVR_BUILD_CUDA_DEVICES
    DEVICE_CUDA,
#endif    
  };

private:
  Device device{ DEVICE_INVALID };

  void* buffer_ptr{ nullptr };
  size_t buffer_size_in_bytes{ 0 };

  std::vector<char> owned_buffer_cpu;
  
#ifdef OVR_BUILD_CUDA_DEVICES
  CUDABuffer owned_buffer_cuda;
#endif
  
  void resize(size_t num_bytes)
  {
    switch (device) {

    case (DEVICE_CPU):
      owned_buffer_cpu.resize(num_bytes);
      std::cout << "allocate CPU buffer" << std::endl;
      buffer_ptr = owned_buffer_cpu.data();
      break;

#ifdef OVR_BUILD_CUDA_DEVICES
    case (DEVICE_CUDA):
      owned_buffer_cuda.resize(num_bytes);
      std::cout << "allocate CUDA buffer" << std::endl;
      buffer_ptr = (void*)owned_buffer_cuda.d_pointer();
      break;
#endif

    default: throw std::runtime_error("incorrect device identifier");
    }

    buffer_size_in_bytes = num_bytes;
  }

  void cleanup_other_devices()
  {
    if (device != DEVICE_CPU)
      owned_buffer_cpu.clear();
#ifdef OVR_BUILD_CUDA_DEVICES
    if (device != DEVICE_CUDA && owned_buffer_cuda.d_pointer())
      owned_buffer_cuda.free();
#endif
  }

  void cleanup_current_devices()
  {
    if (device == DEVICE_CPU)
      owned_buffer_cpu.clear();
#ifdef OVR_BUILD_CUDA_DEVICES
    if (device == DEVICE_CUDA && owned_buffer_cuda.d_pointer())
      owned_buffer_cuda.free();
#endif
  }

public:
  ~CrossDeviceBuffer()
  {
    device = DEVICE_INVALID;
    cleanup_other_devices();
  }

  CrossDeviceBuffer() = default;

  CrossDeviceBuffer(size_t num_bytes, Device device) : device(device)
  {
    resize(num_bytes);
  }

  CrossDeviceBuffer(void* data, size_t num_bytes, Device device)
    : device(device), buffer_ptr(data), buffer_size_in_bytes(num_bytes)
  {
  }

  void set_data(size_t num_bytes, Device d)
  {
    device = d;
    resize(num_bytes);
  }

  void set_data(void* data, size_t num_bytes, Device d)
  {
    device = d;
    buffer_ptr = data;
    buffer_size_in_bytes = num_bytes;

    cleanup_current_devices();
  }

  template<typename T>
  void set_data(std::vector<T>& vec, Device d)
  {
    device = d;
    buffer_ptr = vec.data();
    buffer_size_in_bytes = vec.size() * sizeof(T);

    cleanup_current_devices();
  }

  CrossDeviceBuffer* to_cpu(bool cleanup = false)
  {
    if (device != DEVICE_CPU) {
      if (owned_buffer_cpu.size() != buffer_size_in_bytes) {
        owned_buffer_cpu.resize(buffer_size_in_bytes);
        std::cout << "allocate CPU buffer" << std::endl;
      }

      /* CUDA to CPU */
#ifdef OVR_BUILD_CUDA_DEVICES
      if (device == DEVICE_CUDA) {
        assert(buffer_ptr != nullptr);
        CUDA_CHECK(cudaMemcpy((void*)owned_buffer_cpu.data(), buffer_ptr, buffer_size_in_bytes, cudaMemcpyDeviceToHost));
      } else
#endif
      {
        throw std::runtime_error("incorrect device identifier");
      }

      device = DEVICE_CPU;
      buffer_ptr = owned_buffer_cpu.data();
    }

    /* cleanup other devices */
    if (cleanup) {
      cleanup_other_devices();
    }

    return this;
  }

#ifdef OVR_BUILD_CUDA_DEVICES
  CrossDeviceBuffer* to_cuda(bool cleanup = false)
  {
    if (device != DEVICE_CUDA) {
      if (owned_buffer_cuda.sizeInBytes != buffer_size_in_bytes) {
        owned_buffer_cuda.resize(buffer_size_in_bytes);
        std::cout << "allocate CUDA buffer" << std::endl;
      }

      /* CPU to CUDA */
      if (device == DEVICE_CPU) {
        assert(buffer_ptr != nullptr);
        owned_buffer_cuda.upload((char*)buffer_ptr, buffer_size_in_bytes);
      }
      else {
        throw std::runtime_error("incorrect device identifier");
      }

      device = DEVICE_CUDA;
      buffer_ptr = (void*)owned_buffer_cuda.d_pointer();
    }

    /* cleanup other devices */
    if (cleanup) {
      cleanup_other_devices();
    }

    return this;
  }
#endif

  bool is_on_cpu()
  {
    return device == DEVICE_CPU;
  }

#ifdef OVR_BUILD_CUDA_DEVICES
  bool is_on_cuda()
  {
    return device == DEVICE_CUDA;
  }
#endif

  void* data()
  {
    return buffer_ptr;
  }
};

} // namespace vidi
