/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *//*
 */

/** @file   random.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Collection of CUDA kernels related to random numbers
 */

#pragma once
#ifndef OVR_OPTIX7_RANDOM_RANDOM_H
#define OVR_OPTIX7_RANDOM_RANDOM_H

#include "pcg32.h"

namespace ovr { namespace random {

#define IQ_DEFAULT_STATE 0x853c49e6748fea9bULL

/// Based on https://www.iquilezles.org/www/articles/sfrand/sfrand.htm
struct iqrand {
  /// Initialize the pseudorandom number generator with default seed
  __both__ iqrand() : state((uint32_t)IQ_DEFAULT_STATE) {}

  /// Initialize the pseudorandom number generator with the \ref seed() function
  __both__ iqrand(uint32_t initstate) : state(initstate) {}

  /// Generate a single precision floating point value on the interval [0, 1)
  __both__ float next_float()
  {
    union {
      float fres;
      unsigned int ires;
    };

    state *= 16807;
    ires = ((((unsigned int)state) >> 9) | 0x3f800000);
    return fres - 1.0f;
  }

  uint32_t state; // RNG state.  All values are possible.
};

using default_rng_t = pcg32;

template<typename T, typename RNG, size_t N_TO_GENERATE, typename F>
__global__ void
generate_random_kernel(const size_t n_elements, RNG rng, T* __restrict__ out, const F transform)
{
  const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t n_threads = blockDim.x * gridDim.x;

  rng.advance(i * N_TO_GENERATE);

#pragma unroll
  for (size_t j = 0; j < N_TO_GENERATE; ++j) {
    const size_t idx = i + n_threads * j;
    if (idx >= n_elements) {
      return;
    }

    out[idx] = transform((T)rng.next_float());
  }
}

template<typename T, typename RNG, typename F>
inline void
generate_random(cudaStream_t stream, RNG& rng, size_t n_elements, T* out, F&& transform)
{
  static constexpr size_t N_TO_GENERATE = 4;

  size_t n_threads = misc::div_round_up(n_elements, N_TO_GENERATE);
  generate_random_kernel<T, RNG, N_TO_GENERATE>
    <<<misc::n_blocks_linear(n_threads), misc::n_threads_linear, 0, stream>>>(n_elements, rng, out, transform);

  rng.advance(n_elements);
}

template<typename T, typename RNG>
inline void
generate_random_uniform(cudaStream_t stream,
                        RNG& rng,
                        size_t n_elements,
                        T* out,
                        const T lower = (T)0.0,
                        const T upper = (T)1.0)
{
  generate_random(stream, rng, n_elements, out,
                  [upper, lower] __device__(T val) { return val * (upper - lower) + lower; });
}

template<typename T, typename RNG>
inline void
generate_random_uniform(RNG& rng, size_t n_elements, T* out, const T lower = (T)0.0, const T upper = (T)1.0)
{
  generate_random_uniform(nullptr, rng, n_elements, out, lower, upper);
}

template<typename T, typename RNG>
inline void
generate_random_logistic(cudaStream_t stream,
                         RNG& rng,
                         size_t n_elements,
                         T* out,
                         const T mean = (T)0.0,
                         const T stddev = (T)1.0)
{
  generate_random(stream, rng, n_elements, out,
                  [mean, stddev] __device__(T val) { return (T)logit(val) * stddev * 0.551328895f + mean; });
}

template<typename T, typename RNG>
inline void
generate_random_logistic(RNG& rng, size_t n_elements, T* out, const T mean = (T)0.0, const T stddev = (T)1.0)
{
  generate_random_logistic(nullptr, rng, n_elements, out, mean, stddev);
}

//---------------------------------------------------------------------------//
// TEA - Random numbers based on Tiny Encryption Algorithm                   //
//---------------------------------------------------------------------------//
// https://github.com/openvkl/openvkl/blob/b2cfd8ab94420489e2ed9a25fb0d512c2577e5de/examples/interactive/renderer/Random.ih
// https://github.com/NVIDIA/OptiX_Apps/blob/master/apps/intro_driver/shaders/random_number_generators.h

struct RandomTEA {
private:
  // Tiny Encryption Algorithm (TEA) to calculate a the seed per launch index and iteration.
  // This results in a ton of integer instructions! Use the smallest N necessary.
  template<unsigned int N>
  __forceinline__ __device__ void tea(unsigned int& _v0, unsigned int& _v1)
  {
    unsigned int v0 = _v0; // Operate on registers to avoid slow down!
    unsigned int v1 = _v1;
    unsigned int sum = 0;

    for (int i = 0; i < N; i++) {
      sum += 0x9e3779b9;
      v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + sum) ^ ((v1 >> 5) + 0xc8013ea4);
      v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + sum) ^ ((v0 >> 5) + 0x7e95761e);
    }

    _v0 = v0;
    _v1 = v1;
  }

private:
  unsigned int v0, v1;

public:
  __forceinline__ __device__ RandomTEA(const unsigned int idx, const unsigned int seed)
  {
    this->v0 = idx;
    this->v1 = seed;
  }
    
  __forceinline__ __device__ float get_float()
  {
    return get_floats().x;
  }

  __forceinline__ __device__ float2 get_floats()
  {
    tea<16>(this->v0, this->v1);
    const float tofloat = 2.3283064365386962890625e-10f; // 1/2^32
    return make_float2(this->v0 * tofloat, this->v1 * tofloat);
  }
};

}
}

#endif // OVR_OPTIX7_RANDOM_RANDOM_H
