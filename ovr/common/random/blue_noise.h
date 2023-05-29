//. ======================================================================== //
//. Copyright 2019-2022 David Bauer                                          //
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
#ifndef OVR_OPTIX7_NOISE_BLUE_H
#define OVR_OPTIX7_NOISE_BLUE_H

#include <iostream>
#include <fstream>
#include <vector>

#include "noise_files.h"

#include <cuda_misc.h>
#include <cuda_buffer.h>

#ifdef OVR_OPTIX7_MASKING_NOISE_BLUE
#define OVR_NOISE_TILE_SIZE_XY 64
#endif
#ifdef OVR_OPTIX7_MASKING_NOISE_STBN
#define OVR_NOISE_TILE_SIZE_XY 128
#endif

#if !defined(OVR_OPTIX7_MASKING_NOISE_BLUE) && !defined(OVR_OPTIX7_MASKING_NOISE_STBN)
#define OVR_NOISE_TILE_SIZE_XY 16
#endif
#define OVR_NOISE_TILE_SIZE_T 64

namespace ovr {

inline void
load_blue_noise(CUDABuffer& noise_buffer)
{
    // std::cout << "[optix7] loading noise file" << std::endl;
    // std::vector<float> noise;
    // std::string filename;

// #ifdef OVR_OPTIX7_MASKING_NOISE_STBN
//     filename = "./stbn_128x128x64.bin";
// #endif
// #ifdef OVR_OPTIX7_MASKING_NOISE_BLUE
//     filename = "./blue_64x64x64.bin";
// #endif

    // std::ifstream noisefile(filename, std::ios::in|std::ios::binary|std::ios::ate);
    // 
    // if (noisefile.is_open()) {
    //     noisefile.seekg(0, std::ios::end);
	// 	const auto size = noisefile.tellg();
	// 	noisefile.seekg(0, std::ios::beg);
    // 
    //     noise.resize(size);
    //     noisefile.read(reinterpret_cast<char*>(noise.data()), size*sizeof(float));
    // 
    //     noise_buffer.alloc_and_upload_async<float>(noise, /*stream*/0);
    // } else {
    //     throw std::runtime_error("[optix7] could not load noise file");
    // }

#ifdef OVR_OPTIX7_MASKING_NOISE_STBN
    noise_buffer.alloc_and_upload_async<char>((char*)stbn_128x128x64, stbn_128x128x64_size, /*stream*/0);
#endif
#ifdef OVR_OPTIX7_MASKING_NOISE_BLUE
    noise_buffer.alloc_and_upload_async<char>((char*)blue_64x64x64, blue_64x64x64_size, /*stream*/0);
#endif
}

template<typename T>
__global__ void
generate_blue_noise_kernel(const uint32_t n_elements, T* __restrict__ noise, T* __restrict__ out, const uint32_t height, const uint32_t width, int frame_index)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    int x = i % width;
    int y = i / width;

    if (x >= width || y >= height)
        return;

    
    float val = noise[
        (y%OVR_NOISE_TILE_SIZE_XY)*OVR_NOISE_TILE_SIZE_XY*OVR_NOISE_TILE_SIZE_T +
        (x%OVR_NOISE_TILE_SIZE_XY)*OVR_NOISE_TILE_SIZE_T + 
        (frame_index%OVR_NOISE_TILE_SIZE_T)
        ];

    out[i] = val;
}

template<typename T>
inline void
generate_blue_noise(uint32_t n_elements, T* out, const vec2i size, const int frame_index = 0)
{
    static CUDABuffer noise_buffer;

    if (noise_buffer.d_ptr == nullptr) {
        load_blue_noise(noise_buffer);
    }

    generate_blue_noise_kernel<T><<<((n_elements + (OVR_NOISE_TILE_SIZE_XY-1)) / OVR_NOISE_TILE_SIZE_XY), OVR_NOISE_TILE_SIZE_XY>>>(n_elements, (T*)noise_buffer.d_pointer(), out, size.y, size.x, frame_index);
}

}

#endif
