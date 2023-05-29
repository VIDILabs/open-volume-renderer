//. ======================================================================== //
//. Copyright 2018-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under MIT                                                       //
//. ======================================================================== //
#pragma once

#define OVR_OPTIX7_MASKING_NOISE_STBN
// #define OVR_OPTIX7_MASKING_NOISE_BLUE
// #define OVR_OPTIX7_MASKING_NOISE_UNIFORM

#include "math_def.h"

// the output pointer should be pre-allocated

namespace ovr {

#ifdef OVR_BUILD_CUDA_DEVICES
int64_t
generate_sparse_sampling_mask_d(int32_t* d_output, int frame_index,
                                const vec2i& fbsize,
                                const vec2f& focus_center,
                                float focus_scale,
                                float base_noise);
#endif

int64_t
generate_sparse_sampling_mask_h(int32_t* h_output, int frame_index,
                                const vec2i& fbsize,
                                const vec2f& focus_center,
                                float focus_scale,
                                float base_noise);

}
