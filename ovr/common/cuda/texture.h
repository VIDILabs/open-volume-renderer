//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
#pragma once
#ifndef HELPER_CUDA_TEXTURE_H
#define HELPER_CUDA_TEXTURE_H

#include <cuda_runtime.h>

cudaTextureObject_t
create_mipmap_rgba32f_texture(void* data, int width, int height);

cudaTextureObject_t
create_pitch2d_rgba32f_texture(void* data, int width, int height);

#endif // HELPER_CUDA_TEXTURE_H
