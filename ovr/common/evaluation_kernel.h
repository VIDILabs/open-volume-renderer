//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
#pragma once

#include <tiny-cuda-nn/config.h>
#include <cuda_runtime.h>

template<typename T>
__device__ __forceinline__ T
l1_loss(T prediction, T target)
{
  const T difference = prediction - target;
  return fabsf(difference);
}

template<typename T>
__device__ __forceinline__ T
l2_loss(T prediction, T target)
{
  const T difference = prediction - target;
  return difference * difference;
}

template<typename T>
__device__ __forceinline__ T
relative_l2_loss(T prediction, T target)
{
  const T prediction_sq_plus_epsilon = prediction * prediction + T(0.01);
  const T difference = prediction - target;
  return difference * difference / prediction_sq_plus_epsilon;
}
