//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
#pragma once

#include "cuda/cuda_utils.h"
#include <gdt/math/vec.h>
#include <thrust/reduce.h>
#include <thrust/tuple.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
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

// make a private version of thrust::plus to avoid template instantiation conflicts ...
namespace {

template<typename T>
struct maximum_op {
  typedef T first_argument_type;
  typedef T second_argument_type;
  typedef T result_type;
  __host__ __device__ constexpr T operator() (const T& lhs, const T& rhs) const { return lhs < rhs ? rhs : lhs; }
}; // end maximum

template<typename T>
struct minimum_op {
  typedef T first_argument_type;
  typedef T second_argument_type;
  typedef T result_type;
  __host__ __device__ constexpr T operator() (const T& lhs, const T& rhs) const { return lhs < rhs ? lhs : rhs; }
}; // end minimum

template<typename T>
struct plus {
  typedef T first_argument_type;
  typedef T second_argument_type;
  typedef T result_type;
  __host__ __device__ constexpr T operator() (const T &lhs, const T &rhs) const { return lhs + rhs; }
}; // end plus

template<typename T>
T parallel_sum_gpu(const T* __restrict__ data, size_t count, cudaStream_t stream = nullptr) 
{
  const auto begin = thrust::device_ptr<const T>(data);
  const auto end = begin + count;
  T ret = thrust::reduce(thrust::cuda::par.on(stream), begin, end, T(0), plus<T>());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return ret;
}

template<typename T>
void parallel_minmax_gpu(const T* __restrict__ data, size_t count, T& init_min, T& init_max, cudaStream_t stream = nullptr) 
{
  const auto begin = thrust::device_ptr<const T>(data);
  const auto end = begin + count;
  init_min = thrust::reduce(thrust::cuda::par.on(stream), begin, end, init_min, minimum_op<T>());
  init_max = thrust::reduce(thrust::cuda::par.on(stream), begin, end, init_max, maximum_op<T>());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}

typedef std::function<float*(gdt::vec3i offset, gdt::vec3i block)> accessor_t;

inline float eval_mse(gdt::vec3i dims, accessor_t acc_pred, accessor_t acc_targ)
{
  const gdt::vec3i batch = gdt::min(gdt::vec3i(4096,16,16), dims);
  const auto N = util::next_multiple<size_t>(batch.long_product(), 256);

  float* d_loss = nullptr;
  CUDA_CHECK(cudaMalloc(&d_loss, N * sizeof(float)));

  // compute MSE
  double error_sum = 0.0;

  for (int z = 0; z < dims.z; z += batch.z) {
  for (int y = 0; y < dims.y; y += batch.y) {
  for (int x = 0; x < dims.x; x += batch.x) {
    const auto offset = gdt::vec3i(x, y, z);
    const auto block = gdt::min(batch, dims - offset);
    const auto count = block.long_product();
    if (count == 0) continue;

    float* d_targ = acc_targ(offset, block);
    float* d_pred = acc_pred(offset, block);

    // squared error
    util::parallel_for_gpu(0, 0, count, [d_pred, d_targ, d_loss] __device__ (size_t i) {
      d_loss[i] = l2_loss((float)d_pred[i], (float)d_targ[i]);
    });
    CUDA_CHECK(cudaStreamSynchronize(0));

    // compute total error
    error_sum += parallel_sum_gpu(d_loss, count);
  }
  }
  }

  CUDA_CHECK(cudaFree(d_loss));

  return (float)(error_sum / dims.long_product());
}

template<typename T>
static double eval_mse_full(gdt::vec3i dims, const T* _pred, const T* _targ) 
{
  size_t count = dims.long_product();
  thrust::device_ptr<const T> pred = thrust::device_pointer_cast(_pred);
  thrust::device_ptr<const T> targ = thrust::device_pointer_cast(_targ);  // origin
  auto begin = thrust::make_zip_iterator(thrust::make_tuple(pred, targ));
  auto end = thrust::make_zip_iterator(thrust::make_tuple(pred + count, targ + count));
  // clang-format off
  auto err2      = []  __host__ __device__(thrust::tuple<T,T> t)  { T f = thrust::get<0>(t) - thrust::get<1>(t); return f * f; };
  auto sum_err2  = thrust::transform_reduce(begin, end, err2, 0.0f, plus<T>());
  // clang-format on
  return (double)(sum_err2) / count;
}

inline float eval_psnr(gdt::vec3i dims, float data_range, accessor_t acc_pred, accessor_t acc_targ)
{
  const float mse = eval_mse(dims, acc_pred, acc_targ);
  return (float)(10. * std::log10(data_range * data_range / mse));
}

template<typename T>
static double eval_psnr_full(gdt::vec3i dims, float data_range, const T* pred, const T* targ) 
{
  const float mse = eval_mse_full(dims, pred, targ);
  return (float)(10. * std::log10(data_range * data_range / mse));
}

inline __device__ double clip(double in, double lo = 0.0, double hi = 1.0) {
  return std::max(lo, std::min(hi, in));
}

template<int N>
inline __device__ double quantize(double in) {
  return clip(
    std::min(N-1, (int)(in * N))  /  double(N-1)
  );
}

template<typename T, int WIN_SIZE, int QUANTIZE=0>
__global__ void 
kernel_ssim(const uint32_t dimx, const uint32_t dimy, const uint32_t dimz,
            const T* __restrict__ _fx, const T* __restrict__ _fy, 
            const gdt::vec3i gdims, const double data_min, const double data_max, 
            const double cov_norm, const double K1, const double K2,
            float* __restrict__ out)
{
  const int32_t x = blockIdx.x * blockDim.x + threadIdx.x; if (x >= dimx) return;
  const int32_t y = blockIdx.y * blockDim.y + threadIdx.y; if (y >= dimy) return;
  const int32_t z = blockIdx.z * blockDim.z + threadIdx.z; if (z >= dimz) return;

  double ux  = 0.0;
  double uy  = 0.0;
  double uxx = 0.0;
  double uyy = 0.0;
  double uxy = 0.0;

  const double R = data_max - data_min;
  const double rR = 1.0 / R;

  for (int kz = 0; kz < WIN_SIZE; ++kz) {
  for (int ky = 0; ky < WIN_SIZE; ++ky) {
  for (int kx = 0; kx < WIN_SIZE; ++kx) {

    const gdt::vec3i g = gdt::vec3i(x + kx, y + ky, z + kz);
    const uint32_t gidx = g.x + g.y * gdims.x + g.z * gdims.x * gdims.y;

    double fx = clip((double(_fx[gidx]) - data_min) * rR);
    double fy = clip((double(_fy[gidx]) - data_min) * rR);
    if constexpr (QUANTIZE > 0) {
      fx = quantize<QUANTIZE>(fx);
      fy = quantize<QUANTIZE>(fy);
    }

    ux  += fx;
    uy  += fy;
    uxx += fx * fx;
    uyy += fy * fy;
    uxy += fx * fy;

  }
  }
  }

  const double w = 1.0 / (WIN_SIZE*WIN_SIZE*WIN_SIZE); // uniform filter
  ux  = clip(ux  * w);
  uy  = clip(uy  * w);
  uxx = clip(uxx * w);
  uyy = clip(uyy * w);
  uxy = clip(uxy * w);

  const double vx  = cov_norm * clip(uxx - ux * ux,  0.0, 1.0);
  const double vy  = cov_norm * clip(uyy - uy * uy,  0.0, 1.0);
  const double vxy = cov_norm * clip(uxy - ux * uy, -1.0, 1.0);

  const double C1 = (K1 * R) * (K1 * R);
  const double C2 = (K2 * R) * (K2 * R);

  const double A1 = 2 * ux * uy + C1;
  const double A2 = 2 * vxy + C2;
  const double B1 = ux * ux + uy * uy + C1;
  const double B2 = vx + vy + C2;
  const double D = B1 * B2;
  const double S = (A1 * A2) / D;

  out[x + y * dimx + z * dimx * dimy] = S;
}

template<int QUANTIZE, int WIN_SIZE>
inline float eval_ssim_core(const gdt::vec3i dims, 
  const float data_min, const float data_max, 
  accessor_t acc_pred, accessor_t acc_targ,
  // parameters //
  float K1 = 0.01f, float K2 = 0.03f,
  bool use_sample_covariance = false)
{
  const int crop = WIN_SIZE >> 1;
  const int NP = WIN_SIZE * WIN_SIZE * WIN_SIZE;

  // filter has already normalized by NP
  const float cov_norm = use_sample_covariance 
    ? (float)NP / (NP - 1) // sample covariance
    : 1.f; // population covariance to match Wang et. al. 2004

  const gdt::vec3i batch = gdt::min(gdt::vec3i(4096,16,16),dims);
  const auto batch_count = util::next_multiple<size_t>(batch.long_product(), 256);

  float* d_output = NULL;
  CUDA_CHECK(cudaMalloc(&d_output, batch_count * sizeof(float)));

  float ssim_sum = 0.f;

  for (int z = crop; z < dims.z - crop; z += batch.z) {
  for (int y = crop; y < dims.y - crop; y += batch.y) {
  for (int x = crop; x < dims.x - crop; x += batch.x) {
    const gdt::vec3i block_offset = gdt::vec3i(x,y,z);
    const gdt::vec3i block = gdt::min(batch, dims - crop - block_offset);
    const auto block_count = block.long_product();
    if (block_count == 0) continue;

    // compute grid values
    const gdt::vec3i block_grid_offset = block_offset - crop;
    const gdt::vec3i block_grid = block + WIN_SIZE - 1;

    // reference & inference
    float* d_targ = acc_targ(block_grid_offset, block_grid);
    float* d_pred = acc_pred(block_grid_offset, block_grid);

    // calculate SSIM
    util::trilinear_kernel(kernel_ssim<float, WIN_SIZE, QUANTIZE>, 0, 0, 
      block.x, block.y, block.z, 
      d_targ, d_pred, block_grid,
      data_min, data_max, 
      cov_norm, K1, K2, d_output
    );
    CUDA_CHECK(cudaStreamSynchronize(0));

    // compute total error
    ssim_sum += parallel_sum_gpu(d_output, block_count);
  }
  }
  }

  CUDA_CHECK(cudaFree(d_output));

  return ssim_sum / (dims - WIN_SIZE + 1).long_product();
}

template<typename T, int QUANTIZE, int WIN_SIZE>
inline float eval_ssim_core_full(const gdt::vec3i dims, 
  const float data_min, const float data_max, 
  const T* d_pred, const T* d_targ,
  // parameters //
  float K1 = 0.01f, float K2 = 0.03f,
  bool use_sample_covariance = false)
{
  const int crop = WIN_SIZE >> 1;
  const int NP = WIN_SIZE * WIN_SIZE * WIN_SIZE;

  // filter has already normalized by NP
  const float cov_norm = use_sample_covariance 
    ? (float)NP / (NP - 1) // sample covariance
    : 1.f; // population covariance to match Wang et. al. 2004

  gdt::vec3i bdims = dims - crop*2;

  float* d_output = NULL;
  CUDA_CHECK(cudaMalloc(&d_output, bdims.long_product()*sizeof(float)));

  // calculate SSIM
  util::trilinear_kernel(kernel_ssim<T, WIN_SIZE, QUANTIZE>, 0, 0, 
    bdims.x, bdims.y, bdims.z, d_targ, d_pred, dims,
    data_min, data_max, cov_norm, K1, K2, d_output
  );

  float ssim_sum = parallel_sum_gpu(d_output, bdims.long_product());

  CUDA_CHECK(cudaFree(d_output));

  return ssim_sum / bdims.long_product();
}

constexpr int SSIM_WIN_SIZE = 7;

inline float eval_ssim(gdt::vec3i dims, float data_min, float data_max, accessor_t acc_pred, accessor_t acc_targ)
{
  return eval_ssim_core<0, SSIM_WIN_SIZE>(dims, data_min, data_max, acc_pred, acc_targ);
}

template<typename T>
inline float eval_ssim_full(gdt::vec3i dims, float data_min, float data_max, const T* d_pred, const T* d_targ)
{
  return eval_ssim_core_full<T, 0, SSIM_WIN_SIZE>(dims, data_min, data_max, d_pred, d_targ);
}

// implementing paper: DSSIM: a structural similarity index for floating-point data
// --- https://arxiv.org/abs/2202.02616

inline float eval_dssim(gdt::vec3i dims, float data_min, float data_max, accessor_t acc_pred, accessor_t acc_targ)
{
  constexpr float K1 = 0.0001f;
  constexpr float K2 = 0.0001f;
  return eval_ssim_core<256, SSIM_WIN_SIZE>(dims, data_min, data_max, acc_pred, acc_targ, K1, K2);
}

template<typename T>
inline float eval_dssim_full(gdt::vec3i dims, float data_min, float data_max, const T* d_pred, const T* d_targ)
{
  constexpr float K1 = 0.0001f;
  constexpr float K2 = 0.0001f;
  return eval_ssim_core_full<T, 256, SSIM_WIN_SIZE>(dims, data_min, data_max, d_pred, d_targ, K1, K2);
}
