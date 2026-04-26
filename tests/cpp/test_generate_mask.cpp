// ======================================================================== //
// CPU-only checks for generate_sparse_sampling_mask_h.                    //
// When CUDA is disabled the function throws; that's the only CPU-safe    //
// behaviour we can assert here. The "happy path" is covered by          //
// test_cuda_generate_mask.cu, which runs only on machines with a GPU.   //
// ======================================================================== //

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <generate_mask.h>

#include <cstdint>
#include <vector>

using namespace ovr;

#ifndef OVR_BUILD_CUDA_DEVICES
TEST_CASE("generate_sparse_sampling_mask_h: throws on non-CUDA build") {
  const vec2i fbsize(32, 24);
  std::vector<int32_t> out(fbsize.long_product() * 2, -1);
  CHECK_THROWS_AS(
      generate_sparse_sampling_mask_h(out.data(), 0, fbsize,
                                      vec2f(0.5f, 0.5f), 0.25f, 0.1f),
      std::runtime_error);
}
#else
// The CUDA-enabled path is exercised by test_cuda_generate_mask.cu so
// that it can be tagged with the `gpu` CTest label and auto-skipped on
// hostless runners. Leave a smoke compile-check here so this TU is not
// empty, which would otherwise turn doctest into a no-op runner.
TEST_CASE("generate_mask header compiles in CUDA build") {
  const vec2i fbsize(8, 8);
  CHECK(fbsize.long_product() == 64);
}
#endif
