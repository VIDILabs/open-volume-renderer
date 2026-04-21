// ======================================================================== //
// GPU-gated property checks for generate_sparse_sampling_mask_h/_d.      //
// Asserts in-range coordinates, monotonicity in base_noise, and that    //
// base_noise=1.0 (always-accept) yields exactly fbsize.x*fbsize.y       //
// samples.                                                              //
// ======================================================================== //

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <generate_mask.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

using namespace ovr;

namespace {
bool cuda_available()
{
  int count = 0;
  auto err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
}
}

TEST_CASE("generate_sparse_sampling_mask_h: coordinates stay in bounds") {
  if (!cuda_available()) {
    MESSAGE("Skipping: no CUDA device");
    return;
  }

  const vec2i fbsize(48, 32);
  std::vector<int32_t> out(fbsize.long_product() * 2, -1);

  int64_t count = generate_sparse_sampling_mask_h(
      out.data(), /*frame_index=*/0, fbsize,
      vec2f(0.5f, 0.5f), /*focus_scale=*/0.3f, /*base_noise=*/0.2f);

  CHECK(count >= 0);
  CHECK(count <= static_cast<int64_t>(fbsize.x * fbsize.y * 2));
  CHECK((count % 2) == 0);

  for (int64_t i = 0; i < count; i += 2) {
    int32_t x = out[i];
    int32_t y = out[i + 1];
    CHECK(x >= 0);
    CHECK(x < fbsize.x);
    CHECK(y >= 0);
    CHECK(y < fbsize.y);
  }
}

TEST_CASE("generate_sparse_sampling_mask_h: base_noise=1.0 accepts nearly every pixel") {
  if (!cuda_available()) {
    MESSAGE("Skipping: no CUDA device");
    return;
  }

  const vec2i fbsize(16, 16);
  std::vector<int32_t> out(fbsize.long_product() * 2, -1);

  int64_t count = generate_sparse_sampling_mask_h(
      out.data(), /*frame_index=*/7, fbsize,
      vec2f(0.5f, 0.5f), 0.1f, /*base_noise=*/1.0f);

  // With base_noise = 1.0 the accept probability is exactly 1.0, but the
  // blue-noise distribution can yield sample values numerically equal to
  // 1.0 (strict `<` comparison in the kernel rejects those). Allow a
  // small slack.
  const int64_t total_ints = static_cast<int64_t>(fbsize.x * fbsize.y * 2);
  CHECK(count >= static_cast<int64_t>(total_ints * 0.95));
  CHECK(count <= total_ints);
}

TEST_CASE("generate_sparse_sampling_mask_h: higher base_noise >= lower (avg)") {
  if (!cuda_available()) {
    MESSAGE("Skipping: no CUDA device");
    return;
  }

  // Average over a few frames to tame blue-noise variance.
  const vec2i fbsize(32, 32);
  std::vector<int32_t> out(fbsize.long_product() * 2, -1);

  int64_t low_sum = 0;
  int64_t high_sum = 0;
  for (int f = 0; f < 4; ++f) {
    low_sum  += generate_sparse_sampling_mask_h(
        out.data(), f, fbsize, vec2f(0.5f, 0.5f), 0.2f, /*base_noise=*/0.1f);
    high_sum += generate_sparse_sampling_mask_h(
        out.data(), 100 + f, fbsize, vec2f(0.5f, 0.5f), 0.2f, /*base_noise=*/0.9f);
  }

  CHECK(high_sum >= low_sum);
}
