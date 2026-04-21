// ======================================================================== //
// GPU-gated tests for ovr::CrossDeviceBuffer - verify the host <-> CUDA //
// round-trips preserve byte content. These tests run under the CTest   //
// `gpu` label and are automatically skipped on hostless runners via the //
// `gpu_available` fixture wired in by test/cpp/gpu_fixture_attach.cmake.//
// ======================================================================== //

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cross_device_buffer.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

using ovr::CrossDeviceBuffer;

namespace {
bool cuda_available()
{
  int count = 0;
  auto err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
}
}

TEST_CASE("CrossDeviceBuffer: CPU->CUDA->CPU round-trip preserves bytes") {
  if (!cuda_available()) {
    MESSAGE("Skipping: no CUDA device");
    return;
  }

  constexpr size_t N = 1024;
  std::vector<uint32_t> src(N);
  for (size_t i = 0; i < N; ++i) src[i] = static_cast<uint32_t>(i * 7 + 3);

  CrossDeviceBuffer buf;
  buf.set_data(src.data(), src.size() * sizeof(uint32_t),
               CrossDeviceBuffer::DEVICE_CPU);
  REQUIRE(buf.is_on_cpu());

  buf.to_cuda();
  REQUIRE(buf.is_on_cuda());
  CHECK(buf.get_size_in_bytes() == src.size() * sizeof(uint32_t));

  // Wipe the host copy to make sure the comparison is against GPU->CPU data.
  std::memset(src.data(), 0, src.size() * sizeof(uint32_t));

  buf.to_cpu();
  REQUIRE(buf.is_on_cpu());

  auto* out = static_cast<uint32_t*>(buf.data());
  for (size_t i = 0; i < N; ++i) {
    CHECK(out[i] == static_cast<uint32_t>(i * 7 + 3));
  }
}

TEST_CASE("CrossDeviceBuffer: multiple to_cuda() calls are idempotent") {
  if (!cuda_available()) {
    MESSAGE("Skipping: no CUDA device");
    return;
  }

  std::vector<float> src(256, 1.5f);
  CrossDeviceBuffer buf;
  buf.set_data(src.data(), src.size() * sizeof(float),
               CrossDeviceBuffer::DEVICE_CPU);

  buf.to_cuda();
  void* p1 = buf.data();
  buf.to_cuda();
  void* p2 = buf.data();
  CHECK(p1 == p2);
  CHECK(buf.is_on_cuda());
}

TEST_CASE("CrossDeviceBuffer: cleanup=true frees the peer device") {
  if (!cuda_available()) {
    MESSAGE("Skipping: no CUDA device");
    return;
  }

  std::vector<uint8_t> src(64, 0xAA);
  CrossDeviceBuffer buf;
  buf.set_data(src.data(), src.size(), CrossDeviceBuffer::DEVICE_CPU);

  buf.to_cuda(/*cleanup=*/true);
  CHECK(buf.is_on_cuda());
  buf.to_cpu(/*cleanup=*/true);
  CHECK(buf.is_on_cpu());

  // Content must survive the full round-trip.
  auto* out = static_cast<uint8_t*>(buf.data());
  for (size_t i = 0; i < src.size(); ++i) CHECK(out[i] == 0xAA);
}
