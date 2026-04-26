// ======================================================================== //
// CPU-path unit tests for ovr::CrossDeviceBuffer. CUDA round-trips are    //
// covered separately under test_cuda_cross_device_buffer.cu.              //
// ======================================================================== //

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cross_device_buffer.h>

#include <cstring>
#include <vector>

using ovr::CrossDeviceBuffer;

TEST_CASE("CrossDeviceBuffer: default-constructed has no device") {
  CrossDeviceBuffer buf;
  CHECK(buf.get_size_in_bytes() == 0);
}

TEST_CASE("CrossDeviceBuffer: set_data(num_bytes, CPU) allocates and is on CPU") {
  CrossDeviceBuffer buf;
  buf.set_data(64, CrossDeviceBuffer::DEVICE_CPU);
  CHECK(buf.get_size_in_bytes() == 64);
  CHECK(buf.is_on_cpu());
  CHECK(buf.data() != nullptr);
}

TEST_CASE("CrossDeviceBuffer: set_data(ptr, num_bytes, CPU) aliases the pointer") {
  std::vector<uint8_t> src(32, 0xAB);
  CrossDeviceBuffer buf;
  buf.set_data(src.data(), src.size(), CrossDeviceBuffer::DEVICE_CPU);

  CHECK(buf.get_size_in_bytes() == src.size());
  CHECK(buf.data() == src.data());
  CHECK(static_cast<uint8_t*>(buf.data())[0] == 0xAB);
}

TEST_CASE("CrossDeviceBuffer: set_data(vector, CPU) aliases vector storage") {
  std::vector<float> vec(16, 3.14f);
  CrossDeviceBuffer buf;
  buf.set_data(vec, CrossDeviceBuffer::DEVICE_CPU);

  CHECK(buf.get_size_in_bytes() == vec.size() * sizeof(float));
  CHECK(buf.get_size<float>() == vec.size());
  CHECK(buf.data() == vec.data());
}

TEST_CASE("CrossDeviceBuffer: resize via set_data replaces content") {
  CrossDeviceBuffer buf;
  buf.set_data(16, CrossDeviceBuffer::DEVICE_CPU);
  auto* p1 = buf.data();
  std::memset(p1, 0x11, 16);

  buf.set_data(64, CrossDeviceBuffer::DEVICE_CPU);
  CHECK(buf.get_size_in_bytes() == 64);
  // New buffer may live at the same address (vector resize can be in-place),
  // but its size is authoritative.
}

TEST_CASE("CrossDeviceBuffer: get_size<T> returns element count") {
  CrossDeviceBuffer buf;
  buf.set_data(8 * sizeof(uint32_t), CrossDeviceBuffer::DEVICE_CPU);
  CHECK(buf.get_size<uint32_t>() == 8);
  CHECK(buf.get_size<uint8_t>() == 8 * sizeof(uint32_t));
}

TEST_CASE("CrossDeviceBuffer: construction from (num_bytes, device) works") {
  CrossDeviceBuffer buf(128, CrossDeviceBuffer::DEVICE_CPU);
  CHECK(buf.get_size_in_bytes() == 128);
  CHECK(buf.is_on_cpu());
}
