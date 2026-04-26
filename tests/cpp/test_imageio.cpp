// ======================================================================== //
// Unit tests for ovr::save_image. rendercommon bundles stb_image via      //
// STB_IMAGE_IMPLEMENTATION so we cannot pull stb_image.h here without    //
// duplicate-symbol grief. We therefore validate by inspecting the emitted //
// file's header bytes (PNG magic) and size instead of re-decoding.       //
// ======================================================================== //

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <imageio.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace ovr;

namespace {

std::string tmp_path(const char* stem, const char* ext)
{
  auto p = std::filesystem::temp_directory_path()
         / (std::string("ovr_test_") + stem + "." + ext);
  return p.string();
}

// Reads the first `n` bytes of `path` into a vector. Returns an empty
// vector if the file can't be opened or is shorter than requested.
std::vector<uint8_t> read_prefix(const std::string& path, size_t n)
{
  std::ifstream f(path, std::ios::binary);
  std::vector<uint8_t> buf(n);
  f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n));
  buf.resize(static_cast<size_t>(f.gcount()));
  return buf;
}

bool looks_like_png(const std::vector<uint8_t>& header)
{
  static const uint8_t kMagic[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
  if (header.size() < 8) return false;
  for (int i = 0; i < 8; ++i) if (header[i] != kMagic[i]) return false;
  return true;
}

} // namespace

TEST_CASE("save_image: uint32 RGBA8 writes a valid PNG") {
  const int W = 16, H = 8;
  std::vector<uint32_t> pixels(W * H);
  for (int j = 0; j < H; ++j) {
    for (int i = 0; i < W; ++i) {
      pixels[j * W + i] = (uint32_t(255) << 24)
                        | (uint32_t(64)  << 16)
                        | (uint32_t(j*32) << 8)
                        |  uint32_t(i*16);
    }
  }

  auto out = tmp_path("rgba8", "png");
  CHECK_NOTHROW(save_image(out, pixels.data(), W, H));
  REQUIRE(std::filesystem::exists(out));
  CHECK(std::filesystem::file_size(out) > 0);
  CHECK(looks_like_png(read_prefix(out, 8)));
  std::filesystem::remove(out);
}

TEST_CASE("save_image: vec4f RGBA32F writes a valid PNG") {
  const int W = 8, H = 4;
  std::vector<vec4f> pixels(W * H, vec4f(0.25f, 0.5f, 0.75f, 1.f));

  auto out = tmp_path("rgba32f", "png");
  CHECK_NOTHROW(save_image(out, pixels.data(), W, H));
  REQUIRE(std::filesystem::exists(out));
  CHECK(looks_like_png(read_prefix(out, 8)));
  std::filesystem::remove(out);
}

TEST_CASE("save_image: vec3f RGB32F writes a valid PNG") {
  const int W = 4, H = 4;
  std::vector<vec3f> pixels(W * H, vec3f(1.f, 0.f, 0.f));

  auto out = tmp_path("rgb32f", "png");
  CHECK_NOTHROW(save_image(out, pixels.data(), W, H));
  REQUIRE(std::filesystem::exists(out));
  CHECK(looks_like_png(read_prefix(out, 8)));
  std::filesystem::remove(out);
}
