#include "device.h"
#include "device_impl.h"

#include <chrono>

namespace ovr::optix7 {

DeviceOptix7::~DeviceOptix7()
{
  pimpl.reset();
}

DeviceOptix7::DeviceOptix7() : MainRenderer(), pimpl(new Impl()) {}

void
DeviceOptix7::init(int argc, const char** argv)
{
  pimpl->init(argc, argv, this);
  pimpl->commit();
}

void
DeviceOptix7::swap()
{
  pimpl->swap();
}

void
DeviceOptix7::commit()
{
  pimpl->commit();
}

void
DeviceOptix7::render()
{
  auto start = std::chrono::high_resolution_clock::now();
  pimpl->render();
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  render_time += diff.count();
}

void
DeviceOptix7::mapframe(FrameBufferData* fb)
{
  return pimpl->mapframe(fb);
}

} // namespace ovr::optix7
