#include "device.h"
#include "device_impl.h"

#include <chrono>


namespace ovr::ospray {

DeviceOSPRay::~DeviceOSPRay()
{
  pimpl.reset();
}

DeviceOSPRay::DeviceOSPRay() : MainRenderer(), pimpl(new Impl()) {}

void
DeviceOSPRay::init(int argc, const char** argv)
{
  pimpl->init(argc, argv, this);
  pimpl->commit();
}

void
DeviceOSPRay::swap()
{
  pimpl->swap();
}

void
DeviceOSPRay::commit()
{
  pimpl->commit();
}

void
DeviceOSPRay::render()
{
  auto start = std::chrono::high_resolution_clock::now();
  pimpl->render();

  // CUDA_CHECK(cudaDeviceSynchronize()); // TODO: Confirm the necessary of CUDA Check

  auto end = std::chrono::high_resolution_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  render_time += diff.count();
}

void
DeviceOSPRay::mapframe(FrameBufferData* fb)
{
  return pimpl->mapframe(fb);
}

} // namespace ovr::ospray
