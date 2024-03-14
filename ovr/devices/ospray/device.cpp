#include "device.h"
#include "device_impl.h"

#include <chrono>

#ifdef OVR_BUILD_OPENGL
#include <imgui.h>
#endif

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

void 
DeviceOSPRay::ui() 
{
#ifdef OVR_BUILD_OPENGL
  static struct {
    vec2f focus{ 0.5f, 0.5f };
    float focus_scale{ 0.06f };
    float base_noise{ 0.07f };
    bool sparse_sampling{ false };
  } config;
  if (ImGui::Begin("OSPRay Panel", NULL)) {
    bool updated = false;
    updated |= ImGui::SliderFloat("Focus Center X", &config.focus.x, 0.f, 1.f, "%.3f");
    updated |= ImGui::SliderFloat("Focus Center Y", &config.focus.y, 0.f, 1.f, "%.3f");
    updated |= ImGui::SliderFloat("Focus Scale", &config.focus_scale, 0.01f, 1.f, "%.3f");
    updated |= ImGui::SliderFloat("Base Noise", &config.base_noise, 0.01f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    if (updated) {
      set_focus(config.focus, config.focus_scale, config.base_noise);
    }
    if (ImGui::Checkbox("Sparse Sampling", &config.sparse_sampling)) {
      config.sparse_sampling = config.sparse_sampling;
      set_sparse_sampling(config.sparse_sampling);
    }
  }
  ImGui::End();
#endif
}

} // namespace ovr::ospray
