#pragma once
#ifndef OVR_OSPRAY_DEVICE_H
#define OVR_OSPRAY_DEVICE_H

#include "ovr/renderer.h"
#include <memory>

namespace ovr::ospray {

class DeviceOSPRay : public MainRenderer {
public:
  ~DeviceOSPRay() override;
  DeviceOSPRay();
  DeviceOSPRay(const DeviceOSPRay& other) = delete;
  DeviceOSPRay(DeviceOSPRay&& other) = delete;
  DeviceOSPRay& operator=(const DeviceOSPRay& other) = delete;
  DeviceOSPRay& operator=(DeviceOSPRay&& other) = delete;

  /*! constructor - performs all setup, including initializing ospray, creates scene graph, etc. */
  void init(int argc, const char** argv) override;

  /*! render one frame */
  void swap() override;
  void commit() override;
  void render() override;
  void mapframe(FrameBufferData* fb) override;
  void ui() override;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl; // pointer to the internal implementation
};

} // namespace ovr::ospray

#endif // OVR_OSPRAY_DEVICE_H
