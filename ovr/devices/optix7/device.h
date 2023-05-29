//. ======================================================================== //
//. Copyright 2019-2020 Qi Wu                                                //
//.                                                                          //
//. Licensed under the Apache License, Version 2.0 (the "License");          //
//. you may not use this file except in compliance with the License.         //
//. You may obtain a copy of the License at                                  //
//.                                                                          //
//.     http://www.apache.org/licenses/LICENSE-2.0                           //
//.                                                                          //
//. Unless required by applicable law or agreed to in writing, software      //
//. distributed under the License is distributed on an "AS IS" BASIS,        //
//. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
//. See the License for the specific language governing permissions and      //
//. limitations under the License.                                           //
//. ======================================================================== //

#pragma once
#ifndef OVR_OPTIX7_DEVICE_H
#define OVR_OPTIX7_DEVICE_H

#include "ovr/renderer.h"

#include <memory>

namespace ovr::optix7 {

/*! a sample OptiX-7 renderer that demonstrates how to set up
    context, module, programs, pipeline, SBT, etc, and perform a
    valid launch that renders some pixel (using a simple test
    pattern, in this case */
class DeviceOptix7 : public MainRenderer {
public:
  ~DeviceOptix7() override;
  DeviceOptix7();
  DeviceOptix7(const DeviceOptix7& other) = delete;
  DeviceOptix7(DeviceOptix7&& other) = delete;
  DeviceOptix7& operator=(const DeviceOptix7& other) = delete;
  DeviceOptix7& operator=(DeviceOptix7&& other) = delete;

  /*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
  void init(int argc, const char** argv) override;

  /*! render one frame */
  void swap() override;
  void commit() override;
  void render() override;
  void mapframe(FrameBufferData* fb) override;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl; // pointer to the internal implementation
};

} // namespace ovr::optix7
#endif // OVR_OPTIX7_DEVICE_H
