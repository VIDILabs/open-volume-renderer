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
//. ======================================================================== //
//. Copyright 2018-2019 Ingo Wald                                            //
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

#include "cuda_common.h"
#include "volume.h"

#if defined(__cplusplus)

#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#endif // defined(__cplusplus)

namespace ovr {

// ------------------------------------------------------------------
//
// Shared Definitions
//
// ------------------------------------------------------------------

// for this simple example, we have a single ray type
enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

struct RayPayload { // per-ray payload data
  float alpha = 0.f;
  vec3f color = 0.f;
  vec3f Ng = 0.f;
  vec2f flow = 0.f;

  // other information
  int32_t type = -1; // -1 means primary ray
  float tmax = 0.f;
};

struct LaunchParams { // shared global data
  struct DeviceFrameBuffer {
    uint32_t* rgba;
    vec2i size;
  } frame;

  struct DeviceCamera {
    vec3f position;
    vec3f direction;
    vec3f horizontal;
    vec3f vertical;
  } camera, last_camera;

  OptixTraversableHandle traversable{};

  vec3f lightPos{ -907.108f, 2205.875f, -400.0267f };
};

// ------------------------------------------------------------------
//
// Exclusive Device Functions
//
// ------------------------------------------------------------------
#ifdef __CUDA_ARCH__
#endif

// ------------------------------------------------------------------
//
// Exclusive Host Functions
//
// ------------------------------------------------------------------
#if defined(__cplusplus)
namespace host {

template<typename T>
struct FrameBufferTemplate {
private:
  CUDABuffer render_buffer;
  std::vector<T> rb_fg, rb_bg;

  size_t fb_pixel_count{ 0 };
  vec2i fb_size;

  std::mutex m;

public:
  FrameBufferTemplate() {}

  ~FrameBufferTemplate()
  {
    if (render_buffer.d_pointer())
      render_buffer.free();
  }

  void resize(vec2i s)
  {
    fb_size = s;
    fb_pixel_count = (size_t)fb_size.x * fb_size.y;
    rb_fg.resize(fb_pixel_count);
    rb_bg.resize(fb_pixel_count);
    render_buffer.resize(fb_pixel_count * sizeof(T));
  }

  void safe_swap()
  {
    std::unique_lock<std::mutex> lk(m);
    rb_fg.swap(rb_bg);
  }

  void background_download_async(CUstream stream)
  {
    render_buffer.download_async(rb_bg.data(), fb_pixel_count, stream);
  }

  // void background_download()
  // {
  //   render_buffer.download(rb_bg.data(), fb_pixel_count);
  // }

  T* d_pointer() const
  {
    return (T*)render_buffer.d_pointer();
  }

  void deepcopy(void* dst)
  {
    std::unique_lock<std::mutex> lk(m);
    std::memcpy(dst, rb_fg.data(), fb_pixel_count * sizeof(uint32_t));
  }

  bool empty() const
  {
    return fb_pixel_count == 0;
  }

  vec2i size()
  {
    return fb_size;
  }
};

using FrameBuffer = FrameBufferTemplate<uint32_t>;

// ------------------------------------------------------------------
// I/O helper functions
// ------------------------------------------------------------------

static void
context_log_cb(unsigned int level, const char* tag, const char* message, void*)
{
  fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

static void
general_log_cb(const char* log, size_t sizeof_log)
{
  if (sizeof_log > 1)
    PRINT(log);
}

extern "C" char embedded_ptx_code[];

struct Camera {
public:
  /*! camera position - *from* where we are looking */
  vec3f from;
  /*! which point we are looking *at* */
  vec3f at;
  /*! general up-vector */
  vec3f up;
};

/*! a sample OptiX-7 renderer that demonstrates how to set up
    context, module, programs, pipeline, SBT, etc, and perform a
    valid launch that renders some pixel (using a simple test
    pattern, in this case */
struct MainRenderer {
  // ------------------------------------------------------------------
  // publicly accessible interface
  // ------------------------------------------------------------------
public:
  /*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
  void init();

  /*! render one frame */
  void render();

  /*! resize frame buffer to given resolution */
  void resize(const vec2i& newSize);

  /*! download the rendered color buffer */
  void download_pixels(uint32_t h_pixels[]);

  /*! set camera to render with */
  void set_camera(const Camera& camera);
  void set_camera(vec3f from, vec3f at, vec3f up);

  void set_transfer_function(const std::vector<float>& c, const std::vector<float>& o, const vec2f& r)
  {
    for (auto& v : volumes) {
      v.set_transfer_function(c, o, r);
    }
  }

  void set_scene(int ac, char** av);

  std::vector<float> tfn_colors;
  std::vector<float> tfn_alphas;

  // ------------------------------------------------------------------
  // internal helper functions
  // ------------------------------------------------------------------
protected:
  /*! helper function that initializes optix and checks for errors */
  /*static*/ void initOptix();

  /*! creates and configures a optix device context (in this simple
    example, only for the primary GPU device) */
  void createContext();

  /*! creates the module that contains all the programs we are going
    to use. in this simple example, we use a single module from a
    single .cu file, using a single embedded ptx string */
  void createModule();

  /*! does all setup for the raygen program(s) we are going to use */
  void createRaygenPrograms();

  /*! does all setup for the miss program(s) we are going to use */
  void createMissPrograms();

  /*! does all setup for the hitgroup program(s) we are going to use */
  void createHitgroupPrograms();

  /*! assembles the full pipeline of all programs */
  void createPipeline();

  /*! constructs the shader binding table */
  void buildSBT();

  /*! build the instance acceleration structure */
  OptixTraversableHandle buildAccel_instances(OptixDeviceContext optixContext);

protected:
  /*! @{ CUDA device context and stream that optix pipeline will run
      on, as well as device properties for this device */
  CUcontext context{};
  CUstream stream{};
  cudaDeviceProp device_props{};
  /*! @} */

  // the optix context that our pipeline will run in.
  OptixDeviceContext optix_context{};

  /*! @{ the pipeline we're building */
  OptixPipeline pipeline{};
  OptixPipelineCompileOptions pipelineCompileOptions{};
  OptixPipelineLinkOptions pipelineLinkOptions{};
  /*! @} */

  /*! @{ the module that contains out device programs */
  OptixModule module{};
  OptixModuleCompileOptions moduleCompileOptions{};
  /* @} */

  /*! vector of all our program(group)s, and the SBT built around
      them */
  OptixProgramGroup raygenPG;
  CUDABuffer raygenRecordBuffer;
  std::vector<OptixProgramGroup> missPGs;
  CUDABuffer missRecordsBuffer;
  std::vector<OptixProgramGroup> hitgroupPGs;
  CUDABuffer hitgroupRecordsBuffer;
  OptixShaderBindingTable sbt = {};

  /*! @{ our launch parameters, on the host, and the buffer to store
      them on the device */
  LaunchParams launchParams;
  CUDABuffer launchParamsBuffer;
  /*! @} */

  /*! the rendered image */
  FrameBuffer framebuffer;

  /*! the camera we are to render with. */
  Camera latest_camera;
  std::mutex camera_mutex;

  /*! all volumes share the same transfer function currently */
  std::vector<StructuredRegularVolume> volumes;

  /*! the ISA handlers */
  std::vector<OptixInstance> instances;
  /*! one buffer for all ISAs on GPU */
  CUDABuffer instancesBuffer;
  // buffer that keeps the (final, compacted) accel structure
  CUDABuffer instancesAsBuffer;
};

} // namespace host
#endif

} // namespace ovr
