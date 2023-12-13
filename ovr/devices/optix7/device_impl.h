#pragma once
#ifndef OVR_OPTIX7_DEVICE_IMPL_H
#define OVR_OPTIX7_DEVICE_IMPL_H

#include "device.h"

#include "params.h"

#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ovr::optix7 {

struct DeviceOptix7::Impl {
  DeviceOptix7* parent{ nullptr };

public:
  void init(int argc, const char** argv, DeviceOptix7* parent);
  void swap();
  void commit();
  void render();
  void mapframe(FrameBufferData*);

private:
  /*! helper function that initializes optix and checks for errors */
  /*static*/ void initOptix();

  /*! creates and configures a optix device context (in this simple
    example, only for the primary GPU device) */
  void createContext();

  /*! creates the module that contains all the programs we are going
    to use. in this simple example, we use a single module from a
    single .cu file, using a single embedded ptx string */
  void createModule();

  /*! does all setup for the program(s) we are going to use */
  void createPrograms();

  /*! assembles the full pipeline of all programs */
  void createPipeline();

  /*! constructs the shader binding table */
  void buildSBT();

  /*! constructs the scene data */
  void buildScene(const Scene& scene);

  /*! build the instance acceleration structure */
  OptixTraversableHandle buildTLAS();

  /*! constructs the sparse sampling pattern */
  vec3i createSparseSamples();

  // helper functions

  OptixProgramGroup addProgram(OptixDeviceContext optix_context, OptixProgramGroupDesc desc);

  OptixProgramGroupDesc createRaygenDesc(OptixModule module, const std::string& func) 
  {
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc.raygen.module = module;
    desc.raygen.entryFunctionName = func.c_str();
    return desc;
  }

  OptixProgramGroupDesc createMissDesc(OptixModule module, const std::string& func) {
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc.miss.module = module;
    desc.miss.entryFunctionName = func.c_str();
    return desc;
  }

  OptixProgramGroupDesc createHitgroupDesc(OptixModule module, const std::string& funcCH, const std::string& funcAH, const std::string& funcIS) {
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    if (!funcCH.empty()) {
      desc.hitgroup.moduleCH = module;
      desc.hitgroup.entryFunctionNameCH = funcCH.c_str();
    }
    if (!funcAH.empty()) {
      desc.hitgroup.moduleAH = module;
      desc.hitgroup.entryFunctionNameAH = funcAH.c_str();
    }
    if (!funcIS.empty()) {
      desc.hitgroup.moduleIS = module;
      desc.hitgroup.entryFunctionNameIS = funcIS.c_str();
    }
    return desc;
  }

private:
  /*! @{ CUDA device context and stream that optix pipeline will run on, as well as device properties for this device */
  cudaDeviceProp cuda_device_props{};
  CUcontext cuda_context{};
  OptixDeviceContext optix_context{}; /* the optix context that our pipeline will run in. */
  /*! @} */

  /*! @{ the pipeline we're building */
  struct Pipeline {
    OptixPipeline handle{};
    OptixPipelineCompileOptions compile_opts{};
    OptixPipelineLinkOptions link_opts{};
  } pipeline;
  /*! @} */

  /*! @{ the module that contains out device programs */
  std::vector<OptixProgramGroup> programs;

  struct Module {
    OptixModuleCompileOptions compile_opts{};
    OptixModule handle{};

    void createModule(OptixDeviceContext, const Pipeline&, const std::string& ptx_code);
  };
  /*! @} */

  /*! @{ vector of all our program(group)s, and the SBT built around them */
  struct StbGraph {
    std::vector<AabbGeometry*> aabbs;

    /*! build the instance acceleration structure */
    OptixTraversableHandle buildTLAS();
  };
  struct StbGroup {
    OptixProgramGroup raygen;
    CUDABuffer raygen_buffer;
    std::vector<OptixProgramGroup> misses;
    CUDABuffer miss_buffer;
    std::vector<OptixProgramGroup> hitgroups;
    CUDABuffer hitgroup_buffer;
    OptixShaderBindingTable sbt = {};
    void buildSBT(StbGraph);
  };

  OptixShaderBindingTable* stb_current{ nullptr };
  /*! @} */

  Module module_raymarching, module_pathtracing;
  StbGroup sbt_raymarching_main, sbt_pathtracing_main;

  /*! @{ our launch parameters, on the host, and the buffer to store them on the device */
  LaunchParams params;
  CUDABuffer params_buffer;
  /*! @} */

  /*! the rendered image */
  FrameBuffer framebuffer;
  cudaStream_t framebuffer_stream{};
  bool framebuffer_size_updated{ false };
  bool framebuffer_reset{ false };
  CUDABuffer framebuffer_accum_rgba;
  CUDABuffer framebuffer_accum_grad;

  /*! all volumes share the same transfer function currently */
  std::vector<StructuredRegularVolume> volumes;
  bool volumes_changed{ true };

  /*! the ISA handlers */
  std::vector<OptixInstance> instances;
  /*! one buffer for all ISAs on GPU */
  CUDABuffer instances_buffer;
  /*! buffer that keeps the (final, compacted) accel structure */
  CUDABuffer instances_accel_struct_buffer;

  /*! for mask sampling */
#ifdef OVR_OPTIX7_MASKING_VIA_DIRECT_SAMPLING
  CUDABuffer dist_uniform;
  CUDABuffer dist_logistic;
#endif

#ifdef OVR_OPTIX7_MASKING_VIA_STREAM_COMPACTION
  CUDABuffer xs_and_ys;
#endif
};

void
generate_logistic_dist(CUDABuffer& buffer, float center, float stddev);

void
generate_uniform_dist(CUDABuffer& buffer, float lower, float upper);

void
generate_blue_dist(CUDABuffer& buffer, vec2i size, int frame_index);

int64_t
generate_and_compact_coordinates(int32_t* d_allocated_output, const CUDABuffer& dist, vec2i size, int factor, vec2f mean, float sigma, float base_noise);

} // namespace ovr::optix7

#endif // OVR_OPTIX7_DEVICE_IMPL_H
