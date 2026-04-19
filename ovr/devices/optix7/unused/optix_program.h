#pragma once

#include "../instantvnr_types.h"

#include <string>
#include <vector>
#include <map>

namespace vnr {

// ------------------------------------------------------------------
// Object Definitions
// ------------------------------------------------------------------

enum ObjectType {
  VOLUME_STRUCTURED_REGULAR,
};

#ifndef __NVCC__
static std::string
object_type_string(ObjectType t)
{
  switch (t) {
  case VOLUME_STRUCTURED_REGULAR: return "VOLUME_STRUCTURED_REGULAR";
  default: throw std::runtime_error("unknown object type");
  }
}
#endif

enum {
  VISIBILITY_MASK_GEOMETRY = 0x1,
  VISIBILITY_MASK_VOLUME = 0x2,
};


struct OptixProgram
{
public:
  struct HitGroupShaders
  {
    std::string shader_CH;
    std::string shader_AH;
    std::string shader_IS;
  };

  struct ObjectGroup
  {
    ObjectType type;
    std::vector<HitGroupShaders> hitgroup;
  };

  struct InstanceHandler
  {
    ObjectType type;  // object type
    uint32_t idx = 0; // object index
    OptixInstance handler;
  };

  OptixProgram(std::string ptx_code, uint32_t num_ray_types) : ptx_code(ptx_code), num_ray_types(num_ray_types) {}

  virtual ~OptixProgram()
  {
    program.raygen_buffer.free(0);
    program.miss_buffer.free(0);
    program.hitgroup_buffer.free(0);
  }

  void init(OptixDeviceContext, std::map<ObjectType, std::vector<void*>> records, std::vector<std::vector<OptixProgram::InstanceHandler>> blas);

protected:
  /*! creates the module that contains all the programs we are going
    to use. in this simple example, we use a single module from a
    single .cu file, using a single embedded ptx string */
  void createModule(OptixDeviceContext);

  /*! does all setup for the raygen program(s) we are going to use */
  void createRaygenPrograms(OptixDeviceContext);

  /*! does all setup for the miss program(s) we are going to use */
  void createMissPrograms(OptixDeviceContext);

  /*! does all setup for the hitgroup program(s) we are going to use */
  void createHitgroupPrograms(OptixDeviceContext);

  /*! assembles the full pipeline of all programs */
  void createPipeline(OptixDeviceContext);

  /*! constructs the shader binding table */
  void createSBT(OptixDeviceContext, const std::map<ObjectType, std::vector<void*>>&);

  /*! build the top level acceleration structure */
  void createTLAS(OptixDeviceContext, const std::vector<std::vector<OptixProgram::InstanceHandler>>&);

protected:
  /*! @{ the pipeline we're building */
  struct
  {
    OptixPipeline handle{};
    OptixPipelineCompileOptions compile_opts{};
    OptixPipelineLinkOptions link_opts{};
  } pipeline;
  /*! @} */

  /*! @{ the module that contains out device programs */
  struct
  {
    OptixModule handle{};
    OptixModuleCompileOptions compile_opts{};
  } module;
  /*! @} */

  /*! @{ vector of all our program(group)s, and the SBT built around them */
  struct
  {
    std::vector<OptixProgramGroup> raygens;
    CUDABuffer raygen_buffer;
    std::vector<OptixProgramGroup> misses;
    CUDABuffer miss_buffer;
    std::vector<OptixProgramGroup> hitgroups;
    CUDABuffer hitgroup_buffer;
  } program;

  OptixShaderBindingTable sbt = {};
  /*! @} */

  std::string shader_raygen;
  std::vector<std::string> shader_misses;
  std::vector<ObjectGroup> shader_objects;

  std::map<ObjectType, std::vector<uint32_t>> sbt_offset_table;

  struct IasData
  {
    std::vector<OptixInstance> instances; /*! the ISA handlers */
    CUDABuffer instances_buffer;          /*! one buffer for all ISAs on GPU */
    CUDABuffer as_buffer;                 /*! buffer that keeps the (final, compacted) accel structure */
    OptixTraversableHandle traversable;   // <- the output

    ~IasData()
    {
      instances_buffer.free(0);
      as_buffer.free(0);
    }
  };
  std::shared_ptr<IasData[]> ias;

  const std::string ptx_code;
  const uint32_t num_ray_types;
};

} // namespace vnr
