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

// our own classes, partly shared between host and device
#include "renderer.h"

namespace ovr {
namespace host {

/*! constructor - performs all setup, including initializing
  optix, creates module, pipeline, programs, SBT, etc. */
void
MainRenderer::init()
{
  initOptix();

  std::cout << "#osc: creating optix context ..." << std::endl;

  createContext();

  std::cout << "#osc: setting up module ..." << std::endl;

  createModule();

  std::cout << "#osc: creating raygen programs ..." << std::endl;

  createRaygenPrograms();

  std::cout << "#osc: creating miss programs ..." << std::endl;

  createMissPrograms();

  std::cout << "#osc: creating hitgroup programs ..." << std::endl;

  createHitgroupPrograms();

  launchParams.traversable = buildAccel_instances(optix_context);

  std::cout << "#osc: setting up optix pipeline ..." << std::endl;

  createPipeline();

  std::cout << "#osc: building SBT ..." << std::endl;

  buildSBT();

  launchParamsBuffer.alloc(sizeof(launchParams));

  std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;
  std::cout << GDT_TERMINAL_GREEN;
  std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
  std::cout << GDT_TERMINAL_DEFAULT;
}

/*! helper function that initializes optix and checks for errors */
void
MainRenderer::initOptix()
{
  std::cout << "#osc: initializing OptiX..." << std::endl;

  // -------------------------------------------------------
  // check for available optix7 capable devices
  // -------------------------------------------------------
  cudaFree(0);
  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0)
    throw std::runtime_error("#osc: no CUDA capable devices found!");
  std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

  // -------------------------------------------------------
  // initialize optix
  // -------------------------------------------------------
  OPTIX_CHECK(optixInit());
  std::cout << "#osc: successfully initialized OptiX" << std::endl;
}

/*! creates and configures a optix device context (in this simple
  example, only for the primary GPU device) */
void
MainRenderer::createContext()
{
  // for this sample, do everything on one device
  const int device_id = 0;
  CUDA_CHECK(SetDevice(device_id));
  CUDA_CHECK(StreamCreate(&stream));

  cudaGetDeviceProperties(&device_props, device_id);
  std::cout << "#osc: running on device: " << device_props.name << std::endl;

  CUresult cuRes = cuCtxGetCurrent(&context);
  if (cuRes != CUDA_SUCCESS)
    fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

  OPTIX_CHECK(optixDeviceContextCreate(context, 0, &optix_context));
  OPTIX_CHECK(optixDeviceContextSetLogCallback(optix_context, context_log_cb, nullptr, 4));
}

/*! creates the module that contains all the programs we are going
    to use. in this simple example, we use a single module from a
    single .cu file, using a single embedded ptx string */
void
MainRenderer::createModule()
{
  moduleCompileOptions.maxRegisterCount = 100;

  moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
  // moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_2;
  // moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_1;
  // moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;

  moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
  // moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
  // moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
  // moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  pipelineCompileOptions = {};
  pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
  pipelineCompileOptions.usesMotionBlur = false;
  pipelineCompileOptions.numPayloadValues = 2;
  pipelineCompileOptions.numAttributeValues = 8;
  pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

  pipelineLinkOptions.maxTraceDepth = 2;

  const std::string ptxCode = embedded_ptx_code;

  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK(optixModuleCreateFromPTX(optix_context, &moduleCompileOptions, &pipelineCompileOptions, ptxCode.c_str(),
                                       ptxCode.size(), log, &sizeof_log, &module));
  general_log_cb(log, sizeof_log);
}

/*! assembles the full pipeline of all programs */
void
MainRenderer::createPipeline()
{
  std::vector<OptixProgramGroup> programGroups;
  for (auto pg : raygenPGs)
    programGroups.push_back(pg);
  for (auto pg : missPGs)
    programGroups.push_back(pg);
  for (auto pg : hitgroupPGs)
    programGroups.push_back(pg);

  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK(optixPipelineCreate(optix_context, &pipelineCompileOptions, &pipelineLinkOptions, programGroups.data(),
                                  (int)programGroups.size(), log, &sizeof_log, &pipeline));
  general_log_cb(log, sizeof_log);

  OPTIX_CHECK(optixPipelineSetStackSize(/* [in] The pipeline to configure the stack size for */
                                        pipeline,
                                        /* [in] The direct stack size requirement for direct callables
                                                invoked from IS or AH. */
                                        2 * 1024,
                                        /* [in] The direct stack size requirement for direct callables
                                                invoked from RG, MS, or CH. */
                                        2 * 1024,
                                        /* [in] The continuation stack requirement. */
                                        2 * 1024,
                                        /* [in] The maximum depth of a traversable graph passed to trace. */
                                        3));
}

/*! render one frame */
void
MainRenderer::render()
{
  CUDA_CHECK(StreamSynchronize(stream));

  framebuffer.safe_swap();

  for (auto& v : volumes) {
    v.commit(stream);
  }

  std::unique_lock<std::mutex> camera_lock(camera_mutex);
  Camera camera = latest_camera;
  camera_lock.unlock();

  const float t = 0.66f;
  const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);
  launchParams.last_camera = launchParams.camera;
  launchParams.camera.position = camera.from;
  launchParams.camera.direction = normalize(camera.at - camera.from);
  launchParams.camera.horizontal = t * aspect * normalize(cross(launchParams.camera.direction, camera.up));
  launchParams.camera.vertical = cross(launchParams.camera.horizontal, launchParams.camera.direction) / aspect;

  // sanity check: make sure we launch only after first resize is already done:
  if (launchParams.frame.size.x <= 0 || launchParams.frame.size.y <= 0)
    return;

  launchParamsBuffer.upload_async(&launchParams, 1, stream);

  OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                          pipeline, stream,
                          /*! parameters and SBT */
                          launchParamsBuffer.d_pointer(), launchParamsBuffer.sizeInBytes, &sbt,
                          /*! dimensions of the launch: */
                          launchParams.frame.size.x, launchParams.frame.size.y, 1));

  framebuffer.background_download_async(stream);

  // sync - make sure the frame is rendered before we download and
  // display (obviously, for a high-performance application you
  // want to use streams and double-buffering, but for this simple
  // example, this will have to do)
  // CUDA_SYNC_CHECK();
}

/*! set camera to render with */
void
MainRenderer::set_camera(const Camera& camera)
{
  std::unique_lock<std::mutex> lock(camera_mutex);
  latest_camera = camera;
}

void
MainRenderer::set_camera(vec3f from, vec3f at, vec3f up)
{
  set_camera(Camera{ from, at, up });
}

/*! resize frame buffer to given resolution */
void
MainRenderer::resize(const vec2i& newSize)
{
  // resize our cuda frame buffer
  framebuffer.resize(newSize);

  // update the launch parameters that we'll pass to the optix launch:
  launchParams.frame.rgba = framebuffer.d_pointer();
  launchParams.frame.size = framebuffer.size();

  // and re-set the camera, since aspect may have changed
  set_camera(latest_camera);
}

/*! download the rendered color buffer */
void
MainRenderer::download_pixels(uint32_t h_pixels[])
{
  framebuffer.deepcopy(h_pixels);
}

/*! does all setup for the raygen program(s) we are going to use */
void
MainRenderer::createRaygenPrograms()
{
  // we can only have a single ray gen program per SBT record
  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc = {};
  pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pgDesc.raygen.module = module;
  pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(optix_context, &pgDesc, 1, &pgOptions, log, &sizeof_log, &raygenPG));
  general_log_cb(log, sizeof_log);
}

/*! does all setup for the miss program(s) we are going to use */
void
MainRenderer::createMissPrograms()
{
  missPGs.resize(RAY_TYPE_COUNT);

  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc = {};
  char log[2048];
  size_t sizeof_log = sizeof(log);

  // ------------------------------------------------------------------
  // radiance rays
  // ------------------------------------------------------------------
  sizeof_log = sizeof(log);
  pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgDesc.miss.module = module;
  pgDesc.miss.entryFunctionName = "__miss__background";
  OPTIX_CHECK(
    optixProgramGroupCreate(optix_context, &pgDesc, 1, &pgOptions, log, &sizeof_log, &missPGs[RADIANCE_RAY_TYPE]));
  general_log_cb(log, sizeof_log);

  // ------------------------------------------------------------------
  // shadow rays
  // ------------------------------------------------------------------
  sizeof_log = sizeof(log);
  pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgDesc.miss.module = module;
  pgDesc.miss.entryFunctionName = "__miss__shadow";
  OPTIX_CHECK(
    optixProgramGroupCreate(optix_context, &pgDesc, 1, &pgOptions, log, &sizeof_log, &missPGs[SHADOW_RAY_TYPE]));
  general_log_cb(log, sizeof_log);
}

/*! does all setup for the hitgroup program(s) we are going to use */
void
MainRenderer::createHitgroupPrograms()
{
  OptixProgramGroupDesc pgDesc = {};
  OptixProgramGroupOptions pgOptions = {};
  memset(&pgOptions, 0, sizeof(OptixProgramGroupOptions));

  // for this simple example, we set up a single hit group
  char log[2048];
  size_t sizeof_log;

  hitgroupPGs.resize(RAY_TYPE_COUNT);

  // ------------------------------------------------------------------
  // volumes
  // ------------------------------------------------------------------
  OptixProgramGroup* volumeHitGroupPG = &hitgroupPGs[0];
  {
    /* radiance ray */
    memset(&pgDesc, 0, sizeof(OptixProgramGroupDesc));
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__volume";
    pgDesc.hitgroup.moduleAH = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__volume";
    pgDesc.hitgroup.moduleIS = module;
    pgDesc.hitgroup.entryFunctionNameIS = "__intersection__volume";

    sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optix_context, &pgDesc, 1, &pgOptions, log, &sizeof_log,
                                        &volumeHitGroupPG[RADIANCE_RAY_TYPE]));
    general_log_cb(log, sizeof_log);

    /* shadow ray */
    memset(&pgDesc, 0, sizeof(OptixProgramGroupDesc));
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__volume_shadow";
    pgDesc.hitgroup.moduleAH = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__volume_shadow";
    pgDesc.hitgroup.moduleIS = module;
    pgDesc.hitgroup.entryFunctionNameIS = "__intersection__volume";

    sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optix_context, &pgDesc, 1, &pgOptions, log, &sizeof_log,
                                        &volumeHitGroupPG[SHADOW_RAY_TYPE]));
    general_log_cb(log, sizeof_log);
  }
}

/*! constructs the shader binding table */
void
MainRenderer::buildSBT()
{
  // ------------------------------------------------------------------
  // build raygen records
  // ------------------------------------------------------------------
  RaygenRecord raygenRecord = {};
  OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &raygenRecord));  
  raygenRecord.data = nullptr; /* for now ... */
  raygenRecordBuffer.alloc_and_upload_async(&raygenRecord, 1, stream);
  sbt.raygenRecord = raygenRecordBuffer.d_pointer();

  // ------------------------------------------------------------------
  // build miss records
  // ------------------------------------------------------------------
  std::vector<MissRecord> missRecords;
  for (auto&& ms : missPGs) {
    MissRecord rec = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(ms, &rec));
    rec.data = nullptr; /* for now ... */
    missRecords.push_back(rec);
  }
  missRecordsBuffer.alloc_and_upload_async(missRecords, stream);
  sbt.missRecordBase = missRecordsBuffer.d_pointer();
  sbt.missRecordStrideInBytes = sizeof(MissRecord);
  sbt.missRecordCount = (int)missRecords.size();

  // ------------------------------------------------------------------
  // build hitgroup records
  // ------------------------------------------------------------------
  std::vector<HitgroupRecord> hitgroupRecords;

  // table entries for aabbs
  for (auto&& m : volumes) {
    for (int rayID = 0; rayID < RAY_TYPE_COUNT; rayID++) {
      HitgroupRecord rec{};
      // all meshes use the same code, so all same hit group
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0 * RAY_TYPE_COUNT + rayID], &rec));
      rec.data = m.get_sbt_pointer(stream);
      hitgroupRecords.push_back(rec);
    }
  }

  hitgroupRecordsBuffer.alloc_and_upload_async(hitgroupRecords, stream);
  sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
  sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
  sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

OptixTraversableHandle
MainRenderer::buildAccel_instances(OptixDeviceContext optixContext)
{
  // ==================================================================
  // Create Instances
  // ==================================================================

  // we want to treat one volume as a instance
  for (size_t i = 0; i < volumes.size(); ++i) {
    auto vol = volumes[i];
    auto gas = vol.buildas(optixContext);
    // setup instance
    OptixInstance instance = {};
    vol.transform(instance.transform);
    instance.instanceId = instances.size();
    instance.visibilityMask = 255;
    instance.sbtOffset = i * RAY_TYPE_COUNT;
    instance.flags = OPTIX_INSTANCE_FLAG_NONE;
    instance.traversableHandle = gas;
    instances.push_back(instance);
  }

  instancesBuffer.alloc_and_upload_async(instances, 0 /* stream */);

  // ==================================================================
  // Create Inputs
  // ==================================================================

  OptixBuildInput iasInput = {};
  iasInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  iasInput.instanceArray.instances = instancesBuffer.d_pointer();
  iasInput.instanceArray.numInstances = instances.size();

  std::vector<OptixBuildInput> inputs = { iasInput };
  return buildas_exec(optixContext, inputs, instancesAsBuffer);
}

void
MainRenderer::set_scene(int ac, char** av)
{
  /* range = (-7.6862547926737625e-19,  0.00099257915280759335) */
  auto v = StructuredRegularVolume();
  v.center = vec3f(0.f, 0.f, 0.f);
  v.scale = vec3f(3.0f);
  v.load_from_file("volume/3.900E-04_H2O2.raw", vec3i(600, 750, 500), VALUE_TYPE_FLOAT, 0.f, 0.0003f);
  v.set_transfer_function(CreateColorMap("diverging/RdBu"), CreateArray1D(std::vector<float>{ 0.f, 1.f }));
  volumes.push_back(v);

  const std::vector<float4>& arr_c = *((const std::vector<float4>*)colormap::data.at("diverging/RdBu"));
  for (int i = 0; i < arr_c.size(); ++i) {
    float p = (float)i / (arr_c.size() - 1);
    tfn_colors.push_back(p);
    tfn_colors.push_back(arr_c[i].x);
    tfn_colors.push_back(arr_c[i].y);
    tfn_colors.push_back(arr_c[i].z);
  }
  auto arr_o = std::vector<float>{ 0.f, 1.f };
  for (int i = 0; i < arr_o.size(); ++i) {
    float p = (float)i / (arr_o.size() - 1);
    tfn_alphas.push_back(p);
    tfn_alphas.push_back(arr_o[i]);
  }
}

} // namespace host
} // namespace ovr
