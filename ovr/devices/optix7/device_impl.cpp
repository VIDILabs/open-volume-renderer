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

// our own classes, partly shared between host and device
#include "device_impl.h"

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

#include <generate_mask.h>

namespace ovr::optix7 {

// ------------------------------------------------------------------
// ------------------------------------------------------------------

extern "C" char embedded_ptx_code__raymarching[];
extern "C" char embedded_ptx_code__pathtracing[];

static void
context_log_cb(unsigned int level, const char* tag, const char* message, void*)
{
#ifndef NDEBUG
  fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
#endif
}

static void
general_log_cb(const char* log, size_t sizeof_log)
{
#ifndef NDEBUG
  if (sizeof_log > 1)
    fprintf(stdout, log);
#endif
}

// ------------------------------------------------------------------
// ------------------------------------------------------------------

/*! constructor - performs all setup, including initializing
  optix, creates module, pipeline, programs, SBT, etc. */
void
DeviceOptix7::Impl::init(int argc, const char** argv, DeviceOptix7* p)
{
  if (parent) {
    throw std::runtime_error("[optix7] device already initialized!");
  }
  else {
    parent = p;

    buildScene(parent->current_scene);

    initOptix();

    createContext();

    createModule();

    createPrograms();

    params.traversable = buildTLAS();

    createPipeline();

    buildSBT();

    params_buffer.alloc(sizeof(params));
  
  }

  std::cout << "[optix7] setup done" << std::endl;
}

void
DeviceOptix7::Impl::swap()
{
  CUDA_CHECK(cudaStreamSynchronize(framebuffer_stream));

  framebuffer.safe_swap();

  /* now working on the background stream */
  framebuffer_stream = framebuffer.current_stream();
}

void
DeviceOptix7::Impl::commit()
{
  if (parent->params.fbsize.update()) {
    CUDA_SYNC_CHECK(); /* stio all async rendering */
    params.frame.size = parent->params.fbsize.ref();
    framebuffer.resize(parent->params.fbsize.ref());
    framebuffer_size_updated = true;
    framebuffer_reset = true;
  }

  /* commit other data */
  if (parent->params.camera.update() || framebuffer_size_updated) {
    Camera camera = parent->params.camera.ref();

    /* the factor '2.f' here might be unnecessary, but I want to match ospray's implementation */
    const float fovy = camera.perspective.fovy;
    const float t = 2.f /* (note above) */ * tan(fovy * 0.5f * M_PI / 180.f);
    const float aspect = params.frame.size.x / float(params.frame.size.y);
    params.last_camera = params.camera;
    params.camera.position = camera.from;
    params.camera.direction = normalize(camera.at - camera.from);
    params.camera.horizontal = t * aspect * normalize(cross(params.camera.direction, camera.up));
    params.camera.vertical = cross(params.camera.horizontal, params.camera.direction) / aspect;

    // std::cout << "camera update" << std::endl;
    // std::cout << "  from: " << camera.from << std::endl;
    // std::cout << "  at:   " << camera.at << std::endl;
    // std::cout << "  up:   " << camera.up << std::endl;

    framebuffer_reset = true;
  }

  if (parent->params.tfn.update()) {
    const auto& tfn = parent->params.tfn.ref();
    for (auto& v : volumes) {
      v.set_transfer_function(tfn.tfn_colors, tfn.tfn_alphas, tfn.tfn_value_range);
    }
    volumes_changed = true;
    framebuffer_reset = true;
  }

  if (parent->params.focus_center.update()) {
    params.focus_center = parent->params.focus_center.ref();
    framebuffer_reset = true;
  }

  if (parent->params.focus_scale.update()) {
    params.focus_scale = parent->params.focus_scale.ref();
    framebuffer_reset = true;
  }

  if (parent->params.base_noise.update()) {
    params.base_noise = parent->params.base_noise.ref();
    framebuffer_reset = true;
  }

  if (parent->params.sample_per_pixel.update()) {
    params.sample_per_pixel = parent->params.sample_per_pixel.ref();
    framebuffer_reset = true;
  }

  if (parent->params.path_tracing.update()) {
    params.enable_path_tracing = parent->params.path_tracing.ref();
    framebuffer_reset = true;
  }

  if (parent->params.sparse_sampling.update()) {
    params.enable_sparse_sampling = parent->params.sparse_sampling.ref();
    framebuffer_reset = true;
  }

  if (parent->params.frame_accumulation.update()) {
    params.enable_frame_accumulation = parent->params.frame_accumulation.ref();
    framebuffer_reset = true;
  }

  if (parent->params.volume_sampling_rate.update()) {
    for (auto& v : volumes) {
      v.set_sampling_rate(parent->params.volume_sampling_rate.get());
    }
    volumes_changed = true;
    framebuffer_reset = true;
  }
}

void
DeviceOptix7::Impl::render()
{
  /* commit others */
  framebuffer_size_updated = false;
  framebuffer_accum_rgba.resize(params.frame.size.long_product() * sizeof(vec4f));
  framebuffer_accum_grad.resize(params.frame.size.long_product() * sizeof(vec3f));

  if (volumes_changed) {
    for (auto& v : volumes) {
      v.commit(framebuffer_stream);
    }
    volumes_changed = false;
  }

  /* sanity check: make sure we launch only after first resize is already done: */
  if (params.frame.size.x <= 0 || params.frame.size.y <= 0)
    return;

  /* layout has to match the framebuffer definition in params.h */
  params.frame.rgba = (vec4f*)framebuffer.device_pointer(/*layout=*/0);
  params.frame.grad = (vec3f*)framebuffer.device_pointer(/*layout=*/1);
  params.frame_accum_rgba = (vec4f*)framebuffer_accum_rgba.d_pointer();
  params.frame_accum_grad = (vec3f*)framebuffer_accum_grad.d_pointer();

  /* handle frame accumulation */
  if (params.enable_frame_accumulation) {
    if (framebuffer_reset) {
      framebuffer.reset();
      // framebuffer_accum_rgba.nullify(framebuffer_stream);
      // framebuffer_accum_grad.nullify(framebuffer_stream);
      framebuffer_reset = false;
      params.frame_index = 0;
    }
  }
  else {
    /* in non-sparse sampling mode, the entire frame will be re-written anyway. */
    if (params.enable_sparse_sampling) {
      framebuffer.reset();
    }
  }

  params.frame_index++;
  params.frame.size_rcp = vec2f(1.f) / vec2f(params.frame.size);

  if (params.enable_path_tracing)
    stb_current = &sbt_pathtracing_main.sbt;
  else
    stb_current = &sbt_raymarching_main.sbt;


  /* the number of kernels to launch for optix7 */
  const auto launch_dims =
    params.enable_sparse_sampling ? createSparseSamples() : vec3i(params.frame.size.x, params.frame.size.y, 1);

  /* this has to be the last step! */
  params_buffer.upload_async(&params, 1, framebuffer_stream);

  OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                          pipeline.handle, framebuffer_stream,
                          /*! parameters and SBT */
                          params_buffer.d_pointer(), params_buffer.sizeInBytes, stb_current,
                          /*! dimensions of the launch: */
                          launch_dims.x, launch_dims.y, launch_dims.z));

  CUDA_SYNC_CHECK();

  parent->variance = 0.f; /* TODO compute real variance */

  // framebuffer.download_async(); CUDA_SYNC_CHECK();
}

void
DeviceOptix7::Impl::mapframe(FrameBufferData* fb)
{
  /* layout has to match the framebuffer definition in params.h */
  // fb->rgba = (vec4f*)framebuffer.host_pointer(0);
  // fb->grad = (vec3f*)framebuffer.host_pointer(1);

  const size_t num_bytes = framebuffer.size().long_product();
  fb->rgba->set_data(framebuffer.device_pointer(0), num_bytes * sizeof(vec4f), CrossDeviceBuffer::DEVICE_CUDA);
  fb->grad->set_data(framebuffer.device_pointer(1), num_bytes * sizeof(vec3f), CrossDeviceBuffer::DEVICE_CUDA);
}

void
DeviceOptix7::Impl::buildScene(Scene& scene)
{
  assert(scene.instances.size() == 1);
  assert(scene.instances[0].models.size() == 1);
  assert(scene.instances[0].models[0].type == scene::Model::VOLUMETRIC_MODEL);
  assert(scene.instances[0].models[0].volume_model.volume.type == scene::Volume::STRUCTURED_REGULAR_VOLUME);

  // if (!scene.lights.empty())
  //   params.light_directional_pos = scene.lights[0].position;

  auto& scene_tfn = scene.instances[0].models[0].volume_model.transfer_function;
  auto& scene_volume = scene.instances[0].models[0].volume_model.volume.structured_regular;

  vec3f scale = scene_volume.grid_spacing * vec3f(scene_volume.data->dims);
  vec3f translate = scene_volume.grid_origin;
  
  std::cout << "scale " << scale.x << " " << scale.y << " " << scale.z << std::endl;
  std::cout << "translate " << translate.x << " " << translate.y << " " << translate.z << std::endl;

  // TODO support other parameters //
  auto v = StructuredRegularVolume();
  v.matrix = affine3f::translate(translate) * affine3f::scale(scale);

  v.load_from_array3d_scalar(scene_volume.data);
  v.set_transfer_function(scene_tfn.color, scene_tfn.opacity, scene_tfn.value_range);
  v.set_sampling_rate(scene.volume_sampling_rate);

  volumes.emplace_back(std::move(v));
}

vec3i
DeviceOptix7::Impl::createSparseSamples()
{
#ifdef OVR_OPTIX7_MASKING_VIA_DIRECT_SAMPLING
  {
    /* allocate buffer */
    const size_t num_bytes = params.frame.size.long_product() * sizeof(float) /
                             (params.sparse_sampling.downsample_factor * params.sparse_sampling.downsample_factor);

    if (num_bytes != dist_logistic.sizeInBytes)
      dist_logistic.resize(num_bytes);
    if (num_bytes != dist_uniform.sizeInBytes)
      dist_uniform.resize(num_bytes);

    /* generate radius using logistic distribution, angle using uniform distribution */
    generate_logistic_dist(dist_logistic, 0.f, 1.f);
    generate_uniform_dist(dist_uniform, 0.f, 1.f);
    params.sparse_sampling.dist_logistic = (float*)dist_logistic.d_pointer();
    params.sparse_sampling.dist_uniform = (float*)dist_uniform.d_pointer();

    return vec3i(params.frame.size.x / params.sparse_sampling.downsample_factor,
                 params.frame.size.y / params.sparse_sampling.downsample_factor, 1);
  }
#endif

#ifdef OVR_OPTIX7_MASKING_VIA_STREAM_COMPACTION
  {
    /* allocate buffer */
    const auto& fbsize = params.frame.size;
    xs_and_ys.resize(fbsize.long_product() * sizeof(int32_t) * 2);
    params.sparse_sampling.xs_and_ys = (int32_t*)xs_and_ys.d_pointer();

    const auto size = generate_sparse_sampling_mask_d(params.sparse_sampling.xs_and_ys, params.frame_index, fbsize,
                                                      params.focus_center, params.focus_scale, params.base_noise);

    return vec3i((uint32_t)size / 2, 1, 1);
  }
#endif
}

/*! helper function that initializes optix and checks for errors */
void
DeviceOptix7::Impl::initOptix()
{
  // -------------------------------------------------------
  // check for available optix7 capable devices
  // -------------------------------------------------------
  cudaFree(0);
  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0)
    throw std::runtime_error("[optix7] no CUDA capable devices found!");
  std::cout << "[optix7] found " << numDevices << " CUDA devices" << std::endl;

  // -------------------------------------------------------
  // initialize optix
  // -------------------------------------------------------
  OPTIX_CHECK(optixInit());
  std::cout << "[optix7] successfully initialized OptiX" << std::endl;
}

/*! creates and configures a optix device context (in this simple
  example, only for the primary GPU device) */
void
DeviceOptix7::Impl::createContext()
{
  // for this sample, do everything on one device
  const int device_id = 0;
  CUDA_CHECK(cudaSetDevice(device_id));

  CUDA_CHECK(cudaGetDeviceProperties(&cuda_device_props, device_id));
  std::cout << "[optix7] running on device: " << cuda_device_props.name << std::endl;

  CUresult res = cuCtxGetCurrent(&cuda_context);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "[optix7] Error querying current context: error code %d\n", res);
    throw std::runtime_error("[optix7] Error querying current context");
  }

  OPTIX_CHECK(optixDeviceContextCreate(cuda_context, 0, &optix_context));
  OPTIX_CHECK(optixDeviceContextSetLogCallback(optix_context, context_log_cb, nullptr, 4));

  framebuffer.create();
}

/*! creates the module that contains all the programs we are going
    to use. in this simple example, we use a single module from a
    single .cu file, using a single embedded ptx string */
void 
DeviceOptix7::Impl::Module::createModule(OptixDeviceContext optix_context, const Pipeline& pipeline, const std::string& ptx_code)
{
  compile_opts.maxRegisterCount = 100;

  compile_opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
  // module.compile_opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_2;
  // module.compile_opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_1;
  // module.compile_opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;

  compile_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
  // module.compile_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
  // module.compile_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
  // module.compile_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  // const std::string ptx_code = embedded_ptx_code;

  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK(optixModuleCreateFromPTX(optix_context, &compile_opts, &pipeline.compile_opts,
                                       /* shader program */ ptx_code.c_str(), ptx_code.size(),
                                       /* logs and output */ log, &sizeof_log, &handle));
  general_log_cb(log, sizeof_log);
}

void
DeviceOptix7::Impl::createModule()
{
  pipeline.compile_opts = {};
  pipeline.compile_opts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
  pipeline.compile_opts.usesMotionBlur = false;
  pipeline.compile_opts.numPayloadValues = 2;
  pipeline.compile_opts.numAttributeValues = 8;
  pipeline.compile_opts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline.compile_opts.pipelineLaunchParamsVariableName = "optix_launch_params";
  pipeline.link_opts.maxTraceDepth = 31;

  // create sub-modules
  module_raymarching.createModule(optix_context, pipeline, embedded_ptx_code__raymarching);
  module_pathtracing.createModule(optix_context, pipeline, embedded_ptx_code__pathtracing);
}

OptixProgramGroup 
DeviceOptix7::Impl::addProgram(OptixDeviceContext optix_context, OptixProgramGroupDesc desc)
{
  programs.emplace_back();
  OptixProgramGroupOptions options = {};
  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(optix_context, &desc, 1, &options, log, &sizeof_log, &programs.back()));
  general_log_cb(log, sizeof_log);
  return programs.back();
}

/*! assembles the full pipeline of all programs */
void
DeviceOptix7::Impl::createPipeline()
{
  // std::vector<OptixProgramGroup> program_groups;
  // for (auto pg : program.raygens)
  //   program_groups.push_back(pg);
  // for (auto pg : program.misses)
  //   program_groups.push_back(pg);
  // for (auto pg : program.hitgroups)
  //   program_groups.push_back(pg);

  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK(optixPipelineCreate(optix_context, &pipeline.compile_opts, &pipeline.link_opts, programs.data(),
                                  (int)programs.size(), log, &sizeof_log, &pipeline.handle));
  general_log_cb(log, sizeof_log);

  OPTIX_CHECK(optixPipelineSetStackSize(/* [in] The pipeline to configure the stack size for */
                                        pipeline.handle,
                                        /* [in] The direct stack size requirement for
                                                direct callables from IS or AH. */
                                        2 * 1024,
                                        /* [in] The direct stack size requirement for
                                                direct callables from RG, MS, or CH. */
                                        2 * 1024,
                                        /* [in] The continuation stack requirement. */
                                        2 * 1024,
                                        /* [in] The maximum depth of a traversable graph passed to trace. */
                                        3));
}

/*! does all setup for the raygen program(s) we are going to use */
void
DeviceOptix7::Impl::createPrograms()
{
  {
    auto& main = sbt_raymarching_main;
    auto& handle = module_raymarching.handle;
    main.raygen = addProgram(optix_context, createRaygenDesc(handle, "__raygen__render_frame"));
    main.misses = {
      addProgram(optix_context, createMissDesc(handle, "__miss__raymarching")), // RAYMARCHING_RAY_TYPE
      addProgram(optix_context, createMissDesc(handle, "__miss__shadow"     )), // SHADOW_RAY_TYPE
    };
    main.hitgroups = {
      addProgram(optix_context, createHitgroupDesc(handle, "__closesthit__volume_raymarching", "", "__intersection__volume")), // RAYMARCHING_RAY_TYPE
      addProgram(optix_context, createHitgroupDesc(handle, "__closesthit__volume_shadow"     , "", "__intersection__volume")), // SHADOW_RAY_TYPE
    };
  }

  {
    auto& main = sbt_pathtracing_main;
    auto& handle = module_pathtracing.handle;
    main.raygen = addProgram(optix_context, createRaygenDesc(handle, "__raygen__render_frame"));
    main.misses = {
      addProgram(optix_context, createMissDesc(handle, "__miss__pathtracing")), // PATHTRACING_RAY_TYPE
    };
    main.hitgroups = {
      addProgram(optix_context, createHitgroupDesc(handle, "__closesthit__volume_pathtracing", "", "__intersection__volume")), // PATHTRACING_RAY_TYPE
    };
  }
}

/*! constructs the shader binding table */
void
DeviceOptix7::Impl::StbGroup::buildSBT(StbGraph graph)
{
  // ------------------------------------------------------------------
  // build raygen records
  // ------------------------------------------------------------------
  std::vector<RaygenRecord> raygen_records;
  // for (auto&& rg : records.raygens) 
  {
    RaygenRecord rec = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen, &rec));
    rec.data = nullptr; /* for now ... */
    raygen_records.push_back(rec);
  }
  raygen_buffer.alloc_and_upload_async(raygen_records, /*stream=*/0);
  sbt.raygenRecord = raygen_buffer.d_pointer();

  // ------------------------------------------------------------------
  // build miss records
  // ------------------------------------------------------------------
  std::vector<MissRecord> miss_records;
  for (auto&& ms : misses) {
    MissRecord rec = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(ms, &rec));
    rec.data = nullptr; /* for now ... */
    miss_records.push_back(rec);
  }
  miss_buffer.alloc_and_upload_async(miss_records, /*stream=*/0);
  sbt.missRecordBase = miss_buffer.d_pointer();
  sbt.missRecordStrideInBytes = sizeof(MissRecord);
  sbt.missRecordCount = (int)miss_records.size();

  // ------------------------------------------------------------------
  // build hitgroup records
  // ------------------------------------------------------------------
  std::vector<HitgroupRecord> hitgroup_records;

  // table entries for aabbs
  for (auto&& m : graph.aabbs) {
    for (auto&& hg : hitgroups) {
      HitgroupRecord rec{};
      // all meshes use the same code, so all same hit group
      OPTIX_CHECK(optixSbtRecordPackHeader(hg, &rec));
      rec.data = m->get_sbt_pointer(/*stream=*/0);
      hitgroup_records.push_back(rec);
    }
  }

  hitgroup_buffer.alloc_and_upload_async(hitgroup_records, /*stream=*/0);
  sbt.hitgroupRecordBase = hitgroup_buffer.d_pointer();
  sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
  sbt.hitgroupRecordCount = (int)hitgroup_records.size();
}

void
DeviceOptix7::Impl::buildSBT()
{
  StbGraph graph;
  for (auto& v : volumes) {
    graph.aabbs.push_back(&v);
  }
  sbt_raymarching_main.buildSBT(graph);
  sbt_pathtracing_main.buildSBT(graph);
}

OptixTraversableHandle
DeviceOptix7::Impl::buildTLAS()
{
  // ==================================================================
  // Create Instances
  // ==================================================================

  // we want to treat one volume as a instance
  for (int i = 0; i < volumes.size(); ++i) {
    auto& vol = volumes[i];
    vol.buildas(optix_context);
    // setup instance
    OptixInstance instance = {};
    vol.transform(instance.transform);
    instance.instanceId = (unsigned int)instances.size();
    instance.visibilityMask = VISIBILITY_VOLUME /*255*/;
    instance.sbtOffset = 0; // i * RAY_TYPE_COUNT; // hack, assume only one volume for now
    instance.flags = OPTIX_INSTANCE_FLAG_NONE;
    instance.traversableHandle = vol.blas;
    instances.push_back(instance);
  }

  instances_buffer.alloc_and_upload_async(instances, /*stream=*/0);

  // ==================================================================
  // Create Inputs
  // ==================================================================

  OptixBuildInput iasInput = {};
  iasInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  iasInput.instanceArray.instances = instances_buffer.d_pointer();
  iasInput.instanceArray.numInstances = (unsigned int)instances.size();

  std::vector<OptixBuildInput> inputs = { iasInput };
  return buildas_exec(optix_context, inputs, instances_accel_struct_buffer);
}

} // namespace ovr::optix7
