#include "python.h"
#include <scene.h>
#include <renderer.h>
#include <serializer/serializer.h>

PYBIND11_MODULE(ovrpy, m)
{
/*
 * Math
 *
 */ 
OVR_PY_NAMED_STRUCT(ovr::vec3f, "vec3f")
.def_init()
.def_class_field(ovr::vec3f, x)
.def_class_field(ovr::vec3f, y)
.def_class_field(ovr::vec3f, z);

OVR_PY_NAMED_STRUCT(ovr::vec2f, "vec2f")
.def_init()
.def_class_field(ovr::vec2f, x)
.def_class_field(ovr::vec2f, y);

OVR_PY_NAMED_STRUCT(ovr::vec2i, "vec2i")
.def_init()
.def_class_field(ovr::vec2i, x)
.def_class_field(ovr::vec2i, y);

/*
 * Scene
 *
 */
OVR_PY_NAMED_STRUCT(ovr::scene::Scene, "Scene")
.def_class_field(ovr::scene::Scene, camera)
.def_class_field(ovr::scene::Scene, ao_samples)
.def_class_field(ovr::scene::Scene, spp)
.def_class_field(ovr::scene::Scene, volume_sampling_rate)
.def_class_field(ovr::scene::Scene, roulette_path_length)
.def_class_field(ovr::scene::Scene, max_path_length)
.def_class_field(ovr::scene::Scene, use_dda)
.def_class_field(ovr::scene::Scene, parallel_view)
.def_class_field(ovr::scene::Scene, simple_path_tracing);

def_named_method(ovr::scene::create_scene, "create_scene");

/*
 * Camera
 *
 */ 

OVR_PY_NAMED_STRUCT(ovr::scene::Camera::PerspectiveCamera, "PerspectiveCamera")
.def_class_field(ovr::scene::Camera::PerspectiveCamera, fovy);

OVR_PY_NAMED_STRUCT(ovr::scene::Camera::OrthographicCamera, "OrthographicCamera")
.def_class_field(ovr::scene::Camera::OrthographicCamera, height);

OVR_PY_NAMED_STRUCT(ovr::scene::Camera, "Camera")
.def_init()
.def_class_field(ovr::scene::Camera, eye)
.def_class_field(ovr::scene::Camera, at)
.def_class_field(ovr::scene::Camera, up)
.def_class_field(ovr::scene::Camera, type)
.def_class_field(ovr::scene::Camera, perspective)
.def_class_field(ovr::scene::Camera, orthographic);


/*
 * Renderer
 *
 */ 
OVR_PY_NAMED_STRUCT(ovr::MainRenderer::FrameBufferData, "FrameBufferData")
.def_init()
.def_class_lambda(ovr::MainRenderer:FrameBufferData, rgba, [](ovr::MainRenderer::FrameBufferData& self) {
        float* frame = (float*)self.rgba->to_cpu()->data();
        auto size = self.rgba->get_size<float>();
        return py::array_t<float>(size, frame);
})
.def_class_lambda(ovr::MainRenderer::FrameBufferData, grad, [](ovr::MainRenderer::FrameBufferData& self) {
        float* frame = (float*)self.grad->to_cpu()->data();
        auto size = self.grad->get_size<float>();
        return py::array_t<float>(size, frame);
});

OVR_PY_STRUCT_PTR(ovr::MainRenderer, "Renderer")
.def_class_lambda(ovr::MainRenderer, init, [](ovr::MainRenderer& self, std::vector<std::string> args, ovr::scene::Scene scene, ovr::scene::Camera camera) {
        std::vector<const char*> cstr(args.size());
        for(int i = 0; i < args.size(); i++) cstr[i] = args[i].c_str();
        self.init(cstr.size(), cstr.data(), scene, camera);
})
.def_class_method(ovr::MainRenderer, swap)
.def_class_method(ovr::MainRenderer, commit)
.def_class_method(ovr::MainRenderer, render)
.def_class_method(ovr::MainRenderer, mapframe)
.def_class_method(ovr::MainRenderer, set_fbsize)
.def_class_method_overload(ovr::MainRenderer, set_camera, "set_camera", void, const ovr::scene::Camera&)
.def_class_method_overload(ovr::MainRenderer, set_camera, "set_camera_vectors", void, ovr::vec3f, ovr::vec3f, ovr::vec3f)
.def_class_method(ovr::MainRenderer, set_transfer_function)
.def_class_method(ovr::MainRenderer, set_focus)
.def_class_method(ovr::MainRenderer, set_sample_per_pixel)
.def_class_method(ovr::MainRenderer, set_add_lights)
.def_class_method(ovr::MainRenderer, set_sparse_sampling)
.def_class_method(ovr::MainRenderer, set_path_tracing)
.def_class_method(ovr::MainRenderer, set_photonmapping)
.def_class_method(ovr::MainRenderer, set_frame_accumulation)
.def_class_method(ovr::MainRenderer, set_volume_sampling_rate)
.def_class_method(ovr::MainRenderer, set_volume_density_scale)
.def_class_method(ovr::MainRenderer, set_mat_ambient)
.def_class_method(ovr::MainRenderer, set_mat_diffuse)
.def_class_method(ovr::MainRenderer, set_mat_specular)
.def_class_method(ovr::MainRenderer, set_mat_shininess)
.def_class_method(ovr::MainRenderer, set_light_phi)
.def_class_method(ovr::MainRenderer, set_light_theta)
.def_class_method(ovr::MainRenderer, set_light_radius)
.def_class_method(ovr::MainRenderer, set_light_intensity);

def_named_method(create_renderer, "create_renderer");

}
