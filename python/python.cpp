#include "python.h"
#include <scene.h>
#include <renderer.h>
#include <serializer/serializer.h>
#include <algorithm>
#include <cmath>

PYBIND11_MODULE(ovrpy, m)
{
/*
 * Math
 *
 */ 
OVR_PY_NAMED_STRUCT(ovr::vec3f, "vec3f")
.def_init()
.def_init(float)
.def_init(float,float,float)
.def_class_field(ovr::vec3f, x)
.def_class_field(ovr::vec3f, y)
.def_class_field(ovr::vec3f, z)
.def_class_copy(ovr::vec3f)
.def_class_lambda(ovr::vec3f, __repr__, [](const ovr::vec3f& self) { 
        return "(" + std::to_string(self.x) + ", " + std::to_string(self.y) + ", " + std::to_string(self.z) + ")"; 
});

OVR_PY_NAMED_STRUCT(ovr::vec2f, "vec2f")
.def_init()
.def_init(float)
.def_init(float,float)
.def_class_field(ovr::vec2f, x)
.def_class_field(ovr::vec2f, y)
.def_class_copy(ovr::vec2f)
.def_class_lambda(ovr::vec2f, __repr__, [](const ovr::vec2f& self) { 
        return "(" + std::to_string(self.x) + ", " + std::to_string(self.y) + ")"; 
});

OVR_PY_NAMED_STRUCT(ovr::vec2i, "vec2i")
.def_init()
.def_init(int)
.def_init(int,int)
.def_class_field(ovr::vec2i, x)
.def_class_field(ovr::vec2i, y)
.def_class_copy(ovr::vec2i)
.def_class_lambda(ovr::vec2i, __repr__, [](const ovr::vec2i& self) { 
        return "(" + std::to_string(self.x) + ", " + std::to_string(self.y) + ")"; 
});

OVR_PY_NAMED_STRUCT(ovr::box3f, "box3f")
.def_init()
.def_init(float, float)
.def_init(ovr::vec3f, ovr::vec3f)
.def_class_field(ovr::box3f, lower)
.def_class_field(ovr::box3f, upper)
.def_class_copy(ovr::box3f)
.def_class_lambda(ovr::box3f, __repr__, [](const ovr::box3f& self) { 
        return "[(" + std::to_string(self.lower.x) + ", " + std::to_string(self.lower.y) + ", " + std::to_string(self.lower.z) + ")\n" +
               " (" + std::to_string(self.upper.x) + ", " + std::to_string(self.upper.y) + ", " + std::to_string(self.upper.z) + ")]";
});

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
.def_class_field(ovr::scene::Scene, simple_path_tracing)
.def_class_method(ovr::scene::Scene, get_bounds)
.def_class_method(ovr::scene::Scene, print)
.def_class_copy(ovr::scene::Scene);

def_named_method(create_scene_default, "create_scene");

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
.def_class_field(ovr::scene::Camera, orthographic)
.def_class_copy(ovr::scene::Camera);


/*
 * Renderer
 *
 */ 
OVR_PY_NAMED_STRUCT(ovr::RenderStats, "RenderStats")
.def_class_field(ovr::RenderStats, pixel_index)
.def_class_field(ovr::RenderStats, ray_direction)
.def_class_field(ovr::RenderStats, illumination_direct)
.def_class_field(ovr::RenderStats, illumination_indirect);

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
})
.def_class_lambda(ovr::MainRenderer::FrameBufferData, stats, [](ovr::MainRenderer::FrameBufferData& self) {
        ovr::RenderStats* frame = (ovr::RenderStats*)self.stats->to_cpu()->data();
        auto size = self.stats->get_size<ovr::RenderStats>();
        std::vector<ovr::RenderStats> result (frame, frame+size);
        return result;
})
.def_class_lambda(ovr::MainRenderer::FrameBufferData, stats_as_memoryview, [](ovr::MainRenderer::FrameBufferData& self) {
        uint8_t* frame = (uint8_t*)self.stats->to_cpu()->data();
        auto size = self.stats->get_size_in_bytes();
        return py::memoryview::from_memory(frame, size);
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
.def_class_method(ovr::MainRenderer, set_sparse_sampling)
.def_class_method(ovr::MainRenderer, set_path_tracing)
.def_class_method(ovr::MainRenderer, set_frame_accumulation)
.def_class_method(ovr::MainRenderer, set_volume_sampling_rate)
.def_class_method(ovr::MainRenderer, set_volume_density_scale);

auto render_to_framebuffer = [](ovr::MainRenderer& renderer) {
        renderer.commit();
        renderer.render();
        renderer.swap();

        ovr::MainRenderer::FrameBufferData fb;
        renderer.mapframe(&fb);
        return fb;
};

auto render_to_image = [render_to_framebuffer](
        ovr::MainRenderer& renderer,
        py::object fbsize_obj,
        bool scrub,
        bool clip) {
        auto fb = render_to_framebuffer(renderer);
        const float* frame = (const float*)fb.rgba->to_cpu()->data();
        const auto count = fb.rgba->get_size<float>();

        std::vector<py::ssize_t> shape;
        if (fbsize_obj.is_none()) {
                shape = { (py::ssize_t)count };
        }
        else {
                const auto fbsize = fbsize_obj.cast<ovr::vec2i>();
                const auto expected = (size_t)fbsize.x * (size_t)fbsize.y * 4;
                if (expected != count) {
                        throw std::runtime_error("fbsize does not match mapped RGBA buffer size");
                }
                shape = { (py::ssize_t)fbsize.y, (py::ssize_t)fbsize.x, 4 };
        }

        py::array_t<float> result(shape);
        auto* out = (float*)result.mutable_data();
        for (size_t i = 0; i < count; ++i) {
                float value = frame[i];
                if (scrub && !std::isfinite(value)) {
                        value = value > 0.f ? 1.f : 0.f;
                }
                if (clip) {
                        value = std::min(1.f, std::max(0.f, value));
                }
                out[i] = value;
        }
        return result;
};

m.def("render_to_framebuffer", render_to_framebuffer,
      py::arg("renderer"),
      R"pbdoc(
Render the current renderer state and return a mapped framebuffer.

This is a convenience wrapper for the normal frame lifecycle:
``commit()`` -> ``render()`` -> ``swap()`` -> ``mapframe()``.

The ``swap()`` step is required for backends such as OptiX that render into
a back buffer and only expose the completed frame to ``mapframe()`` after the
front/back buffers are swapped. OSPRay currently treats ``swap()`` as a no-op,
so this helper is safe to use for both backends.

Parameters
----------
renderer:
    An initialized ``ovrpy.Renderer``. Configure camera, framebuffer size,
    sampling parameters, transfer function, etc. before calling this helper.

Returns
-------
ovrpy.FrameBufferData
    The mapped framebuffer. Use ``rgba()``, ``grad()``, or ``stats()`` to read
    individual buffers.
)pbdoc");

m.def("render_to_image", render_to_image,
      py::arg("renderer"),
      py::arg("fbsize") = py::none(),
      py::kw_only(),
      py::arg("scrub") = true,
      py::arg("clip") = false,
      R"pbdoc(
Render the current renderer state to a detached float32 RGBA NumPy array.

This helper performs ``commit()``, ``render()``, ``swap()``, and ``mapframe()``
before copying the RGBA buffer into a NumPy array that is independent of the
renderer-owned framebuffer memory.

Parameters
----------
renderer:
    An initialized ``ovrpy.Renderer``.
fbsize:
    Optional ``ovrpy.vec2i`` framebuffer size. When provided, the returned
    array is reshaped to ``(fbsize.y, fbsize.x, 4)``. When omitted, the result
    is a flat ``(width * height * 4,)`` array.
scrub:
    If true, replace non-finite values with finite display-friendly values:
    ``NaN`` and ``-Inf`` become ``0.0``; ``+Inf`` becomes ``1.0``.
clip:
    If true, clamp values to ``[0.0, 1.0]``. This is useful before converting
    to 8-bit image formats such as PNG.

Returns
-------
numpy.ndarray
    A copied float32 RGBA array.
)pbdoc");

m.def("render_scene_to_image",
      [render_to_image](
              const std::string& backend,
              ovr::scene::Scene scene,
              ovr::vec2i fbsize,
              py::object camera_obj,
              py::object sample_per_pixel_obj,
              py::object path_tracing_obj,
              py::object frame_accumulation_obj,
              py::object volume_sampling_rate_obj,
              py::object volume_density_scale_obj,
              bool scrub,
              bool clip) {
              auto renderer = create_renderer(backend);
              renderer->set_fbsize(fbsize);
              const auto camera = camera_obj.is_none()
                      ? scene.camera
                      : camera_obj.cast<ovr::scene::Camera>();
              renderer->init(0, nullptr, scene, camera);

              if (!sample_per_pixel_obj.is_none())
                      renderer->set_sample_per_pixel(sample_per_pixel_obj.cast<int>());
              if (!path_tracing_obj.is_none())
                      renderer->set_path_tracing(path_tracing_obj.cast<bool>());
              if (!frame_accumulation_obj.is_none())
                      renderer->set_frame_accumulation(frame_accumulation_obj.cast<bool>());
              if (!volume_sampling_rate_obj.is_none())
                      renderer->set_volume_sampling_rate(volume_sampling_rate_obj.cast<float>());
              if (!volume_density_scale_obj.is_none())
                      renderer->set_volume_density_scale(volume_density_scale_obj.cast<float>());

              return render_to_image(*renderer, py::cast(fbsize), scrub, clip);
      },
      py::arg("backend"),
      py::arg("scene"),
      py::arg("fbsize"),
      py::kw_only(),
      py::arg("camera") = py::none(),
      py::arg("sample_per_pixel") = py::none(),
      py::arg("path_tracing") = py::none(),
      py::arg("frame_accumulation") = py::none(),
      py::arg("volume_sampling_rate") = py::none(),
      py::arg("volume_density_scale") = py::none(),
      py::arg("scrub") = true,
      py::arg("clip") = false,
      R"pbdoc(
Create a renderer for a scene and return one rendered float32 RGBA image.

This is a one-shot convenience wrapper around ``create_renderer()``,
``set_fbsize()``, ``init()``, optional render-parameter setters, and
``render_to_image()``.

Parameters
----------
backend:
    Renderer backend name, for example ``"optix7"`` or ``"ospray"``.
scene:
    Scene object returned by ``ovrpy.create_scene(...)``.
fbsize:
    ``ovrpy.vec2i`` framebuffer size. The returned array has shape
    ``(fbsize.y, fbsize.x, 4)``.
camera:
    Optional camera. Defaults to ``scene.camera``.
sample_per_pixel:
    Optional value passed to ``set_sample_per_pixel``.
path_tracing:
    Optional value passed to ``set_path_tracing``.
frame_accumulation:
    Optional value passed to ``set_frame_accumulation``.
volume_sampling_rate:
    Optional value passed to ``set_volume_sampling_rate``.
volume_density_scale:
    Optional value passed to ``set_volume_density_scale``.
scrub:
    If true, replace non-finite values with finite display-friendly values.
clip:
    If true, clamp values to ``[0.0, 1.0]``.

Returns
-------
numpy.ndarray
    A copied float32 RGBA array with shape ``(fbsize.y, fbsize.x, 4)``.
)pbdoc");

def_named_method(create_renderer, "create_renderer");

}
