import unittest
import numpy as np
import os
import sys
sys.path.append(os.path.join(sys.path[0], "../build"))

import ovrpy


def has_device(name):
    devicelibname = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            f"../build/libdevice_{name}.a")
    return os.path.exists(devicelibname)


class CreationFunctionTestCase(unittest.TestCase):

    def test_create_scene(self):
        filename = os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                "test_scene.json")

        scene = ovrpy.create_scene(filename)
        self.assertIsNotNone(scene)

    def test_create_renderer_ospray(self):
        if not has_device("ospray"):
            self.skipTest("Device 'ospray' not available")
        renderer = ovrpy.create_renderer("ospray")
        self.assertIsNotNone(renderer)

    def test_create_renderer_optix7(self):
        if not has_device("optix7"):
            self.skipTest("Device 'optix7' not available")
        renderer = ovrpy.create_renderer("optix7")
        self.assertIsNotNone(renderer)


class RendererTestCase(unittest.TestCase):

    def setUp(self):
        filename = os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                "test_scene.json")
        self.scene = ovrpy.create_scene(filename)

        renderername = "optix7" if has_device("optix7") else "ospray"
        self.renderer = ovrpy.create_renderer(renderername)

        self.fbsize = ovrpy.vec2i()
        self.fbsize.x = 640
        self.fbsize.y = 480
        self.renderer.set_fbsize(self.fbsize)

        self.args = []

    def test_init(self):
        try:
            self.renderer.init(self.args, self.scene, self.scene.camera)
        except Exception:
            self.fail("Renderer init failed")

    def test_swap(self):
        try:
            self.renderer.swap()
        except Exception:
            self.fail("Renderer swap failed")

    def test_commit(self):
        try:
            self.renderer.init(self.args, self.scene, self.scene.camera)
            self.renderer.commit()
        except Exception:
            self.fail("Renderer commit failed")

    def test_render(self):
        try:
            self.renderer.init(self.args, self.scene, self.scene.camera)
            self.renderer.commit()
            self.renderer.render()
        except Exception:
            self.fail("Renderer render failed")

    def test_mapframe(self):
        try:
            framebufferdata = ovrpy.FrameBufferData()

            # Uninitialized framebufferdata should be invalid
            self.assertRaises(RuntimeError, framebufferdata.rgba)

            self.renderer.init(self.args, self.scene, self.scene.camera)
            self.renderer.commit()
            self.renderer.mapframe(framebufferdata)

            # Mapped framebufferdata should match the size of the renderer's framebuffer size
            pixeldata = framebufferdata.rgba()
            self.assertEqual(len(pixeldata),
                             self.fbsize.x * self.fbsize.y * 4,
                             "Mapped framebuffer does not match renderer framebuffer size")
            # Without having called render(), the framebuffer should be completely empty
            self.assertTrue((pixeldata == np.zeros(len(pixeldata), dtype=np.float32)).all(),
                            "Mapped frame is not empty")

        except Exception:
            self.fail("Renderer mapframe failed")


if __name__ == "__main__":
    unittest.main()
