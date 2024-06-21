import os, sys

import numpy as np

import ovrpy

# renderername = "optix7"
renderername = "ospray"
# renderername = "vulkan"
# filename = os.path.join(
#         os.path.abspath(os.path.dirname(__file__)),
#         "test_scene.json")
filename = sys.argv[1]

scene = ovrpy.create_scene(filename)
renderer = ovrpy.create_renderer(renderername)

fbsize = ovrpy.vec2i()
# fbsize.x = 640
# fbsize.y = 480

fbsize.x = 640*4
fbsize.y = 480*4

renderer.set_fbsize(fbsize)

framebufferdata = ovrpy.FrameBufferData()

renderer.init([], scene, scene.camera)
renderer.commit()

for i in range(16):
    renderer.render()
    renderer.swap()

renderer.mapframe(framebufferdata)

pixeldata = framebufferdata.rgba()
pixelstats = framebufferdata.stats()
# print("Pixel data:", pixeldata)
# print("Pixel stats:", pixelstats)

pixeldata = (np.clip(pixeldata, 0, 1) * 255).astype(np.uint8)
pixeldata = pixeldata.reshape(fbsize.y, fbsize.x, 4)

from PIL import Image  
im = Image.fromarray(pixeldata)
im.save("test_rendering.png")
print("done saving test_rendering.png...")

# LD_PRELOAD=/mnt/scratch/ssd/qadwu/miniconda3/envs/ovrpy/lib/libstdc++.so.6 PYTHONPATH=/mnt/scratch/fast0/qadwu/open-volume-renderer/build/Release python ./test_rendering.py ../data/configs/scene_heatrelease_1atm.json 
