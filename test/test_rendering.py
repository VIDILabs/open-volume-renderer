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
scene.spp = 128
renderer = ovrpy.create_renderer(renderername)

print(scene.get_bounds())
scene.print()

fbsize = ovrpy.vec2i()
# fbsize.x = 640
# fbsize.y = 480

fbsize.x = 640 * 2
fbsize.y = 480 * 2

renderer.set_fbsize(fbsize)

framebufferdata = ovrpy.FrameBufferData()

renderer.init([], scene, scene.camera)
renderer.set_path_tracing(1)
renderer.commit()

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

# PYTHONPATH=/mnt/scratch/fast0/qadwu/open-volume-renderer/build/Release python ./test_rendering.py ../data/configs/scene_heatrelease_1atm.json 

# ID 0  min: 3.395984797497505e-07  max: 0.005162133198886688  mean: 0.0026227774599384157  std: 0.0025100889076609756
# ID 1  min: 5.172714594250978e-14  max: 0.04024680547316535  mean: 0.020968996386520656  std: 0.019328995509583148
# ID 2  min: -1.4963685689749903e-16  max: 0.01089632168743805  mean: 0.0007606342341715767  std: 0.0008273887988946774
# ID 3  min: -1.376375761043291e-16  max: 0.007133583180370828  mean: 0.00018962039705239922  std: 0.0007400773794882409
# ID 4  min: 0.05638380919429669  max: 0.21781442268050988  mean: 0.16884458369525057  std: 0.04753884719437635
# ID 5  min: -1.7697095091806619e-16  max: 4.748223746807794e-05  mean: 3.698302943848551e-08  std: 3.802983376961142e-07
# ID 6  min: -1.8501725237820012e-16  max: 0.000558710120740207  mean: 2.4671402299748254e-06  std: 7.70009571970101e-06
# ID 7  min: -1.851214538801249e-16  max: 0.0038874615672837482  mean: 9.574803771657988e-05  std: 0.00011189018498885032
# ID 8  min: 1.5788134777714386e-14  max: 0.0005892863073116689  mean: 1.251361984700772e-05  std: 5.029648690939495e-05
# ID 9  min: -1.3759738532915957e-16  max: 0.14083038404280296  mean: 0.051811752926426466  std: 0.052839303849693625
# ID 10  min: -1.3754414150299344e-16  max: 0.00029378884886239456  mean: 5.806889436837171e-06  std: 2.5446423703962534e-05
# ID 11  min: -1.3760909048587693e-16  max: 0.0005899682246169599  mean: 2.219420148171238e-05  std: 4.760504209931545e-05
# ID 12  min: -1.3757221731719059e-16  max: 0.00022019450321203823  mean: 3.526318911548266e-06  std: 1.5257033140314808e-05
# ID 13  min: -3.189961954556927e-16  max: 1.7550976128907115e-05  mean: 1.0436495249041686e-08  std: 1.4177237766938335e-07
# ID 14  min: -1.8277755972416456e-16  max: 4.5896800226178627e-07  mean: 6.571518021984061e-10  std: 6.321440676804638e-09
# ID 15  min: -1.676044000290523e-16  max: 0.0011315294678591482  mean: 1.187014589242582e-05  std: 6.18735017703852e-05
# ID 16  min: -2.044410517409339e-16  max: 8.936931444506212e-05  mean: 3.441285109276434e-07  std: 2.3355736314013143e-06
# ID 17  min: -1.376375761043291e-16  max: 0.00010579453298142547  mean: 1.6601972859308458e-06  std: 7.770587929797547e-06
# ID 18  min: 0.7358604111069482  max: 0.785754736767009  mean: 0.7546454561446435  std: 0.015509483304074917
# (T) ID 19  min: 6.225112248101931  max: 17.789794854597424  mean: 10.300052507021645  std: 4.128385620753477
# (P) ID 20  min: 14.256400474103675  max: 14.310383928408385  mean: 14.299600041183641  std: 0.0011157289264419295
# (X) ID 21  min: -0.09851698687593262  max: 0.11783292482314509  mean: 0.0006960929530651931  std: 0.061684395738840596
# (Y) ID 22  min: -0.09763172342216822  max: 0.09070377851938949  mean: 0.00154917381970876  std: 0.006969626766545699
# (Z) ID 23  min: -0.11063593755138236  max: 0.09582531587139316  mean: -4.865476337039316e-06  std: 0.005805681901214857

# fields = np.memmap("ptj_nr_864x480x640.5.8640E-04.field.mpi", mode='r', dtype=np.float64)
# fields = fields.reshape((-1, 640, 480, 864))
# for i in range(len(fields)):
#     field = fields[i]
#     print(f"ID {i}  min: {field.min()}  max: {field.max()}  mean: {field.mean()}  std: {field.std()}")
#     field.tofile(f"ptj_nr_{i}_864x480x640.5.8640E-04.field.raw")
