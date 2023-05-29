#!/bin/bash

mkdir -p build
cd build

# CUDA without OptiX
# unset OptiX_INSTALL_DIR
# ~/Software/cmake-3.22.1-linux-x86_64/bin/cmake /home/qadwu/Work/ovr \
#       -DOVR_BUILD_DEVICE_OPTIX7=OFF \
#       -DOVR_BUILD_DEVICE_OSPRAY=ON  \
#       -Dospray_DIR=/home/qadwu/Software/ospray-2.6.0.x86_64.linux/lib/cmake/ospray-2.6.0

# without CUDA
unset OptiX_INSTALL_DIR
~/Software/cmake-3.22.1-linux-x86_64/bin/cmake /home/qadwu/Work/ovr \
      -DOVR_BUILD_DEVICE_OPTIX7=OFF \
      -DOVR_BUILD_DEVICE_OSPRAY=ON  \
      -DOVR_BUILD_DEVICE_NNCACHE=OFF \
      -Dospray_DIR=/home/qadwu/Software/ospray-2.8.0.x86_64.linux/lib/cmake/ospray-2.8.0
