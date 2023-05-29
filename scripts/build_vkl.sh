#!/bin/bash

mkdir -p build
cd build
cmake -DOVR_BUILD_OPTIX7=OFF \
-DOVR_BUILD_OSPRAY=OFF \
-DOVR_BUILD_TORCH=OFF \
-DOVR_BUILD_SCENE_USD=OFF \
-DOVR_BUILD_DEVICE_VIDI3D=ON \
-DOVR_BUILD_OPENVKL=ON \
-Drkcommon_DIR="/home/davbauer/repos/openvkl/build/install/lib/cmake/rkcommon-1.10.0/" \
-Dopenvkl_DIR="/home/davbauer/repos/openvkl/build/install/lib/cmake/openvkl-1.3.1/" \
..
      
cmake --build . --config Debug -j 16
