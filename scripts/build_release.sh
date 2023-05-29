#!/bin/bash


mkdir -p build
cd build
~qadwu/Software/cmake-3.22.1-linux-x86_64/bin/cmake $1 -Dospray_DIR=/home/qadwu/Software/ospray-2.6.0.x86_64.linux/lib/cmake/ospray-2.6.0 -DOVR_BUILD_DEVICE_OSPRAY=ON
~qadwu/Software/cmake-3.22.1-linux-x86_64/bin/cmake --build . --config Release --parallel 32
