#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# export DISPLAY=:2
# export LD_LIBRARY_PATH=/home/weishen/USD/lib:/home/qadwu/Software/Intel/ospray-2.5.0.x86_64.linux/lib:/home/qadwu/Software/ospray-2.8.0.x86_64.linux/lib:${LD_LIBRARY_PATH}

export LD_LIBRARY_PATH=\
/home/qadwu/Software/libtorch-shared-with-deps-1.12.1-cu116/libtorch/lib:\
/home/davbauer/software/torch_tensorrt/lib:\
/home/davbauer/software/TensorRT-8.2.3.0/lib:\
/home/qadwu/Work/ovr/build/Release:${LD_LIBRARY_PATH}:\
/home/qadwu/Work/ospray/build/Debug/install/lib:/home/qadwu/Work/ospray/deps/lib:/home/qadwu/Work/ospray/deps/lib/intel64/gcc4.8:\
/home/weishen/USD/lib:

vglrun $@
