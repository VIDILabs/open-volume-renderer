mkdir build-release
cd build-release
cmake .. -G "Visual Studio 16 2019" -T host=x64 -A x64 ^
    -DCMAKE_BUILD_TYPE=Release -DOVR_BUILD_DEVICE_VIDI3D=ON ^
    -DCMAKE_PREFIX_PATH=C:\Users\wilson\Work\softwares\win\libtorch\libtorch-1.10.1-cu113\libtorch ^
    -Dospray_DIR=C:\Users\wilson\Work\softwares\intel-environment\ospray-2.5.0.x86_64.windows\lib\cmake\ospray-2.5.0 ^
    -DTBB_DIR=C:\Users\wilson\Work\softwares\intel-environment\oneapi-tbb-2021.1.1-win\oneapi-tbb-2021.1.1\lib\cmake\tbb
cmake --build . --config Release
