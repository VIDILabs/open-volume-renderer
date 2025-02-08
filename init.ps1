if (Test-Path build) {
    Remove-Item build\* -Recurse -Force
} else {
    New-Item -Path 'build' -ItemType Directory
}
cmake `
    -DCUDAToolkit_ROOT:STRING=C:\Libraries\CUDA\12.4 `
    -DOptiX_INSTALL_DIR:STRING=C:\Libraries\OptiX\7.3.0 `
    -Dospray_DIR:STRING=C:\Libraries\OSPRay\2.11.0\lib\cmake\ospray-2.11.0 `
    -DTBB_DIR:STRING=C:\Libraries\OneAPITBB\2021.13.1.13\tbb\2021.13\lib\cmake\tbb `
    -DOVR_BUILD_MODULE_FOVOLNET:BOOL=FALSE `
    -DOVR_BUILD_MODULE_STEREO:BOOL=TRUE `
    -DOVR_BUILD_MODULE_VULKAN:BOOL=FALSE `
    -DOVR_BUILD_DEVICE_OPTIX7:BOOL=TRUE `
    -DOVR_BUILD_DEVICE_OSPRAY:BOOL=TRUE `
    -S . -B build -G "Visual Studio 17 2022" -T host=x64 -A x64
