#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <cstdlib>

#include "torch/script.h"
#include <torch/torch.h>
#include <torch/types.h>
#include "torch_tensorrt/torch_tensorrt.h"


int main(int ac, char** av) {
    if (ac < 7) {
        std::cout << "Usage " << av[0] << " SOURCE DEST B C H W" << std::endl;
        return -1;
    }

    // Parse arguments
    std::string in_name = av[1];
    std::string out_name = av[2];
    int B = atoi(av[3]);
    int C = atoi(av[4]);
    int H = atoi(av[5]);
    int W = atoi(av[6]);

    // Load TorchScript module
    std::cout << "Loading module " << in_name << std::endl;
    torch::jit::Module module;
    try {
        module = torch::jit::load(in_name);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    // Find GPU device if possible
    torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
    std::cout << "Using device " << device << std::endl;

    // Configure module
    module.to(device);
    module.eval();

    // Trace the module
    std::cout << "Tracing model with input sizes [" << B << " " << C << " " << H << " " << W << "]" << std::endl;
    auto in = torch_tensorrt::Input((std::vector<int64_t>){B, C, H, W});
    auto spec = torch_tensorrt::torchscript::CompileSpec((std::vector<torch_tensorrt::Input>){in});
    auto trt_module = torch_tensorrt::torchscript::compile(module, spec);

    // Save TorchTensorRT optimized model
    trt_module.save(out_name);

    std::cout << "done" << std::endl;
}