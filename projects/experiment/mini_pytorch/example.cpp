#include <torch/torch.h>
#include <torch/script.h>
#include <torch/types.h>
#include <iostream>

int main()
{
  if (torch::cuda::is_available())
    std::cout << "PyTorch CUDA is available with " << torch::cuda::device_count() << " device." << std::endl;
  else
    std::cout << "PyTorch CUDA is not available." << std::endl;

  torch::jit::Module module;
  try
  {
    module = torch::jit::load("temporal_wnet_small_inputema_traced.pt", torch::kCPU);
  }
  catch (const torch::Error &e)
  {
    std::cerr << "error loading the model" << std::endl
              << e.what() << std::endl;
    return -1;
  }

  const auto device = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
  const auto dtype = torch::kHalf;

  module.to(device, dtype);
  module.eval();
  std::cout << "done initializing model." << std::endl;

  // Define input shape
  const int batches = 1, channels = 6, height = 200, width = 200;
  auto opts = torch::TensorOptions().dtype(dtype).device(device); /* one way to specify tensor*/

  // Create a vector of inputs
  torch::Tensor recurrent = torch::zeros({batches, channels, height, width}, opts);
  torch::Tensor input = torch::rand({batches, channels, height, width}, opts);

  // Do inferencing
  auto start = std::chrono::high_resolution_clock::now();

  torch::Tensor output;
  std::cout << "done initializing inputs." << std::endl;
  try
  {
    auto _output = module.forward({input, recurrent}).toTuple();
    output = (*_output).elements()[0].toTensor().index(/* only access the first image in the batch */ {0});
    recurrent = (*_output).elements()[1].toTensor();
  }
  catch (...)
  {
    std::cerr << "error inferencing the model" << std::endl;
    return -1;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "done inferencing within " << duration.count() << "ms." << std::endl;

  torch::Tensor image = output.permute({2, 0, 1});
  std::cout << image.options() << std::endl;

  return 0;
}
