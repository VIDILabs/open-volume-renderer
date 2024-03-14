#include "imageop.h"

#ifdef OVR_BUILD_OPTIX7
#include "optix7_denoiser.h"
#endif

// #ifdef OVR_BUILD_OSPRAY
// #include "ospray_denoiser.h"
// #endif

namespace ovr {

struct ImageNoOp : ImageOp {
private:
  std::shared_ptr<CrossDeviceBuffer> cache;

public:
  void initialize(int ac, const char** av) override {}
  void resize(int width, int height) override  {}
  void process(std::shared_ptr<CrossDeviceBuffer>& input) override  { cache = input; }
  void map(std::shared_ptr<CrossDeviceBuffer>& output) const override  { output = cache; }
};

}

std::shared_ptr<ovr::ImageOp>
create_imageop(std::string name)
{
  if (name == "denoiser") {
#ifdef OVR_BUILD_OPTIX7
    return std::make_shared<ovr::optix7::Optix7Denoiser>();
#endif
  }

  std::cerr << "[warning] unknown imageop name: " << name << std::endl;
  return std::make_shared<ovr::ImageNoOp>();
}
