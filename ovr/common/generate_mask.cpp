//. ======================================================================== //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the Apache License, Version 2.0 (the "License");          //
//. you may not use this file except in compliance with the License.         //
//. You may obtain a copy of the License at                                  //
//.                                                                          //
//.     http://www.apache.org/licenses/LICENSE-2.0                           //
//.                                                                          //
//. Unless required by applicable law or agreed to in writing, software      //
//. distributed under the License is distributed on an "AS IS" BASIS,        //
//. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
//. See the License for the specific language governing permissions and      //
//. limitations under the License.                                           //
//. ======================================================================== //

#include "generate_mask.h"

namespace ovr {

#ifndef OVR_BUILD_CUDA_DEVICES
int64_t
generate_sparse_sampling_mask_h(int32_t* h_output, 
                                int frame_index,
                                const vec2i& fbsize,
                                const vec2f& focus_center,
                                float focus_scale,
                                float base_noise)
{
  throw std::runtime_error("unimplemented");
}
#endif

} // namespace ovr
