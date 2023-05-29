//. ======================================================================== //
//. Copyright 2021-2022 David Bauer                                          //
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

#pragma once
#ifndef OVR_COMMON_SCREENSHOT_H
#define OVR_COMMON_SCREENSHOT_H

#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>

#include "math.h"

using namespace ovr::math;

namespace vidi {

struct Screenshot {

    // template<typename T>
    static void save(vec3f* image, vec2i size) {
        std::ofstream stream;
        std::string filename;
        auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

        filename = "ovr_screenshot_" + std::to_string(timestamp) + ".ppm";
        stream.open(filename);

        if (!stream.good()) {
            throw std::runtime_error("Unable to create screenshot file " + filename);
        }

        // Write PPM header
        stream << "P3\n" << size.x << " " << size.y << "\n" << "255\n";

        for (int v = size.y-1; v >= 0; v--) {
            for (int u = 0; u < size.x; u++) {
                int i = v*size.x + u;
                stream << f2s(image[i].x) << " " << f2s(image[i].y) << " " << f2s(image[i].z) << "\n";
            }
        }

        stream.flush();
        stream.close();
    }

    static void save(vec4f* image, vec2i size) {
        vec3f* img = new vec3f [size.x * size.y];

        for (int i = 0; i < size.x*size.y; i++) {
            img[i] = vec3f( image[i].x * image[i].w,
                            image[i].y * image[i].w,
                            image[i].z * image[i].w);
        }

        Screenshot::save(img, size);
        delete[] img;
    }

    private:
    static std::string f2s(float val) {
        return std::to_string((int)(clamp(val, 0.0f, 1.0f)*255));
    }
};

}

#endif