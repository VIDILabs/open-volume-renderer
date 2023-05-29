#!/bin/bash

FILES=$(find * -type f -iname "*.txt" ! -iname "CMakeLists.txt")
ROOT=${PWD}/../
BASE=colormap

# Generate one source for each map
for f in ${FILES}; do
    DATA=$(sed -e 's/)/},/g' -e 's/(/{/g' ${f}) # replace '(' and ')'
    NAME=${f%.*}                                # sequence/hot
    NVAR=$(echo ${NAME} | sed -e 's/\//\_/g')   # sequence/hot -> sequence_hot
    cat > ${PWD}/${NAME}.cpp << EOF
//
// Automatically generated file, do not modify.
//
// ======================================================================== //
// Copyright 2019-2020 Qi Wu                                                //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //
//
// clang-format off
//
#include <vector>
#include <array>
#include <string>
namespace colormap {
struct color_t { float r, g, b, a; };
extern const std::vector<color_t> data_${NVAR};
}
const std::vector<colormap::color_t> colormap::data_${NVAR} = /* NOLINT(cert-err58-cpp) */
{
${DATA}
};
EOF
done

# Generate the header file first
DATA=$(for f in ${FILES}; do NAME=${f%.*}; echo "// ${NAME}"; done)
cat > ${BASE}.h << EOF
//
// Automatically generated file, do not modify.
//
// ======================================================================== //
// Copyright 2019-2020 Qi Wu                                                //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //
//
// clang-format off
//
// available colormap keys:
//
${DATA}

#ifndef TFN_COLORMAP_H
#define TFN_COLORMAP_H

#include <vector>
#include <array>
#include <string>
#include <unordered_map>
namespace colormap {
struct color_t { float r, g, b, a; };
extern const std::unordered_map<std::string, const std::vector<color_t>*> data;
extern const std::vector<std::string> name;
}

#endif // TFN_COLORMAP_H
EOF

# Now the Source File
DATA1=$(for f in ${FILES};                                     \
	do NVAR=$(echo ${f%.*} | sed -e 's/\//\_/g');              \
	   echo "extern const std::vector<color_t> data_${NVAR};"; \
	done)
DATA2=$(for f in ${FILES};                               \
	do NAME=${f%.*};                                     \
	   NVAR=$(echo ${NAME} | sed -e 's/\//\_/g');        \
	   echo "{ \"${NAME}\", &colormap::data_${NVAR} },"; \
	done)
DATA3=$(for f in ${FILES}; do echo "\"${f%.*}\","; done)
cat > ${BASE}.cpp << EOF
//
// Automatically generated file, do not modify.
//
// ======================================================================== //
// Copyright 2019-2020 Qi Wu                                                //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //
//
// clang-format off
//
#include <vector>
#include <string>
#include <unordered_map>
namespace colormap {
struct color_t { float r, g, b, a; };
extern const std::unordered_map<std::string, const std::vector<color_t>*> data;
extern const std::vector<std::string> name;
${DATA1}
}
// definitions
const std::unordered_map<std::string, const std::vector<colormap::color_t>*>
  colormap::data = /* NOLINT(cert-err58-cpp) */
{
${DATA2}
};
const std::vector<std::string> colormap::name = /* NOLINT(cert-err58-cpp) */
{
${DATA3}
};
EOF

# add cmake files
DATA=$(for f in ${FILES}; do echo "\${CMAKE_CURRENT_LIST_DIR}/${f%.*}.cpp"; done)
cat > CMakeLists.txt << EOF
#
# Automatically generated file, do not modify.
#
# ======================================================================== #
# Copyright 2019-2020 Qi Wu                                                #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #
set(embedded_colormap 
\${CMAKE_CURRENT_LIST_DIR}/${BASE}.cpp
\${CMAKE_CURRENT_LIST_DIR}/${BASE}.h
${DATA}
PARENT_SCOPE)
EOF
