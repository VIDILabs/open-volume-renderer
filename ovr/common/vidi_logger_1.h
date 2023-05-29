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
#ifndef OVR_COMMON_LOGGER_H
#define OVR_COMMON_LOGGER_H

#include <fstream>
#include <string>
#include <vector>
#include <chrono>

namespace vidi {

struct Logger {
    public:
        std::string filename;
        std::ofstream stream;
    public:
        Logger() {}

        ~Logger() {
            if (stream.is_open()) {
                stream.close();
            }
        }

        void
        initialize(std::string filesuffix = "") {
            auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::system_clock::now().time_since_epoch()).count();
            if (filesuffix != "") filesuffix = "_"+filesuffix;
            filename = "ovr_log_" + std::to_string(timestamp) + filesuffix + ".csv";
            stream.open(filename);

            if (!stream.good()) {
                throw std::runtime_error("Unable to create log file " + filename);
            }
        }

        void log(std::string line) {
            stream << line << std::endl;
        }
};

struct CsvLogger : public Logger {
    public:
        char delim;
        int num_columns;

    public:
        CsvLogger() {}

        void
        initialize(std::vector<std::string> header, char delim = ',', std::string filesuffix = "") {
            Logger::initialize(filesuffix);

            this->delim = delim;
            this->num_columns = header.size();

            log_entry(header);
        }

        template<typename T>
        void
        log_entry(std::vector<T> entry) {

            for (int i = 0; i < num_columns; i++) {
                stream << entry[i];
                if (i != num_columns-1)
                    stream << delim;
            }
            stream << std::endl;
        }
};

}

#endif