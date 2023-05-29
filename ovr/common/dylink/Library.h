//===========================================================================//
//                                                                           //
// LibViDi3D                                                                 //
// Copyright(c) 2018 Qi Wu (Wilson), Min Shih                                //
// University of California, Davis                                           //
// MIT Licensed                                                              //
//                                                                           //
//===========================================================================//
/*
 * The following code is modified based on codes from the ospray project
 */
// ========================================================================= //
// Copyright 2009-2018 Intel Corporation                                     //
//                                                                           //
// Licensed under the Apache License, Version 2.0 (the "License");           //
// you may not use this file except in compliance with the License.          //
// You may obtain a copy of the License at                                   //
//                                                                           //
//     http://www.apache.org/licenses/LICENSE-2.0                            //
//                                                                           //
// Unless required by applicable law or agreed to in writing, software       //
// distributed under the License is distributed on an "AS IS" BASIS,         //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  //
// See the License for the specific language governing permissions and       //
// limitations under the License.                                            //
// ========================================================================= //

#pragma once

#include <map>
#include <memory>
#include <string>
#include <set>

namespace ovr {
namespace dynamic {


class UnknownLibraryError : public std::runtime_error {
public:
  UnknownLibraryError(const std::string& message)
    : runtime_error(message)
  {
  }
};

class UnknownDynamicModuleError : public std::runtime_error {
public:
  UnknownDynamicModuleError(const std::string& type, const std::string& name)
    : runtime_error(stringf("Could not find '%s' of type '%s'. "
                            "Make sure you have the correct libraries linked.", 
                            type, restore_namespace(name)))
  {
  }

protected:
  std::string restore_namespace(std::string in) {
    return ReplaceAll(in, "__", "::");
  }

  std::string ReplaceAll(std::string str, const std::string& from, const std::string& to) {
      size_t start_pos = 0;
      while((start_pos = str.find(from, start_pos)) != std::string::npos) {
          str.replace(start_pos, from.length(), to);
          start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
      }
      return str;
  }

  static const char* c_str(const std::string& s) { return s.c_str(); }

  template<typename T>
  static T c_str(T s)
  {
    return s;
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
  template<typename... Ts>
  static std::string stringf(const std::string& format, Ts... rest)
  {
    int64_t sz = snprintf(NULL, 0, format.c_str(), c_str(rest)...);
    char* bf = static_cast<char*>(malloc(sz + 1));
    snprintf(bf, sz + 1, format.c_str(), c_str(rest)...);
    std::string ret(bf);
    free(bf);
    return ret;
  }
#pragma GCC diagnostic pop
};

class LibraryRepository;

/**
 * @brief Internal API
 */
namespace details {
/**
 * This is a class for loading dynamic libraries from the system.
 * Tested systems: Linux
 */
class Library {
private:
  friend class ovr::dynamic::LibraryRepository;

public:
  /* opens a shared library */
  explicit Library(const std::string& name);
  ~Library();
  /* returns address of a symbol from the library */
  void*
  getSymbol(const std::string& sym) const;

private:
  explicit Library(void* const);
  std::string libraryName;
  void*       lib             = nullptr;
  bool        freeLibOnDelete = true;
};
} // namespace details

/**
 * This is the class for managing all loaded libraries. One can access the
 * global instance of this class using LibraryRepository::GetInstance()
 */
class LibraryRepository {
public:
  /* obtain the global instance */
  static LibraryRepository*
  GetInstance();

  /* delete the global instance */
  static void
  CleanupInstance();

  ~LibraryRepository();

  /* add a library to the repo */
  void
  add(const std::string& name, bool silent = false);

  /* remove a library to the repo */
  void
  remove(const std::string& name);

  /* returns address of a symbol from any library in the repo */
  void*
  getSymbol(const std::string&) const;

  /* check if the library exists */
  bool
  libraryExists(const std::string& name) const;

  /* load default stuffs */
  void
  addDefaultLibrary();

  /* add a library to the repo */
  void
  registerSymbols(const std::set<std::string>& names);

private:
  LibraryRepository() = default;

  std::map<std::string, details::Library*> repo;
  std::set<std::string> known_symbols;

  static std::unique_ptr<LibraryRepository> instance;
};

} // namespace dynamic
} // namespace vidi
