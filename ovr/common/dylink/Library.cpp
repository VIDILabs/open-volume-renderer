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

#include "Library.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/times.h>
#endif
#include <iostream>

namespace ovr {
namespace dynamic {
namespace details {

Library::Library(const std::string& name) : libraryName(name)
{
  std::string file = name;
  std::string errorMsg;
#ifdef _WIN32
  std::string fullName = file + ".dll";
  lib                  = LoadLibrary(fullName.c_str());
  if (lib == nullptr) {
    DWORD  err = GetLastError();
    LPTSTR lpMsgBuf;
    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS,
                  NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                  (LPTSTR)&lpMsgBuf, 0, NULL);

    errorMsg = lpMsgBuf;

    LocalFree(lpMsgBuf);
  }
#else
#if defined(__MACOSX__) || defined(__APPLE__)
  std::string fullName = "lib" + file + ".dylib";
#else
  std::string fullName = "lib" + file + ".so";
#endif
  lib                  = dlopen(fullName.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) {
    errorMsg = dlerror(); // remember original error
    lib      = dlopen(fullName.c_str(), RTLD_NOW | RTLD_LOCAL);
  }
#endif
  if (lib == nullptr) {
#ifdef _WIN32
    throw UnknownLibraryError("cannot open shared library " + fullName);
#else
    throw UnknownLibraryError(dlerror());
#endif
  }
  // do NOT try to find the library in another location
  // if you want that use LD_LIBRARY_PATH or equivalents
}

Library::~Library()
{
  if (freeLibOnDelete) {
#ifdef _WIN32
    FreeLibrary((HMODULE)lib);
#else
    dlclose(lib);
#endif
    // vidi::log() << "[library] closing " << libraryName << std::endl;
  }
}

Library::Library(void* const _lib)
  : libraryName("<pre-loaded>"), lib(_lib), freeLibOnDelete(false)
{
}

void*
Library::getSymbol(const std::string& sym) const
{
#ifdef _WIN32
  return GetProcAddress((HMODULE)lib, sym.c_str());
#else
  return dlsym(lib, sym.c_str());
#endif
}
} // namespace details

std::unique_ptr<LibraryRepository> LibraryRepository::instance =
  std::unique_ptr<LibraryRepository>(new LibraryRepository);

LibraryRepository*
LibraryRepository::GetInstance()
{
  if (instance == nullptr) {
    instance = std::unique_ptr<LibraryRepository>(new LibraryRepository);
  }
  return instance.get();
}

void
LibraryRepository::CleanupInstance()
{
  LibraryRepository::instance.reset();
}

LibraryRepository::~LibraryRepository()
{
  for (auto& l : repo) {
    delete l.second;
  }
}

void
LibraryRepository::add(const std::string& name, bool silent)
{
  try {
    if (libraryExists(name)) {
      return; // lib already loaded.
    }
    repo[name] = new details::Library(name);
  }
  catch (const UnknownLibraryError& e) {
    if (silent) {
      std::cerr << e.what() << std::endl;
    }
    else {
      throw e;
    }
  }
}

void
LibraryRepository::remove(const std::string& name)
{
  if (!libraryExists(name)) {
    return; // lib does not exist.
  }
  delete repo[name];
  repo.erase(name);
}

void*
LibraryRepository::getSymbol(const std::string& name) const
{
  void* sym = nullptr;
  for (auto lib = repo.cbegin(); sym == nullptr && lib != repo.end(); ++lib) {
    sym = lib->second->getSymbol(name);
  }
  return sym;
}

void
LibraryRepository::addDefaultLibrary()
{
  // already populate the repo with "virtual" libs, representing the
  // default core lib
#ifndef _WIN32
  repo["default"] = new details::Library(RTLD_DEFAULT);

  // to also open the current library / executable
  // not sure how to do the same on windows
  repo["self"] = new details::Library(dlopen(NULL, RTLD_LAZY));
#endif
}

void
LibraryRepository::registerSymbols(const std::set<std::string>& names)
{
  known_symbols.insert(names.begin(), names.end());
}

bool
LibraryRepository::libraryExists(const std::string& name) const
{
  return repo.find(name) != repo.end();
}

} // namespace dynamic
} // namespace vidi
