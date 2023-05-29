#pragma once

#include "diva_constants.h"

#include <gdt/math/mat.h>
#include <gdt/math/vec.h>

#include <3rdparty/json.hpp>
#include <type_traits>

namespace diva {
using namespace gdt;
using json = nlohmann::json;

// #define DIVA_MEMBER_FUNCTION_CHECKER(func, ...)                                                                \
//   namespace detail {                                                                                           \
//   template<class T, class R, class... Arg>                                                                     \
//   using check_t_##func = std::is_same<decltype(std::declval<T>().##func##(std::declval<Arg>()...)), R>;        \
//   template<class T, class R, class... Arg, std::enable_if_t<check_t_##func<T, R, Arg...>::value, bool> = true> \
//   static auto check_##func(int) -> std::true_type;                                                             \
//   template<class T, class R, class... Arg> /* default option */                                                \
//   static auto check_##func(long) -> std::false_type;                                                           \
//   }                                                                                                            \
//   template<class T> struct has_##func : decltype(detail::check_##func<T, __VA_ARGS__>(0)) {                    \
//   };

// DIVA_MEMBER_FUNCTION_CHECKER(to_json, json)
// DIVA_MEMBER_FUNCTION_CHECKER(from_json, void, const json&)

} // namespace diva

NLOHMANN_JSON_SERIALIZE_ENUM(DivaType,
                             {
                               { DivaType::TYPE_UINT8, "TYPE_UINT8" },
                               { DivaType::TYPE_UINT16, "TYPE_UINT16" },
                               { DivaType::TYPE_UINT32, "TYPE_UINT32" },
                               { DivaType::TYPE_INT8, "TYPE_INT8" },
                               { DivaType::TYPE_INT16, "TYPE_INT16" },
                               { DivaType::TYPE_INT32, "TYPE_INT32" },
                               { DivaType::TYPE_FLOAT, "TYPE_FLOAT" },
                               { DivaType::TYPE_DOUBLE, "TYPE_DOUBLE" },
                             });

namespace diva {

#define define_vector_serialization(T)                  \
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(vec2##T, x, y);    \
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(vec3##T, x, y, z); \
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(vec4##T, x, y, z, w);

define_vector_serialization(i);
define_vector_serialization(f);
define_vector_serialization(d);

// inline json
// to_json(const void* v)
// {
//   uint64_t code = (uint64_t)v;
//   return json("memory://" + std::to_string(code));
// }

// template<>
// inline void*
// from_json(const json& j)
// {
//   auto str = j.get<std::string>();
//   auto i = str.find_first_of(":");
//   auto type = str.substr(0, i);
//   assert(type == "memory");
//   auto data = str.substr(i + 3, str.length() - i - 2);
//   return (void*)std::stoull(data);
// }

inline std::string
to_base64(const char* byteArray, size_t size, bool urlEncoding = false)
{
  const char alphabetBase64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const char alphabetBase64Url[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
  const char* const alphabet = (urlEncoding ? alphabetBase64Url : alphabetBase64);
  const char padchar = '=';
  int padlen = 0;
  std::unique_ptr<char[]> tmp(new char[size * 4 / 3 + 3 + 1]);
  size_t i = 0;
  char* out = tmp.get();
  while (i < size) {
    // encode 3 bytes at a time
    int chunk = 0;
    chunk |= int((unsigned char)(byteArray[i++])) << 16;
    if (i == size) {
      padlen = 2;
    }
    else {
      chunk |= int((unsigned char)(byteArray[i++])) << 8;
      if (i == size) {
        padlen = 1;
      }
      else {
        chunk |= int((unsigned char)(byteArray[i++]));
      }
    }
    int j0 = (chunk & 0x00fc0000) >> 18;
    int j1 = (chunk & 0x0003f000) >> 12;
    int j2 = (chunk & 0x00000fc0) >> 6;
    int j3 = (chunk & 0x0000003f);
    *out++ = alphabet[j0];
    *out++ = alphabet[j1];
    *out++ = (padlen > 1 ? padchar : alphabet[j2]);
    *out++ = (padlen > 0 ? padchar : alphabet[j3]);
  }
  *out = '\0';
  return std::string(tmp.get());
}

inline void
from_base64(const std::string& base64, char* byteArray, bool urlEncoding = false)
{
  unsigned int buf = 0;
  int nbits = 0;
  char* tmp = byteArray;
  int offset = 0;
  for (int i = 0; i < base64.size(); ++i) {
    int ch = base64[i];
    int d;
    if (ch >= 'A' && ch <= 'Z') {
      d = ch - 'A';
    }
    else if (ch >= 'a' && ch <= 'z') {
      d = ch - 'a' + 26;
    }
    else if (ch >= '0' && ch <= '9') {
      d = ch - '0' + 52;
    }
    else if (ch == '+' && !urlEncoding) {
      d = 62;
    }
    else if (ch == '-' && urlEncoding) {
      d = 62;
    }
    else if (ch == '/' && !urlEncoding) {
      d = 63;
    }
    else if (ch == '_' && urlEncoding) {
      d = 63;
    }
    else {
      d = -1;
    }
    if (d != -1) {
      buf = (buf << 6) | d;
      nbits += 6;
      if (nbits >= 8) {
        nbits -= 8;
        tmp[offset++] = buf >> nbits;
        buf &= (1 << nbits) - 1;
      }
    }
  }
}

inline size_t
size_base64(const std::string& base64)
{
  size_t size = base64.size() * 3 / 4;
  if (base64.size() >= 1 && base64[base64.size() - 1] == '=')
    size--;
  if (base64.size() >= 2 && base64[base64.size() - 2] == '=')
    size--;
  return size;
}

// ======================================================================== //
//                                                                          //
// ======================================================================== //
struct SerializableObject {
  virtual ~SerializableObject() {}
};

} // namespace diva
