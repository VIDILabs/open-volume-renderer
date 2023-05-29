// ======================================================================== //
// Copyright Qi Wu, since 2019                                              //
// Copyright SCI Institute, University of Utah, 2018                        //
// ======================================================================== //
#pragma once

#define TFN_MODULE_INTERFACE // place-holder

#include "helper.h"
#include "json.h"

#include <array>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tfn {

// ========================================================================
#define TFN_MODULE_VERSION "0.03 WIP"
inline TFN_MODULE_INTERFACE const char *GetVersion()
{
  return TFN_MODULE_VERSION;
  return TFN_MODULE_VERSION;
  return TFN_MODULE_VERSION;
};

// ========================================================================
// The magic number is 'OSTF' in ASCII
const static uint32_t MAGIC_NUMBER = 0x4f535446;
const static uint64_t CURRENT_VERSION = 1;

#ifdef TFN_MODULE_EXTERNAL_VECTOR_TYPES
/*! we assume the app already defines osp::vec types. Note those
  HAVE to be compatible with the data layouts used below.
  Note: this feature allows the application to use its own vector
  type library in the following way
  a) include your own vector library (say, ospcommon::vec3f etc, when using
     the ospcommon library)
  b) make sure the proper vec3f etc are defined in a osp:: namespace, e.g.,
     using
     namespace tfn {
       typedef ospcommon::vec3f vec3f;
     }
  c) defines OSPRAY_EXTERNAL_VECTOR_TYPES
  d) include vec.h
  ! */
#else
// clang-format off
struct vec2f { float x, y; };
struct vec2i { int x, y; };
struct vec3f { float x, y, z; };
struct vec3i { int x, y, z; };
struct vec4f { float x, y, z, w; };
struct vec4i { int x, y, z, w; };
// clang-format on
#endif

using json = nlohmann::json;
#define define_vector_serialization(T)                  \
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(vec2##T, x, y);    \
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(vec3##T, x, y, z); \
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(vec4##T, x, y, z, w);
define_vector_serialization(i);
define_vector_serialization(f);
#undef define_vector_serialization

using list1f = std::vector<float>;
using list2f = std::vector<vec2f>;
using list3f = std::vector<vec3f>;
using list4f = std::vector<vec4f>;

class TransferFunctionCore
{
  template <typename T1, typename T2>
  inline T1 mix(const T1 &x, const T1 &y, const T2 &a)
  {
    return (x + a * (y - x));
  }

 public:
  struct ColorControl
  {
    ColorControl() : position(0.0f), color(1.0f, 1.0f, 1.0f) {}
    ColorControl(float _value, float r, float g, float b) : position(_value), color(r, g, b) {}
    ColorControl(float _value, const vec3f &rgb) : position(_value), color(rgb) {}
    bool operator<(const ColorControl &other) const { return (position < other.position); }

    float& p() { return position; }
    const float& p() const  { return position; }

    float position;
    vec3f color;
  };

  static_assert(sizeof(ColorControl) == sizeof(vec4f), "ColorControl = vec4f");

  struct AlphaControl
  {
    AlphaControl() : pos(0.0) {}
    AlphaControl(const vec2f &pos_) : pos(pos_) {}

    float& p() { return pos.x; }
    const float& p() const { return pos.x; }

    vec2f pos;
  };
  
  static_assert(sizeof(AlphaControl) == sizeof(vec2f), "AlphaControl = vec2f");

  struct GaussianObject
  {
    GaussianObject();
    GaussianObject(float _mean, float _sigma, float _heightFactor, int resolution);
    float value(float x) const;
    float height() const;
    void setHeight(float h);
    void update();

    float mean;
    float sigma;
    float heightFactor;
    std::vector<float> alphaArray;
  };

 public:
  TransferFunctionCore(int resolution = 1024);
  TransferFunctionCore(const TransferFunctionCore &other);
  ~TransferFunctionCore() {}

  TransferFunctionCore &operator=(const TransferFunctionCore &) = default;  // deep copy

  void clear();

  const vec4f *data() const;
  int size() const;
  int resolution() const;

  float       *alphaArray();
  const float *alphaArray() const;
  void clearAlphaTable();

  int colorControlCount() const;
  std::vector<ColorControl>       *colorControlVector()       { return &m_colorControls; }
  const std::vector<ColorControl> *colorControlVector() const { return &m_colorControls; }
  ColorControl       &colorControl(int index);
  const ColorControl &colorControl(int index) const;
  ColorControl &addColorControl(const ColorControl &control);
  ColorControl &addColorControl(float value, float r, float g, float b);
  ColorControl &insertColorControl(float pos);
  void removeColorControl(int index);
  void clearColorControls();

  int alphaControlCount() const;
  std::vector<AlphaControl>       *alphaControlVector()       { return &m_alphaControls; }
  const std::vector<AlphaControl> *alphaControlVector() const { return &m_alphaControls; }
  AlphaControl       &alphaControl(int index);
  const AlphaControl &alphaControl(int index) const;
  AlphaControl &addAlphaControl(const AlphaControl &ctrl);
  AlphaControl &addAlphaControl(const vec2f &pos);
  void removeAlphaControl(int index);
  void clearAlphaControls();

  int gaussianObjectCount() const;
  GaussianObject       &gaussianObject(int index);
  const GaussianObject &gaussianObject(int index) const;
  GaussianObject &addGaussianObject(const GaussianObject &gaussObj);
  GaussianObject &addGaussianObject(float mean, float sigma, float heightFactor);
  void removeGaussianObject(int index);
  void clearGaussianObjects();

  void updateColorMap();

  static std::unique_ptr<TransferFunctionCore> fromRainbowMap(int resolution = 1024);

 protected:
  void updateFromAlphaControls();

 private:
  std::vector<vec4f> m_rgbaTable;
  std::vector<float> m_alphaArray;
  std::vector<ColorControl> m_colorControls;  // color control points
  std::vector<AlphaControl> m_alphaControls;
  std::vector<GaussianObject> m_gaussianObjects;
};

// ========================================================================
// ========================================================================

template <typename ScalarT>
static ScalarT scalarFromJson(const json &in)
{
  ScalarT v;
  from_json(in, v);
  return v;
}
template <typename ScalarT>
static json scalarToJson(const ScalarT &in)
{
  json v;
  to_json(v, in);
  return v;
}

static std::string toBase64(const char *byteArray, size_t size, bool urlEncoding = false)
{
  const char alphabetBase64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const char alphabetBase64Url[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
  const char *const alphabet = (urlEncoding ? alphabetBase64Url : alphabetBase64);
  const char padchar = '=';
  int padlen = 0;
  std::unique_ptr<char[]> tmp(new char[size * 4 / 3 + 3 + 1]);
  size_t i = 0;
  char *out = tmp.get();
  while (i < size) {
    // encode 3 bytes at a time
    int chunk = 0;
    chunk |= int((unsigned char)(byteArray[i++])) << 16;
    if (i == size) {
      padlen = 2;
    } else {
      chunk |= int((unsigned char)(byteArray[i++])) << 8;
      if (i == size) {
        padlen = 1;
      } else {
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

static void fromBase64(const std::string &base64, char *byteArray, bool urlEncoding = false)
{
  unsigned int buf = 0;
  int nbits = 0;
  char *tmp = byteArray;
  int offset = 0;
  for (int i = 0; i < base64.size(); ++i) {
    int ch = base64[i];
    int d;
    if (ch >= 'A' && ch <= 'Z') {
      d = ch - 'A';
    } else if (ch >= 'a' && ch <= 'z') {
      d = ch - 'a' + 26;
    } else if (ch >= '0' && ch <= '9') {
      d = ch - '0' + 52;
    } else if (ch == '+' && !urlEncoding) {
      d = 62;
    } else if (ch == '-' && urlEncoding) {
      d = 62;
    } else if (ch == '/' && !urlEncoding) {
      d = 63;
    } else if (ch == '_' && urlEncoding) {
      d = 63;
    } else {
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

static size_t sizeBase64(const std::string &base64)
{
  size_t size = base64.size() * 3 / 4;
  if (base64.size() >= 1 && base64[base64.size() - 1] == '=')
    size--;
  if (base64.size() >= 2 && base64[base64.size() - 2] == '=')
    size--;
  return size;
}

namespace {

template <typename T>
T colorFromJson(json jscolor);

template <>
inline vec3f colorFromJson(json jscolor)
{
  if (!jscolor.contains("r") || !jscolor.contains("g") || !jscolor.contains("b"))
    return vec3f();
  return vec3f(jscolor["r"].get<float>(), jscolor["g"].get<float>(), jscolor["b"].get<float>());
}

template <>
inline vec4f colorFromJson(json jscolor)
{
  if (!jscolor.contains("r") || !jscolor.contains("g") || !jscolor.contains("b") || !jscolor.contains("a"))
    return vec4f();
  return vec4f(jscolor["r"].get<float>(), jscolor["g"].get<float>(), jscolor["b"].get<float>(), jscolor["a"].get<float>());
}

}

static json colorToJson(const vec3f &color)
{
  json js;
  js["r"] = color.x;
  js["g"] = color.y;
  js["b"] = color.z;
  return js;
}

static json colorToJson(const vec4f &color)
{
  json js;
  js["r"] = color.x;
  js["g"] = color.y;
  js["b"] = color.z;
  js["a"] = color.w;
  return js;
}

static vec2f rangeFromJson(json jsrange)
{
  if (!jsrange.contains("minimum") || !jsrange.contains("maximum"))
    return vec2f(0.0, 0.0);
  return vec2f(jsrange["minimum"].get<float>(), jsrange["maximum"].get<float>());
}

// ========================================================================
// ========================================================================

inline TransferFunctionCore::GaussianObject::GaussianObject() : mean(0.5f), sigma(1.0f), heightFactor(1.0f), alphaArray(1024)
{
  update();
}

inline TransferFunctionCore::GaussianObject::GaussianObject(float _mean, float _sigma, float _heightFactor, int resolution)
    : mean(_mean), sigma(_sigma), heightFactor(_heightFactor), alphaArray(resolution)
{
  update();
}

inline float TransferFunctionCore::GaussianObject::value(float x) const
{
  float diff = x - mean;
  return heightFactor / (sigma * std::sqrt(2.0f * float(M_PI))) * std::exp(-(diff * diff) / (2.0f * sigma * sigma));
}

inline float TransferFunctionCore::GaussianObject::height() const
{
  return value(mean);
}

inline void TransferFunctionCore::GaussianObject::setHeight(float h)
{
  heightFactor = h * sigma * std::sqrt(2.0f * float(M_PI));
}

inline void TransferFunctionCore::GaussianObject::update()
{
  float invRes = 1.0f / float(alphaArray.size());
  for (size_t i = 0; i < alphaArray.size(); ++i) {
    float val = value((float(i) + 0.5f) * invRes);
    alphaArray[i] = clamp(val, 0.0f, 1.0f);
  }
}

inline TransferFunctionCore::TransferFunctionCore(int resolution)
  : m_rgbaTable(resolution), m_alphaArray(resolution, 0)
{
  assert(resolution > 0);
  // m_colorControls.push_back(ColorControl(0.0f, 0.0f, 0.0f, 0.0f));
  // m_colorControls.push_back(ColorControl(1.0f, 1.0f, 1.0f, 1.0f));
  // GaussianObject gaussObj = GaussianObject(0.5f, 0.1f, 1.0f, resolution);
  // gaussObj.setHeight(0.5f);
  // gaussObj.update();
  // for (int i = 0; i < resolution; ++i) m_alphaArray[i] = gaussObj.alphaArray[i];
  updateColorMap();
}

inline TransferFunctionCore::TransferFunctionCore(const TransferFunctionCore &other)
{
  *this = other;
}

inline void TransferFunctionCore::clear()
{
  m_rgbaTable.clear();
  m_alphaArray.clear();
  m_colorControls.clear();
  m_alphaControls.clear();
  m_gaussianObjects.clear();
}

inline const vec4f * TransferFunctionCore::data() const
{
  return m_rgbaTable.data();
}

inline int TransferFunctionCore::size() const
{
  return int(m_rgbaTable.size());
}

inline int TransferFunctionCore::resolution() const
{
  return int(m_rgbaTable.size());
}

inline float * TransferFunctionCore::alphaArray()
{
  return &m_alphaArray.front();
}

inline const float * TransferFunctionCore::alphaArray() const
{
  return &m_alphaArray.front();
}

inline void TransferFunctionCore::clearAlphaTable()
{
  for (int i = 0; i < int(m_alphaArray.size()); ++i) m_alphaArray[i] = 0.0f;
  updateColorMap();
}

inline int TransferFunctionCore::colorControlCount() const
{
  return int(m_colorControls.size());
}

inline TransferFunctionCore::ColorControl& TransferFunctionCore::colorControl(int index)
{
  return m_colorControls[index];
}

inline const TransferFunctionCore::ColorControl& TransferFunctionCore::colorControl(int index) const
{
  return m_colorControls[index];
}

inline TransferFunctionCore::ColorControl& TransferFunctionCore::addColorControl(const ColorControl &control)
{
  m_colorControls.push_back(control);
  updateColorMap();
  return m_colorControls.back();
}

inline TransferFunctionCore::ColorControl& TransferFunctionCore::addColorControl(float value, float r, float g, float b)
{
  return addColorControl(ColorControl(value, r, g, b));
}

inline TransferFunctionCore::ColorControl& TransferFunctionCore::insertColorControl(float pos)
{
  ColorControl control;
  control.position = pos;
  std::vector<ColorControl> colorControls = m_colorControls;
  std::sort(colorControls.begin(), colorControls.end());
  int controlCount = int(colorControls.size());
  // find the first color control greater than the value
  int firstLarger = 0;
  while (firstLarger < controlCount && pos > colorControls[firstLarger].position)
    firstLarger++;
  // less than the leftmost color control
  if (firstLarger <= 0) {
    control.color = colorControls[firstLarger].color;
  }
  // greater than the rightmost color control
  else if (firstLarger >= controlCount) {
    control.color = colorControls[firstLarger - 1].color;
  }
  // between two color controls
  else {
    ColorControl &left = colorControls[firstLarger - 1];
    ColorControl &right = colorControls[firstLarger];
    float w = std::abs(pos - left.position) / std::abs(right.position - left.position);
    control.color = mix(left.color, right.color, w);
  }
  m_colorControls.push_back(control);
  updateColorMap();
  return m_colorControls.back();
}

inline void TransferFunctionCore::removeColorControl(int index)
{
  assert(index >= 0 && index < int(m_colorControls.size()));
  for (int i = index; i < int(m_colorControls.size()) - 1; ++i) m_colorControls[i] = m_colorControls[i + 1];
  m_colorControls.pop_back();
  updateColorMap();
}

inline void TransferFunctionCore::clearColorControls()
{
  m_colorControls.clear();
  updateColorMap();
}

inline int TransferFunctionCore::alphaControlCount() const
{
  return int(m_alphaControls.size());
}

inline TransferFunctionCore::AlphaControl& TransferFunctionCore::alphaControl(int index)
{
  return m_alphaControls[index];
}

inline const TransferFunctionCore::AlphaControl& TransferFunctionCore::alphaControl(int index) const
{
  return m_alphaControls[index];
}

inline TransferFunctionCore::AlphaControl& TransferFunctionCore::addAlphaControl(const AlphaControl &ctrl)
{
  m_alphaControls.push_back(ctrl);
  updateColorMap();
  return m_alphaControls.back();
}

inline TransferFunctionCore::AlphaControl& TransferFunctionCore::addAlphaControl(const vec2f &pos)
{
  return addAlphaControl(AlphaControl(pos));
}

inline void TransferFunctionCore::removeAlphaControl(int index)
{
  assert(index >= 0 && index < int(m_alphaControls.size()));
  for (int i = index; i < int(m_alphaControls.size()) - 1; ++i) m_alphaControls[i] = m_alphaControls[i + 1];
  m_alphaControls.pop_back();
  updateColorMap();
}

inline void TransferFunctionCore::clearAlphaControls()
{
  m_alphaControls.clear();
  updateColorMap();
}

inline int TransferFunctionCore::gaussianObjectCount() const
{
  return int(m_gaussianObjects.size());
}

inline TransferFunctionCore::GaussianObject& TransferFunctionCore::gaussianObject(int index)
{
  return m_gaussianObjects[index];
}

inline const TransferFunctionCore::GaussianObject& TransferFunctionCore::gaussianObject(int index) const
{
  return m_gaussianObjects[index];
}

inline TransferFunctionCore::GaussianObject& TransferFunctionCore::addGaussianObject(const GaussianObject &gaussObj)
{
  m_gaussianObjects.push_back(gaussObj);
  if (m_gaussianObjects.back().alphaArray.size() != m_alphaArray.size()) {
    m_gaussianObjects.back().alphaArray.resize(m_alphaArray.size());
    m_gaussianObjects.back().update();
  }
  updateColorMap();
  return m_gaussianObjects.back();
}

inline TransferFunctionCore::GaussianObject& TransferFunctionCore::addGaussianObject(float mean, float sigma, float heightFactor)
{
  m_gaussianObjects.push_back(GaussianObject(mean, sigma, heightFactor, resolution()));
  updateColorMap();
  return m_gaussianObjects.back();
}

inline void TransferFunctionCore::removeGaussianObject(int index)
{
  assert(index >= 0 && index < int(m_gaussianObjects.size()));
  for (int i = index; i < int(m_gaussianObjects.size()) - 1; ++i) m_gaussianObjects[i] = m_gaussianObjects[i + 1];
  m_gaussianObjects.pop_back();
  updateColorMap();
}

inline void TransferFunctionCore::clearGaussianObjects()
{
  m_gaussianObjects.clear();
  updateColorMap();
}

inline void TransferFunctionCore::updateColorMap()
{
  std::vector<ColorControl> colorControls = m_colorControls;
  if (colorControls.empty()) {
    colorControls.push_back(ColorControl(0.0f, 0.0f, 0.0f, 0.0f));
  }
  std::sort(colorControls.begin(), colorControls.end());
  int controlCount = int(colorControls.size());
  auto upperBound = colorControls.begin();
  for (int i = 0; i < resolution(); ++i) {
    float value = (float(i) + 0.5f) / float(resolution());
    vec3f color;
    upperBound = std::upper_bound(upperBound, colorControls.end(), ColorControl(value, 0.0f, 0.0f, 0.0f));
    int ubIndex = int(upperBound - colorControls.begin());
    // less than the leftmost color control
    if (ubIndex <= 0) {  
      color = colorControls[ubIndex].color;
    } 
    
    // greater than the rightmost color control
    else if (ubIndex >= controlCount) { 
      color = colorControls[ubIndex - 1].color;
    } 
    // between two color controls
    else {  
      ColorControl &left = colorControls[ubIndex - 1];
      ColorControl &right = colorControls[ubIndex];
      float w = std::abs(value - left.position) / std::abs(right.position - left.position);
      color = mix(left.color, right.color, w);
    }
    m_rgbaTable[i] = vec4f(color, m_alphaArray[i]);
    for (int j = 0; j < int(m_gaussianObjects.size()); ++j) {
      m_rgbaTable[i].w = std::max(m_rgbaTable[i].w, m_gaussianObjects[j].alphaArray[i]);
    }
  }
  updateFromAlphaControls();
}

inline std::unique_ptr<TransferFunctionCore> TransferFunctionCore::fromRainbowMap(int resolution)
{
  std::unique_ptr<TransferFunctionCore> ret(new TransferFunctionCore(resolution));
  ret->m_colorControls.clear();
  ret->m_colorControls.push_back(ColorControl(0.0f / 6.0f, 0.0f, 0.364706f, 1.0f));
  ret->m_colorControls.push_back(ColorControl(1.0f / 6.0f, 0.0f, 1.0f, 0.976471f));
  ret->m_colorControls.push_back(ColorControl(2.0f / 6.0f, 0.0f, 1.0f, 0.105882f));
  ret->m_colorControls.push_back(ColorControl(3.0f / 6.0f, 0.968627f, 1.0f, 0.0f));
  ret->m_colorControls.push_back(ColorControl(4.0f / 6.0f, 1.0f, 0.490196f, 0.0f));
  ret->m_colorControls.push_back(ColorControl(5.0f / 6.0f, 1.0f, 0.0f, 0.0f));
  ret->m_colorControls.push_back(ColorControl(6.0f / 6.0f, 0.662745f, 0.0f, 1.0f));
  ret->m_gaussianObjects.clear();
  ret->updateColorMap();
  return ret;
}

inline void TransferFunctionCore::updateFromAlphaControls()
{
  if (m_alphaControls.empty()) return;
  std::vector<AlphaControl> opcControls = m_alphaControls;
  auto compFunc = [](const AlphaControl &a, const AlphaControl &b) {
    return (a.pos.x < b.pos.x);
  };
  std::sort(opcControls.begin(), opcControls.end(), compFunc);
  int controlCount = int(opcControls.size());
  auto upperBound = opcControls.begin();
  for (int i = 0; i < resolution(); ++i) {
    // double value = (double(i) + 0.5f) / double(resolution());
    double value = double(i) / double(resolution() - 1);
    upperBound = std::upper_bound(upperBound, opcControls.end(), AlphaControl(vec2f(value, 0.0)), compFunc);
    int ubIndex = int(upperBound - opcControls.begin());
    double alpha;
    // less than the leftmost control
    if (ubIndex <= 0) {
      alpha = opcControls[ubIndex].pos.y;
    } 
    // greater than the rightmost control
    else if (ubIndex >= controlCount) { 
      alpha = opcControls[ubIndex - 1].pos.y;
    } 
    // between two color controls
    else {  
      auto &left = opcControls[ubIndex - 1];
      auto &right = opcControls[ubIndex];
      double w = std::abs(value - left.pos.x) / std::abs(right.pos.x - left.pos.x);
      alpha = mix(left.pos.y, right.pos.y, w);
    }
    m_rgbaTable[i].w = std::max(m_rgbaTable[i].w, float(alpha));
  }
}

inline void saveTransferFunction(const TransferFunctionCore& tfn, json& jstfn)
{
  jstfn["resolution"] = tfn.resolution();
  jstfn["alphaArray"]["encoding"] = "BASE64";
  jstfn["alphaArray"]["data"] = toBase64(reinterpret_cast<const char *>(tfn.alphaArray()), sizeof(float) * tfn.resolution());

  for (int i = 0; i < tfn.colorControlCount(); ++i) {
    auto cControl = tfn.colorControl(i);
    jstfn["colorControls"][i]["position"] = cControl.position;
    jstfn["colorControls"][i]["color"] = colorToJson(cControl.color);
  }
  for (int i = 0; i < tfn.alphaControlCount(); ++i) {
    auto oControl = tfn.alphaControl(i);
    jstfn["opacityControl"][i]["position"] = scalarToJson<vec2f>(oControl.pos);
  }
  for (int i = 0; i < tfn.gaussianObjectCount(); ++i) {
    auto gaussianObject = tfn.gaussianObject(i);
    jstfn["gaussianObjects"][i]["mean"] = gaussianObject.mean;
    jstfn["gaussianObjects"][i]["sigma"] = gaussianObject.sigma;
    jstfn["gaussianObjects"][i]["heightFactor"] = gaussianObject.heightFactor;
  }
}

inline void loadTransferFunction(const json& jstfn, TransferFunctionCore& tfn)
{
  int resolution = 1024;
  if (jstfn.contains("resolution")) {
    resolution = jstfn["resolution"].get<int>();
  }

  std::string alphaArrayBase64;
  if (jstfn.contains("alphaArray") && jstfn["alphaArray"].contains("data")) {
    if (jstfn["alphaArray"].contains("encoding") && jstfn["alphaArray"]["encoding"].get<std::string>() == "BASE64") {
      alphaArrayBase64 = jstfn["alphaArray"]["data"].get<std::string>();
      resolution = int(sizeBase64(alphaArrayBase64) / sizeof(float));
    }
  }

  TransferFunctionCore tf(resolution);
  if (!alphaArrayBase64.empty()) {
    fromBase64(alphaArrayBase64, reinterpret_cast<char *>(tf.alphaArray()));
  }

  tf.clearColorControls();
  if (jstfn.contains("colorControls")) {
    int count = jstfn["colorControls"].size();
    for (int i = 0; i < count; ++i) {
      const json &js_cc = jstfn["colorControls"][i];
      if (!js_cc.contains("position") || !js_cc.contains("color")) continue;
      TransferFunctionCore::ColorControl colorControl;
      colorControl.position = js_cc["position"].get<float>();
      colorControl.color = colorFromJson<vec3f>(js_cc["color"]);
      tf.addColorControl(colorControl);
    }
  }

  tf.clearAlphaControls();
  if (jstfn.contains("opacityControl")) {
    int count = jstfn["opacityControl"].size();
    for (int i = 0; i < count; ++i) {
      const json &js_oc = jstfn["opacityControl"][i];
      if (!js_oc.contains("position")) continue;
      TransferFunctionCore::AlphaControl opcControl(scalarFromJson<vec2f>(js_oc["position"]));
      tf.addAlphaControl(opcControl);
    }
  }

  tf.clearGaussianObjects();
  if (jstfn.contains("gaussianObjects")) {
    int count = jstfn["gaussianObjects"].size();
    for (int i = 0; i < count; ++i) {
      const json &json_go = jstfn["gaussianObjects"][i];
      if (!json_go.contains("mean") || !json_go.contains("sigma") || !json_go.contains("heightFactor")) continue;
      TransferFunctionCore::GaussianObject gaussianObject(json_go["mean"].get<float>(), json_go["sigma"].get<float>(), json_go["heightFactor"].get<float>(), resolution);
      tf.addGaussianObject(gaussianObject);
    }
  }
  tf.updateColorMap();

  /* finalize */

  // auto *table = (vec4f *)tf.data();
  // std::vector<vec4f> color(tf.resolution());
  // std::vector<vec2f> alpha(tf.resolution());
  // for (int i = 0; i < tf.resolution(); ++i) {
  //   auto rgba = table[i];
  //   color[i] = vec4f((float)i / (tf.resolution() - 1), rgba.x, rgba.y, rgba.z);
  //   alpha[i] = vec2f((float)i / (tf.resolution() - 1), rgba.w);
  // }
  // tfn.color = std::move(color);
  // tfn.alpha = std::move(alpha);

  // if (jstfn.contains("valueRange")) {
  //   auto r = rangeFromJson(jstfn["valueRange"]);
  //   tfn.range.x = r.x;
  //   tfn.range.y = r.y;
  // }
  // else {
  //   tfn.range.x = 0.f;
  //   tfn.range.y = 1.f;
  // }

  tfn = std::move(tf);
}

} // namespace tfn
