// ======================================================================== //
// Copyright Qi Wu, since 2019                                              //
// Copyright SCI Institute, University of Utah, 2018                        //
// ======================================================================== //
#pragma once

#include <cmath>
#include <limits>

namespace tfn {

template <class T>
inline const T &clamp(const T &v, const T &lo, const T &hi)
{
  return (v < lo) ? lo : (hi < v ? hi : v);
}

template <typename T>
inline std::pair<int, int> find_interval(std::vector<T> *array, float p)
{
  p = clamp(p, 0.f, 1.f);

  typename std::vector<T>::iterator it_lower, it_upper;

  if (array->begin()->p() > p)
    return std::make_pair(0, 1);

  it_upper = std::lower_bound(array->begin(), array->end(), p, [](const T &pt, float p) { return pt.p() < p; });

  if (it_upper == array->end()) {
    it_lower = array->end() - 1;
    it_upper = array->end() - 1;
  } else if (it_upper == array->begin()) {
    it_lower = array->begin();
    it_upper = array->begin();
  } else {
    it_lower = it_upper - 1;
  }

  return std::make_pair(std::distance(array->begin(), it_lower), std::distance(array->begin(), it_upper));
}

inline float lerp(const float &l, const float &r, const float &pl, const float &pr, const float &p)
{
  const float dl = std::abs(pr - pl) > std::numeric_limits<float>::epsilon() ? (p - pl) / (pr - pl) : 0.f;
  const float dr = 1.f - dl;
  return l * dr + r * dl;
}

struct GaussianKernel
{
  GaussianKernel() : mean(0.5f), sigma(1.0f), height_factor(1.0f) {}

  GaussianKernel(float _mean, float _sigma, float _height_factor) : mean(_mean), sigma(_sigma), height_factor(_height_factor) {}

  float operator()(float x) const
  {
    float diff = x - mean;
    return height_factor / (sigma * std::sqrt(2.0f * float(M_PI))) * std::exp(-(diff * diff) / (2.0f * sigma * sigma));
  }

  float get_height() const
  {
    return operator()(mean);
  }

  void set_height(float h)
  {
    height_factor = h * sigma * std::sqrt(2.0f * float(M_PI));
  }

  float get_mean() const
  {
    return mean;
  }

  void set_mean(float m)
  {
    mean = m;
  }

  float get_sigma() const
  {
    return sigma;
  }

  void set_sigma(float s)
  {
    sigma = s > 1e-4f ? s : 1e-4f;
  }

 private:
  float mean;
  float sigma;
  float height_factor;
};

} // namespace tfn
