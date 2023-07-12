#ifndef VIDI_PARALLEL_ALGORITHM_H
#define VIDI_PARALLEL_ALGORITHM_H

/* inline functions for basic parallel operations */
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>

namespace vidi {

template<typename T>
void
std_atomic_max(std::atomic<T>& maximum_value, T const& value) noexcept
{
  T prev_value = maximum_value;
  while (prev_value < value && !maximum_value.compare_exchange_weak(prev_value, value)) {
  }
}

template<typename T>
void
std_atomic_min(std::atomic<T>& minimum_value, T const& value) noexcept
{
  T prev_value = minimum_value;
  while (prev_value > value && !minimum_value.compare_exchange_weak(prev_value, value)) {
  }
}

namespace parallel {

template<typename TSum, typename TArray>
static TSum
prefixsum(TSum y[], const TArray& z, size_t n)
{
  return tbb::parallel_scan(
    tbb::blocked_range<size_t>(0, n), TSum(0),
    [&y, &z](const tbb::blocked_range<size_t>& r, TSum sum, bool is_final_scan) -> TSum {
      TSum temp = sum;
      for (size_t i = r.begin(); i < r.end(); ++i) {
        temp = temp + TSum(z[i]);
        if (is_final_scan)
          y[i] = temp;
      }
      return temp;
    },
    [&y, &z](TSum left, TSum right) { return left + right; });
}

template<typename order_iterator>
void
reorder(order_iterator order_begin, order_iterator order_end, void* _source, size_t size)
{
  typedef typename std::iterator_traits<order_iterator>::value_type index_t;

  const index_t length = order_end - order_begin;
  char* source = (char*)_source;
  char* buffer = new char[size * length];
  std::memcpy(buffer, source, size * length);

  tbb::parallel_for(index_t(), length, [&](const index_t& s) {
    const index_t d = order_begin[s];
    auto* vsrc = buffer + size * s;
    auto* vdst = source + size * d;
    std::memcpy(vdst, vsrc, size);
  });

  delete[] buffer;
}

template<typename T>
T
findmax(const T* array, size_t size)
{
  std::atomic<T> mv(-std::numeric_limits<T>::max());

  tbb::parallel_for((size_t)0, size, [&](const size_t& i) {
    T v = array[i];
    std_atomic_max(mv, v);
  });

  return mv.load();
}

template<typename T>
T
findmin(const T* array, size_t size)
{
  std::atomic<T> mv(std::numeric_limits<T>::max());

  tbb::parallel_for((size_t)0, size, [&](const size_t& i) {
    T v = array[i];
    std_atomic_min(mv, v);
  });

  return mv.load();
}

template<typename T>
std::pair<T, T>
compute_scalar_minmax(const void* _array, size_t count, size_t stride)
{
  static_assert(std::is_scalar<T>::value, "expecting a scalar type");

  if (stride == 0)
    stride = sizeof(T);

  const T* array = (const T*)_array;
  auto value = [array, stride](size_t index) -> T {
    const auto begin = (const uint8_t*)array;
    const auto curr = (T*)(begin + index * stride);
    return static_cast<T>(*curr);
  };

  T init;

  init = std::numeric_limits<T>::min();
  T actual_max = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, count), init,
    [value](const tbb::blocked_range<size_t>& r, T v) -> T {
      for (auto i = r.begin(); i != r.end(); ++i)
        v = std::max(v, value(i));
      return v;
    },
    [](T x, T y) -> T { return std::max(x, y); });

  init = std::numeric_limits<T>::max();
  T actual_min = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, count), init,
    [value](const tbb::blocked_range<size_t>& r, T v) -> T {
      for (auto i = r.begin(); i != r.end(); ++i)
        v = std::min(v, value(i));
      return v;
    },
    [](T x, T y) -> T { return std::min(x, y); });

  return std::make_pair(actual_min, actual_max);
}

} // namespace parallel

} // namespace vidi

#endif // VIDI_PARALLEL_ALGORITHM_H
