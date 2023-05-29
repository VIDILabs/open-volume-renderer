//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <functional>
#include <mutex>

namespace vidi {

/* This implements a 1-to-1 value fence. One thread can set (or "queue") a
 * value for another thread to later get. This is conceptually similar to
 * "doublebuffering" a single value. Note that all values from the producer
 * thread overwrite the "queued" value, where the consumer thread will
 * always get the last value set by the producer thread.
 */
template<typename T>
class TransactionalValue {
public:
  using FuncType = std::function<void(T&)>;
  using FuncTypeConst = std::function<void(const T&)>;

public:
  TransactionalValue() = default;
  ~TransactionalValue() = default;
  template<typename OtherType>
  TransactionalValue(const OtherType& ot);

  /* calls used by the producer thread */
  template<typename OtherType>
  TransactionalValue& operator=(const OtherType& ot);

  TransactionalValue<T>& operator=(const TransactionalValue<T>& fp);

  void assign(const FuncType&& func);

  /* calls used by the consumer thread */
  T& ref();
  const T& ref() const;

  T get();
  T get() const;

  bool update();
  bool update(const FuncTypeConst&&);

  void access(const FuncTypeConst&&);

private:
  std::atomic<bool> newValue{ false };
  T queuedValue;
  T currentValue;

  std::mutex mutex;
};

// Inlined TransactionalValue Members /////////////////////////////////////

template<typename T>
template<typename OtherType>
inline TransactionalValue<T>::TransactionalValue(const OtherType& ot)
{
  currentValue = ot;
}

template<typename T>
template<typename OtherType>
inline TransactionalValue<T>&
TransactionalValue<T>::operator=(const OtherType& ot)
{
  std::lock_guard<std::mutex> lock{ mutex };
  queuedValue = ot;
  newValue = true;
  return *this;
}

template<typename T>
inline TransactionalValue<T>&
TransactionalValue<T>::operator=(const TransactionalValue<T>& fp)
{
  std::lock_guard<std::mutex> lock{ mutex };
  queuedValue = fp.ref();
  newValue = true;
  return *this;
}

template<typename T>
inline void
TransactionalValue<T>::assign(const FuncType&& func)
{
  std::lock_guard<std::mutex> lock{ mutex };
  func(queuedValue);
  newValue = true;
}

template<typename T>
inline T&
TransactionalValue<T>::ref()
{
  return currentValue;
}

template<typename T>
inline const T&
TransactionalValue<T>::ref() const
{
  return currentValue;
}

template<typename T>
inline T
TransactionalValue<T>::get()
{
  return currentValue;
}

template<typename T>
inline T
TransactionalValue<T>::get() const
{
  return currentValue;
}

template<typename T>
inline bool
TransactionalValue<T>::update()
{
  bool didUpdate = false;
  if (newValue) {
    std::lock_guard<std::mutex> lock{ mutex };
    std::swap(currentValue, queuedValue);
    newValue = false;
    didUpdate = true;
  }

  return didUpdate;
}

template<typename T>
inline bool
TransactionalValue<T>::update(const FuncTypeConst&& func)
{
  bool didUpdate = false;
  if (newValue) {
    std::lock_guard<std::mutex> lock{ mutex };
    std::swap(currentValue, queuedValue);
    func(currentValue);
    newValue = false;
    didUpdate = true;
  }
  return didUpdate;
}

template<typename T>
inline void TransactionalValue<T>::access(const FuncTypeConst&& func)
{
  std::lock_guard<std::mutex> lock{ mutex };
  func(currentValue);
}

}
