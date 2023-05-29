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

#ifndef DIVA_VOLVIS_COMMON_H
#define DIVA_VOLVIS_COMMON_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

inline const std::vector<float>&
makeColorwheel()
{
  // color encoding scheme
  // adapted from the color circle idea described at
  // http://members.shaw.ca/quadibloc/other/colint.htm
  const int RY = 15;
  const int YG = 6;
  const int GC = 4;
  const int CB = 11;
  const int BM = 13;
  const int MR = 6;
  const int ncols = RY + YG + GC + CB + BM + MR;

  static std::vector<float> colorwheel(ncols * 3); // r g b
  static bool first = true;

  if (!first)
    return colorwheel;

  first = false;

  int col = 0;
  /* % RY
    colorwheel(1 : RY, 1) = 255;
    colorwheel(1 : RY, 2) = floor(255 * (0 : RY - 1) / RY)';
    col = col + RY; */
  for (int i = 0; i < RY; ++i) {
    colorwheel[3 * i + 0] = 1.f;
    colorwheel[3 * i + 1] = std::floor(255.f * (i - 1) / RY) / 255.f;
    colorwheel[3 * i + 2] = 0.f;
  }
  col = col + RY;

  /* % YG
    colorwheel(col + (1 : YG), 1) = 255 - floor(255 * (0 : YG - 1) / YG)';
    colorwheel(col + (1 : YG), 2) = 255;
    col = col + YG; */
  for (int i = 0; i < YG; ++i) {
    colorwheel[3 * (col + i) + 0] = 1.f - std::floor(255.f * (i - 1) / YG) / 255.f;
    colorwheel[3 * (col + i) + 1] = 1.f;
    colorwheel[3 * (col + i) + 2] = 0.f;
  }
  col = col + YG;

  /* % GC
    colorwheel(col + (1 : GC), 2) = 255;
    colorwheel(col + (1 : GC), 3) = floor(255 * (0 : GC - 1) / GC)';
    col = col + GC; */
  for (int i = 0; i < GC; ++i) {
    colorwheel[3 * (col + i) + 0] = 0.f;
    colorwheel[3 * (col + i) + 1] = 1.f;
    colorwheel[3 * (col + i) + 2] = std::floor(255.f * (i - 1) / GC) / 255.f;
  }
  col = col + GC;

  /* % CB
    colorwheel(col + (1 : CB), 2) = 255 - floor(255 * (0 : CB - 1) / CB)';
    colorwheel(col + (1 : CB), 3) = 255;
    col = col + CB; */
  for (int i = 0; i < CB; ++i) {
    colorwheel[3 * (col + i) + 0] = 0.f;
    colorwheel[3 * (col + i) + 1] = 1.f - std::floor(255.f * (i - 1) / CB) / 255.f;
    colorwheel[3 * (col + i) + 2] = 1.f;
  }
  col = col + CB;

  /* % BM
    colorwheel(col + (1 : BM), 3) = 255;
    colorwheel(col + (1 : BM), 1) = floor(255 * (0 : BM - 1) / BM)';
    col = col + BM; */
  for (int i = 0; i < BM; ++i) {
    colorwheel[3 * (col + i) + 0] = std::floor(255.f * (i - 1) / BM) / 255.f;
    colorwheel[3 * (col + i) + 1] = 0.f;
    colorwheel[3 * (col + i) + 2] = 1.f;
  }
  col = col + BM;

  /* % MR
    colorwheel(col + (1 : MR), 3) = 255 - floor(255 * (0 : MR - 1) / MR)';
    colorwheel(col + (1 : MR), 1) = 255; */
  for (int i = 0; i < MR; ++i) {
    colorwheel[3 * (col + i) + 0] = 1.f - std::floor(255.f * (i - 1) / MR) / 255.f;
    colorwheel[3 * (col + i) + 1] = 0.f;
    colorwheel[3 * (col + i) + 2] = 1.f;
  }

  return colorwheel;
}

// ------------------------------------------------------------------
// I/O helper functions
// ------------------------------------------------------------------

template<size_t Size>
inline void
swapBytes(void* data)
{
  char* p = reinterpret_cast<char*>(data);
  char* q = p + Size - 1;
  while (p < q)
    std::swap(*(p++), *(q--));
}

template<>
inline void
swapBytes<1>(void*)
{
}

template<>
inline void
swapBytes<2>(void* data)
{
  char* p = reinterpret_cast<char*>(data);
  std::swap(p[0], p[1]);
}

template<>
inline void
swapBytes<4>(void* data)
{
  char* p = reinterpret_cast<char*>(data);
  std::swap(p[0], p[3]);
  std::swap(p[1], p[2]);
}

template<>
inline void
swapBytes<8>(void* data)
{
  char* p = reinterpret_cast<char*>(data);
  std::swap(p[0], p[7]);
  std::swap(p[1], p[6]);
  std::swap(p[2], p[5]);
  std::swap(p[3], p[4]);
}

template<typename T>
inline void
swapBytes(T* data)
{
  swapBytes<sizeof(T)>(reinterpret_cast<void*>(data));
}

inline void
reverseByteOrder(char* data, size_t elemCount, size_t elemSize)
{
  switch (elemSize) {
  case 1: break;
  case 2:
    for (size_t i = 0; i < elemCount; ++i)
      swapBytes<2>(&data[i * elemSize]);
    break;
  case 4:
    for (size_t i = 0; i < elemCount; ++i)
      swapBytes<4>(&data[i * elemSize]);
    break;
  case 8:
    for (size_t i = 0; i < elemCount; ++i)
      swapBytes<8>(&data[i * elemSize]);
    break;
  default: assert(false);
  }
}

/*! This calls a given function in a continuous loop on a background
    thread owned by AsyncLoop. While it is running, the function it was
    constructed with is called over and over in a loop. When stopped, the
    thread is put to sleep until it is started again.

    An AsyncLoop has to be explicitly started, it is not automatically
    started on construction.
 */
class AsyncLoop {
public:
  template<typename LOOP_BODY_FCN> AsyncLoop(LOOP_BODY_FCN&& fcn);

  ~AsyncLoop();

  void start();
  void stop();

private:
  // Struct shared with the background thread to avoid dangling ptrs or
  // tricky synchronization when destroying the AsyncLoop and scheduling
  // threads with TBB, since we don't have a join point to sync with
  // the running thread
  struct AsyncLoopData {
    std::atomic<bool> threadShouldBeAlive{ true };
    std::atomic<bool> shouldBeRunning{ false };
    std::atomic<bool> insideLoopBody{ false };

    std::condition_variable runningCond;
    std::mutex runningMutex;
  };

  std::shared_ptr<AsyncLoopData> loop;
  std::thread backgroundThread;
};

// Inlined members
// //////////////////////////////////////////////////////////

template<typename LOOP_BODY_FCN> inline AsyncLoop::AsyncLoop(LOOP_BODY_FCN&& fcn) : loop(nullptr)
{
  std::shared_ptr<AsyncLoopData> l = std::make_shared<AsyncLoopData>();
  loop = l;

  auto mainLoop = [l, fcn]() {
    while (l->threadShouldBeAlive) {
      if (!l->threadShouldBeAlive)
        return;

      if (l->shouldBeRunning) {
        l->insideLoopBody = true;
        fcn();
        l->insideLoopBody = false;
      }
      else {
        std::unique_lock<std::mutex> lock(l->runningMutex);
        l->runningCond.wait(lock, [&] { return l->shouldBeRunning.load() || !l->threadShouldBeAlive.load(); });
      }
    }
  };

  backgroundThread = std::thread(mainLoop);
}

inline AsyncLoop::~AsyncLoop()
{
  // Note that the mutex here is still required even though these vars
  // are atomic, because we need to sync with the condition variable waiting
  // state on the async thread. Otherwise we might signal and the thread
  // will miss it, since it wasn't watching.
  {
    std::unique_lock<std::mutex> lock(loop->runningMutex);
    loop->threadShouldBeAlive = false;
    loop->shouldBeRunning = false;
  }
  loop->runningCond.notify_one();

  if (backgroundThread.joinable()) {
    backgroundThread.join();
  }
}

inline void
AsyncLoop::start()
{
  if (!loop->shouldBeRunning) {
    // Note that the mutex here is still required even though these vars
    // are atomic, because we need to sync with the condition variable
    // waiting state on the async thread. Otherwise we might signal and the
    // thread will miss it, since it wasn't watching.
    {
      std::unique_lock<std::mutex> lock(loop->runningMutex);
      loop->shouldBeRunning = true;
    }
    loop->runningCond.notify_one();
  }
}

inline void
AsyncLoop::stop()
{
  if (loop->shouldBeRunning) {
    loop->shouldBeRunning = false;
    while (loop->insideLoopBody.load()) {
      std::this_thread::yield();
    }
  }
}

namespace detail {

template<typename TASK_T> struct AsyncTaskImpl {
  AsyncTaskImpl(TASK_T&& fcn);
  void wait();

private:
  std::thread thread;
};

template<typename TASK_T> inline AsyncTaskImpl<TASK_T>::AsyncTaskImpl(TASK_T&& fcn) : thread(std::forward<TASK_T>(fcn))
{
}

template<typename TASK_T>
inline void
AsyncTaskImpl<TASK_T>::wait()
{
  if (thread.joinable())
    thread.join();
}
} // namespace detail

template<typename T> struct AsyncTask {
  AsyncTask(std::function<T()> fcn)
    : taskImpl([this, fcn]() {
      retValue = fcn();
      jobFinished = true;
    })
  {
  }

  virtual ~AsyncTask() noexcept
  {
    wait();
  }

  bool finished() const
  {
    return jobFinished;
  }
  bool valid() const
  {
    return jobFinished;
  }
  void wait()
  {
    taskImpl.wait();
  }

  T get()
  {
    if (!jobFinished)
      wait();
    return retValue;
  }

private:
  detail::AsyncTaskImpl<std::function<void()>> taskImpl;
  std::atomic<bool> jobFinished{ false };
  T retValue;
};

#endif // DIVA_VOLVIS_COMMON_H
