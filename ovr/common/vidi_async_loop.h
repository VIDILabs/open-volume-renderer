//. ======================================================================== //
//. Copyright 2018-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under MIT                                                       //
//. ======================================================================== //

// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef VIDI_ASYNC_LOOP_H
#define VIDI_ASYNC_LOOP_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

namespace vidi {

/*! This calls a given function in a continuous loop on a background
    thread owned by AsyncLoop. While it is running, the function it was
    constructed with is called over and over in a loop. When stopped, the
    thread is put to sleep until it is started again.

    An AsyncLoop has to be explicitly started, it is not automatically
    started on construction.
 */
class AsyncLoop
{
 public:
  template <typename LOOP_BODY_FCN>
  AsyncLoop(LOOP_BODY_FCN &&fcn);

  ~AsyncLoop();

  void start();
  void stop();

 private:
  // Struct shared with the background thread to avoid dangling ptrs or
  // tricky synchronization when destroying the AsyncLoop and scheduling
  // threads with TBB, since we don't have a join point to sync with
  // the running thread
  struct AsyncLoopData
  {
    std::atomic<bool> threadShouldBeAlive{true};
    std::atomic<bool> shouldBeRunning{false};
    std::atomic<bool> insideLoopBody{false};

    std::condition_variable runningCond;
    std::mutex runningMutex;
  };

  std::shared_ptr<AsyncLoopData> loop;
  std::thread backgroundThread;
};

// Inlined members
// //////////////////////////////////////////////////////////

template <typename LOOP_BODY_FCN>
inline AsyncLoop::AsyncLoop(LOOP_BODY_FCN &&fcn) : loop(nullptr)
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
      } else {
        std::unique_lock<std::mutex> lock(l->runningMutex);
        l->runningCond.wait(
            lock, [&] { return l->shouldBeRunning.load() || !l->threadShouldBeAlive.load(); });
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

inline void AsyncLoop::start()
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

inline void AsyncLoop::stop()
{
  if (loop->shouldBeRunning) {
    loop->shouldBeRunning = false;
    while (loop->insideLoopBody.load()) {
      std::this_thread::yield();
    }
  }
}

}  // namespace vidi

#endif  // VIDI_ASYNC_LOOP_H
