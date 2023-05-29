#ifndef VIDI_HIGH_PERFORMANCE_TIMER_H
#define VIDI_HIGH_PERFORMANCE_TIMER_H

#if defined(_WIN32)
#include <windows.h>
#endif

#include <chrono>
#include <cassert>
#include <string>
#include <functional>
#include <iostream>

namespace vidi {
namespace details {

struct HighPerformanceTimer {
private:
  static const char* c_str(const std::string& s) { return s.c_str(); }

  template<typename T>
  static T c_str(T s)
  {
    return s;
  }

public:
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

private:
#if defined(_WIN32)
  LARGE_INTEGER beginClock, endClock, cpuClockFreq, wallTime_ticks;
#else
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> time_span;
#endif
  bool inUse = false;
  double time_ms;

public:
  HighPerformanceTimer() : inUse(false)
  {
#if defined(_WIN32)
    QueryPerformanceFrequency(&cpuClockFreq);
#endif
    reset();
  }

  void start()
  {
    inUse = true;
#if defined(_WIN32)
    QueryPerformanceCounter(&beginClock);
#else
    t1 = std::chrono::high_resolution_clock::now();
#endif
  }

  void stop()
  {
    inUse = false;
#if defined(_WIN32)
    QueryPerformanceCounter(&endClock);
    wallTime_ticks.QuadPart += endClock.QuadPart - beginClock.QuadPart;
#else
    t2 = std::chrono::high_resolution_clock::now();
    time_span += std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
#endif
  }

  double milliseconds()
  {
    assert(!inUse);
#if defined(_WIN32)
    double wallTime_us = double(wallTime_ticks.QuadPart) * 1000.0 * 1000.0 / double(cpuClockFreq.QuadPart);
    time_ms = wallTime_us / 1000.0;
#else
    time_ms = time_span.count() * 1000.0f;
#endif
    return time_ms;
  }

  void reset()
  {
#if defined(_WIN32)
    wallTime_ticks.QuadPart = 0;
#else
    time_span = std::chrono::duration<double>(0);
#endif
  }

  void measure_time(std::ostream& os, const std::string& msg)
  {
    stop();
    milliseconds();
    if (time_ms < 1000) { os << stringf("%s [time: %.3f ms]\n", msg.c_str(), time_ms); }
    else {
      os << stringf("%s [time: %.3f s ]\n", msg.c_str(), time_ms / 1000) << std::flush;
    }
  }

  template<typename... Ts>
  void measure(std::ostream& os, const std::string& format, Ts... rest)
  {
    measureTime(os, stringf(format, rest...));
  }

  void run(std::function<void()> func, std::ostream& os, const std::string& msg)
  {
    reset();
    start();
    func();
    measure_time(os, msg);
  }
};

struct HighPerformanceBandwidth {
private:
  using Timer = HighPerformanceTimer;
  Timer timer;

public:
  void reset() { timer.reset(); }

  void start() { timer.start(); }

  void stop() { timer.stop(); }

  void measure_bandwidth(std::ostream& os, size_t bytes, const std::string& msg)
  {
    stop();
    double time_ms = timer.milliseconds();
    const double mb = double(bytes) / 1000.0 / 1000.0;
    const double time_s = time_ms / 1000.0;
    const double bwmb_s = mb / time_s;
    os << Timer::stringf("%s (%.3f MB) [I/O: %.3f MB/s]\n", msg.c_str(), mb, bwmb_s) << std::flush;
  }

  template<typename... Ts>
  void measure(std::ostream& os, size_t bytes, const std::string& format, Ts... rest)
  {
    measure_bandwidth(os, bytes, Timer::stringf(format, rest...));
  }

  void run(std::function<void()> func, size_t bytes, std::ostream& os, const std::string& msg)
  {
    reset();
    start();
    func();
    measure_bandwidth(os, bytes, msg);
  }
};

} // namespace details

struct StackBandwidth {
private:
  using Bandwidth = details::HighPerformanceBandwidth;
  Bandwidth timer;
  size_t bytes;
  std::ostream* os;
  std::string msg;

public:
  StackBandwidth(size_t bytes, std::string msg = "") : bytes(bytes), os(&std::cout), msg(msg) { timer.start(); }
  StackBandwidth(size_t bytes, std::ostream& os, std::string msg = "") : bytes(bytes), os(&os), msg(msg) { timer.start(); }
  ~StackBandwidth()
  {
    timer.stop();
    timer.measure_bandwidth(*os, bytes, msg);
  }
};

struct StackTimer {
private:
  using Timer = details::HighPerformanceTimer;
  Timer timer;
  std::ostream* os = nullptr;
  std::string msg = "";

public:
  StackTimer() : os(&std::cout) { timer.start(); }

  StackTimer(std::ostream& os, std::string msg = "") : os(&os) { timer.start(); }

  template<typename... Ts>
  StackTimer(const std::string& format, Ts... rest) : os(&std::cout)
  {
    msg = Timer::stringf(format, rest...);
    timer.start();
  }

  template<typename... Ts>
  StackTimer(std::ostream& os, const std::string& format, Ts... rest) : os(&os)
  {
    msg = Timer::stringf(format, rest...);
    timer.start();
  }

  ~StackTimer()
  {
    timer.stop();
    timer.measure_time(*os, msg);
  }
};

} // namespace vidi

#endif // VIDI_HIGH_PERFORMANCE_TIMER_H
