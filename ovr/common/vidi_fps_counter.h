#pragma once

#include <atomic>
#include <chrono>
#include <vector>

namespace vidi {

struct FPSCounter {
  std::atomic<double> fps;
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  int counter;
  int counter_interval;

  FPSCounter() : start(std::chrono::high_resolution_clock::now()), counter(0), counter_interval(20), fps(0.0) {}

  virtual bool count()
  {
    ++counter;
    if (counter == counter_interval) {
      end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end - start;
      fps = counter / diff.count();
      start = end;
      counter = 0;
      return true;
    }
    return false;
  }
};

struct HistoryFPSCounter : public FPSCounter {
  std::vector<float> indices;
  std::vector<float> fps_history;
  std::vector<float> frame_time_history;
  std::vector<float> render_time_history;
  std::vector<float> inference_time_history;

  long frame;
  int size;


  HistoryFPSCounter() : frame(0), size(50) {
    for (int i = 0; i < size*counter_interval; i+=counter_interval) {
      indices.push_back((float)i);
    }
  }

  bool count() override {
    ++frame;
    return FPSCounter::count();
  }

  void update_history(float frame_time, float render_time, float inference_time) {
    fps_history.push_back((float)fps);
    frame_time_history.push_back(frame_time);
    render_time_history.push_back(render_time);
    inference_time_history.push_back(inference_time);

    if (fps_history.size() > size) {
      fps_history.erase(fps_history.begin());
      frame_time_history.erase(frame_time_history.begin());
      render_time_history.erase(render_time_history.begin());
      inference_time_history.erase(inference_time_history.begin());
    }
  }
};

} // namespace vidi
