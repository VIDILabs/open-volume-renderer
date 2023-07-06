#pragma once

#include <iostream>
#include <string>

class ProgressBar {
  std::string msg;
  float progress{};
  int barWidth;

  void print()
  {
    if (msg.empty())
      std::cout << "[";
    else
      std::cout << msg << " [";
    const int pos = (int)(barWidth * progress);
    for (int i = 0; i < barWidth; ++i) {
      if (i < pos)
        std::cout << "=";
      else if (i == pos)
        std::cout << ">";
      else
        std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
  }

public:
  ProgressBar(std::string prompt = "", int width = 80) : msg(prompt), barWidth(width - (int)prompt.length()) {}

  void update(float p, std::string prompt = "")
  {
    progress = p < 0.f ? 0.f : p > 1.f ? 1.f : p;
    msg = prompt;
    print();
  }
  
  void finalize()
  {
    progress = 1.f;
    print();
    std::cout << std::endl;
  }

};
