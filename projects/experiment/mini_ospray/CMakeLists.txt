add_executable(exp_mini_ospray mini_ospray.cpp)
target_link_libraries(exp_mini_ospray ospray::ospray TBB::tbb)
set_property(TARGET exp_mini_ospray PROPERTY CXX_STANDARD 17)
