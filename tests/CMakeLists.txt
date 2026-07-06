cmake_minimum_required(VERSION 3.12..3.18)
project(da-tests LANGUAGES C CXX)

add_executable(da_tests test_ops.cpp)
target_link_libraries(da_tests PRIVATE depth-anything)
target_include_directories(da_tests PRIVATE ../include)
