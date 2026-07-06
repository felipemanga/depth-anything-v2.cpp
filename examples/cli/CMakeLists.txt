cmake_minimum_required(VERSION 3.12..3.18)
project(da-cli LANGUAGES C CXX)

add_executable(da_cli main.cpp)
target_link_libraries(da_cli PRIVATE depth-anything)
target_include_directories(da_cli PRIVATE ../..)
