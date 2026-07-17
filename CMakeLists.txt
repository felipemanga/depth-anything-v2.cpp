cmake_minimum_required(VERSION 3.12..3.18)
project(depth-anything LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP QUIET)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

option(DA_CUDA   "Enable CUDA backend" OFF)
option(DA_METAL  "Enable Metal backend" OFF)
option(DA_VULKAN "Enable Vulkan backend" OFF)

if (DA_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()
if (DA_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()
if (DA_VULKAN)
    add_compile_definitions(GGML_USE_VULKAN)
endif()

set(GGML_CUDA   ${DA_CUDA})
set(GGML_METAL  ${DA_METAL})
set(GGML_VULKAN ${DA_VULKAN})
set(GGML_NATIVE ON)
add_subdirectory(third_party/ggml)

add_library(depth-anything STATIC
    src/backend.cpp
    src/ggml_graph.cpp
    src/model.cpp
    src/image.cpp
    src/vision_ops.cpp
    src/predict.cpp
)
target_include_directories(depth-anything PUBLIC include PRIVATE src)
target_link_libraries(depth-anything PUBLIC ggml)
if (OPENMP_FOUND)
    target_link_libraries(depth-anything PRIVATE OpenMP::OpenMP_CXX)
endif()

add_subdirectory(examples/cli)
add_subdirectory(tests)
