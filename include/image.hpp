// Copyright 2026 depth-anything-v2.cpp contributors
// Licensed under the Apache License, Version 2.0

#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace da {

// RGB image in row-major order (H x W x 3).
struct Image {
    std::vector<uint8_t> data; // H * W * 3, RGB
    int32_t h = 0;
    int32_t w = 0;

    bool empty() const { return data.empty(); }
    uint8_t* row(int y) { return data.data() + (size_t)y * w * 3; }
    const uint8_t* row(int y) const { return data.data() + (size_t)y * w * 3; }
};

// Load an image from file. Returns empty Image on failure.
Image load_image(const std::string& path);

// Save a 16-bit grayscale depth map.
bool save_depth(const std::string& path, const std::vector<uint16_t>& data, int32_t w, int32_t h);

// Save an 8-bit RGB image.
bool save_image(const std::string& path, const std::vector<uint8_t>& data, int32_t w, int32_t h);

} // namespace da
