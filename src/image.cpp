// Copyright 2026 depth-anything-v2.cpp contributors
// Licensed under the Apache License, Version 2.0

#include "image.hpp"
#include "common.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstring>

namespace da {

Image load_image(const std::string& path) {
    Image img;
    int x, y, ch;
    uint8_t* pixels = stbi_load(path.c_str(), &x, &y, &ch, 0);
    if (!pixels) {
        DA_LOG("Failed to load image: %s", path.c_str());
        return img;
    }
    img.w = x;
    img.h = y;
    img.data.resize((size_t)x * y * 3);
    // stbi_load returns RGB/RGBA; convert to RGB
    if (ch == 4) {
        for (int i = 0; i < x * y; ++i) {
            img.data[i*3+0] = pixels[i*4+0];
            img.data[i*3+1] = pixels[i*4+1];
            img.data[i*3+2] = pixels[i*4+2];
        }
    } else if (ch == 3) {
        std::memcpy(img.data.data(), pixels, (size_t)x * y * 3);
    } else if (ch == 1) {
        for (int i = 0; i < x * y; ++i) {
            img.data[i*3+0] = img.data[i*3+1] = img.data[i*3+2] = pixels[i];
        }
    }
    stbi_image_free(pixels);
    return img;
}

bool save_depth(const std::string& path, const std::vector<uint16_t>& data, int32_t w, int32_t h) {
    // Save as 8-bit grayscale PNG (stb_image_write doesn't support 16-bit)
    // Scale uint16 [0, 65535] → uint8 [0, 255]
    std::vector<uint8_t> gray(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        gray[i] = (uint8_t)((data[i] * 255u) / 65535u);
    }
    return stbi_write_png(path.c_str(), w, h, 1, gray.data(), w) != 0;
}

bool save_image(const std::string& path, const std::vector<uint8_t>& data, int32_t w, int32_t h) {
    return stbi_write_png(path.c_str(), w, h, 3, data.data(), w * 3) != 0;
}

} // namespace da
