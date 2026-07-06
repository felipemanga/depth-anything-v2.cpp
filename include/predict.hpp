// Copyright 2026 depth-anything-v2.cpp contributors
// Licensed under the Apache License, Version 2.0

#pragma once
#include "model.hpp"
#include "image.hpp"
#include <vector>

namespace da {

// High-level inference API.
// Runs the full Depth-Anything-V2 forward pass on a single image.

// Preprocess image: resize with aspect ratio, pad to multiple of 14, normalize.
// Returns CHW f32 tensor data (C=3, H, W) and the preprocessed dimensions.
struct PreprocessedImage {
    std::vector<float> data; // CHW, float32
    int32_t h = 0;
    int32_t w = 0;
    int32_t orig_h = 0; // original image height
    int32_t orig_w = 0; // original image width
};

PreprocessedImage preprocess(const Image& img, uint32_t target_size, uint32_t patch_size);

// Run inference: takes preprocessed image, returns per-pixel depth values (H x W).
// Depth values are raw (not normalized to 0-255).
std::vector<float> predict(const ModelLoader& ml, const PreprocessedImage& input, int n_threads);

// Post-process: bilinear upscale depth map to original image size.
std::vector<float> postprocess_depth(const std::vector<float>& depth,
                                     int32_t depth_h, int32_t depth_w,
                                     int32_t target_h, int32_t target_w);

// Convert raw depth to 16-bit uint.
std::vector<uint16_t> depth_to_uint16(const std::vector<float>& depth);

} // namespace da
