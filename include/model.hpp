// Copyright 2026 depth-anything-v2.cpp contributors
// Licensed under the Apache License, Version 2.0

#pragma once
#include <atomic>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

struct ggml_tensor;
struct ggml_context;
struct gguf_context;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer* ggml_backend_buffer_t;
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;

namespace da {

// Depth-Anything-V2 model config, read from GGUF metadata.
struct DAConfig {
    std::string arch;       // "vits", "vitb", "vitl", "vitg"
    uint32_t embed_dim = 0;
    uint32_t depth = 0;     // number of transformer blocks
    uint32_t num_heads = 0;
    uint32_t mlp_ratio = 4;
    uint32_t patch_size = 14;
    uint32_t input_size = 518;
    float init_values = 1.0f;
    std::string ffn_layer = "mlp"; // "mlp" or "swiglufused"
    float interpolate_offset = 0.1f;
    bool interpolate_antialias = false;

    // DPT head
    uint32_t head_features = 256;
    std::vector<uint32_t> out_channels = {256, 512, 1024, 1024};
    // Intermediate layer indices for feature extraction
    std::vector<int32_t> intermediate_layers = {2, 5, 8, 11};
};

class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader();
    bool load(const std::string& path);
    const DAConfig& config() const { return cfg_; }
    ggml_tensor* tensor(const std::string& name) const; // nullptr if absent
    ggml_context* ggml_ctx() const { return ctx_; }

    bool realize_weights(ggml_backend_t backend);
    bool weights_realized() const { return weights_realized_; }

private:
    DAConfig cfg_;
    gguf_context* gguf_ = nullptr;
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer_t weights_buf_ = nullptr;
    std::atomic<bool> weights_realized_{false};
    ggml_context* device_ctx_ = nullptr;
    std::unordered_map<std::string, ggml_tensor*> tensors_;
};

} // namespace da
