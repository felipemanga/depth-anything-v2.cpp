// Copyright 2026 depth-anything-v2.cpp contributors
// Licensed under the Apache License, Version 2.0

#pragma once
#include "model.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>

struct ggml_context;
struct ggml_tensor;

namespace da {

// Linear (fully-connected): y = x @ W^T + b
// W shape: [out_features, in_features] (row-major, as in PyTorch state dict)
ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x,
                    const ModelLoader& ml, const char* w_name, const char* b_name);

// Transpose conv2d weight from PyTorch [cout,cin,kh,kw] to ggml [kw,kh,cin,cout]
// Caches transposed data. Returns ggml shape [kw,kh,cin,cout].
std::vector<float>& get_transposed_conv2d_weight(const ModelLoader& ml, const std::string& name,
                                                  int64_t ne[4]);

// Conv2d: y = conv2d(x, W, b, kernel, stride, padding)
// W shape: [out_ch, in_ch/groups, kh, kw] — automatically transposed to ggml layout
ggml_tensor* conv2d(ggml_context* ctx, ggml_tensor* x,
                    const ModelLoader& ml, const char* w_name, const char* b_name,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w);

// ConvTranspose2d (deconv): upsampling convolution
// W shape: [in_ch, out_ch/groups, kh, kw] — automatically transposed to ggml layout
ggml_tensor* conv_transpose2d(ggml_context* ctx, ggml_tensor* x,
                               const ModelLoader& ml, const char* w_name, const char* b_name,
                               int kernel_h, int kernel_w,
                               int stride_h, int stride_w);

// LayerNorm: LN(x, weight, bias, eps=1e-6)
ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* weight, ggml_tensor* bias);

// GELU activation
ggml_tensor* gelu(ggml_context* ctx, ggml_tensor* x);

// SiLU activation
ggml_tensor* silu(ggml_context* ctx, ggml_tensor* x);

// ReLU activation
ggml_tensor* relu(ggml_context* ctx, ggml_tensor* x);

// Multi-head self-attention (standard ViT attention)
// QKV is a single linear: W_qkv [3*d, d], b_qkv [3*d]
// Proj: W_proj [d, d], b_proj [d]
ggml_tensor* self_attention(ggml_context* ctx, ggml_tensor* x,
                            const ModelLoader& ml,
                            const char* qkv_w_name, const char* qkv_b_name,
                            const char* proj_w_name, const char* proj_b_name,
                            uint32_t num_heads);

// MLP: fc1 -> act -> fc2
ggml_tensor* mlp(ggml_context* ctx, ggml_tensor* x,
                 const ModelLoader& ml,
                 const char* fc1_w_name, const char* fc1_b_name,
                 const char* fc2_w_name, const char* fc2_b_name,
                 const std::string& act_layer);

// LayerScale: element-wise multiplication by gamma
ggml_tensor* layer_scale(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gamma);

// Bilinear upscale: interpolate to target size
ggml_tensor* bilinear_upscale(ggml_context* ctx, ggml_tensor* x, int target_h, int target_w);

// ResidualConvUnit: act -> conv1 -> act -> conv2 -> + residual
// Used in DPT head refinenet blocks
ggml_tensor* residual_conv_unit(ggml_context* ctx, ggml_tensor* x,
                                 const ModelLoader& ml,
                                  const char* conv1_w_name, const char* conv1_b_name,
                                  const char* conv2_w_name, const char* conv2_b_name);

} // namespace da
