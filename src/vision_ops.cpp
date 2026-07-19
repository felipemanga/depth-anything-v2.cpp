// Copyright 2026 depth-anything-v2.cpp contributors
// Licensed under the Apache License, Version 2.0

#include "vision_ops.hpp"
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "ggml.h"
#include "common.hpp"
#include <cmath>
#include <cassert>
#include <cstring>
#include <vector>
#include <omp.h>

namespace da {

namespace {
// Cached host copies of conv weights: name -> contiguous float data.
std::unordered_map<std::string, std::vector<float>> g_conv2d_cache;
std::mutex g_conv2d_mutex;
}

std::vector<float>& get_transposed_conv2d_weight(const ModelLoader& ml, const std::string& name,
                                                     int64_t ne[4]) {
    std::lock_guard<std::mutex> lock(g_conv2d_mutex);

    ggml_tensor* src = ml.tensor(name.c_str());
    assert(src && "conv2d weight not found");

    assert(ggml_n_dims(src) == 4 && "conv2d weight must be 4D");

    // src->ne is ggml order [kw,kh,cin,cout] after GGML shape reversal
    int64_t kw = src->ne[0], kh = src->ne[1], cin_ = src->ne[2], cout_ = src->ne[3];
    ne[0] = kw; ne[1] = kh; ne[2] = cin_; ne[3] = cout_;

    auto it = g_conv2d_cache.find(name);
    if (it != g_conv2d_cache.end()) {
        return it->second;
    }

    std::vector<float> src_data;
    weight_to_host_f32(ml, name.c_str(), src_data);

    // ggml tensor memory for these conv weights already matches expected layout
    // [kw, kh, cin, cout] for ggml_conv_2d, so we only cache a host copy.
    auto& dst = g_conv2d_cache[name];
    dst = std::move(src_data);
    return dst;
}

ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x,
                    const ModelLoader& ml, const char* w_name, const char* b_name) {
    // GGUF stores weight in ggml order: [in_features, out_features]
    // ggml_mul_mat(a, b) = a^T @ b
    // x: [seq, in_features], x_t: [in_features, seq]
    // W: [in_features, out_features]
    // ggml_mul_mat(W, x_t) = W^T @ x_t = [out_features, in_features] @ [in_features, seq] = [out_features, seq]
    ggml_tensor* W = clone_weight(ctx, ml, w_name);
    ggml_tensor* x_t = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor* y = ggml_mul_mat(ctx, W, x_t);
    y = ggml_cont(ctx, ggml_transpose(ctx, y));
    ggml_tensor* b = clone_weight_opt(ctx, ml, b_name);
    if (b) {
        ggml_tensor* b_2d = ggml_reshape_2d(ctx, b, 1, ggml_nelements(b));
        y = ggml_add(ctx, y, b_2d);
    }
    return y;
}

ggml_tensor* conv2d(ggml_context* ctx, ggml_tensor* x,
                    const ModelLoader& ml, const char* w_name, const char* b_name,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w) {
    // We use a cached host copy of [kw, kh, cin, cout] weights.
    int64_t ggml_ne[4];
    std::vector<float>& wdata = get_transposed_conv2d_weight(ml, w_name, ggml_ne);

    ggml_tensor* W = graph_input_tensor(ctx, GGML_TYPE_F32, 4, ggml_ne,
        wdata.data(), wdata.size() * sizeof(float));

    ggml_tensor* y = ggml_conv_2d(ctx, W, x, stride_w, stride_h, pad_w, pad_h, 1, 1);

    {
        ggml_tensor* b = clone_weight_opt(ctx, ml, b_name);
        if (b) {
            ggml_tensor* b_4d = ggml_reshape_4d(ctx, b, 1, 1, ggml_nelements(b), 1);
            y = ggml_add(ctx, y, b_4d);
        } else if (b_name) {
            DA_LOG("conv2d: bias %s not found", b_name);
        }
    }
    return y;
}

// Host-side bilinear interpolation (align_corners=False)
void bilinear_upscale_host(const float* input, float* output,
                            int in_w, int in_h, int c, int batch,
                            int out_w, int out_h) {
    float x_ratio = (float)in_w / (float)out_w;
    float y_ratio = (float)in_h / (float)out_h;

    for (int n = 0; n < batch; ++n) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float sx = (ow + 0.5f) * x_ratio - 0.5f;
                float sy = (oh + 0.5f) * y_ratio - 0.5f;

                int x0 = (int)std::floor(sx);
                int y0 = (int)std::floor(sy);
                int x1 = std::min(x0 + 1, in_w - 1);
                int y1 = std::min(y0 + 1, in_h - 1);
                x0 = std::max(0, x0); y0 = std::max(0, y0);

                float dx = sx - x0;
                float dy = sy - y0;

                for (int ch = 0; ch < c; ++ch) {
                    size_t idx00 = (size_t)x0 + (size_t)y0 * in_w + (size_t)ch * in_w * in_h + (size_t)n * in_w * in_h * c;
                    size_t idx10 = (size_t)x1 + (size_t)y0 * in_w + (size_t)ch * in_w * in_h + (size_t)n * in_w * in_h * c;
                    size_t idx01 = (size_t)x0 + (size_t)y1 * in_w + (size_t)ch * in_w * in_h + (size_t)n * in_w * in_h * c;
                    size_t idx11 = (size_t)x1 + (size_t)y1 * in_w + (size_t)ch * in_w * in_h + (size_t)n * in_w * in_h * c;
                    float v00 = input[idx00];
                    float v10 = input[idx10];
                    float v01 = input[idx01];
                    float v11 = input[idx11];
                    float val = v00*(1-dx)*(1-dy) + v10*dx*(1-dy) + v01*(1-dx)*dy + v11*dx*dy;
                    size_t oidx = (size_t)ow + (size_t)oh * out_w + (size_t)ch * out_w * out_h + (size_t)n * out_w * out_h * c;
                    output[oidx] = val;
                }
            }
        }
    }
}

// Host-side bilinear interpolation (align_corners=True), OpenMP-parallelized
// Used by DPT refinenet blocks
void bilinear_upscale_host_corners(const float* input, float* output,
                                     int in_w, int in_h, int c, int batch,
                                     int out_w, int out_h) {
    // Precompute source coordinates for each output row
    std::vector<int> sx_floor(out_h), sy_floor(out_h);
    std::vector<float> sdx(out_h), sdy(out_h);
    std::vector<int> x0s(out_h), x1s(out_h), y0s(out_h), y1s(out_h);

    for (int oh = 0; oh < out_h; ++oh) {
        float sy = (out_h > 1) ? ((float)oh / (out_h - 1)) * (in_h - 1) : 0.0f;
        int y0 = (int)std::floor(sy);
        int y1 = std::min(y0 + 1, in_h - 1);
        y0 = std::max(0, y0);
        sy_floor[oh] = y0; y1s[oh] = y1; sdy[oh] = sy - y0;
    }

#pragma omp parallel for collapse(2) schedule(static)
    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            float sx = (out_w > 1) ? ((float)ow / (out_w - 1)) * (in_w - 1) : 0.0f;
            int x0 = std::max(0, (int)std::floor(sx));
            int x1 = std::min(x0 + 1, in_w - 1);
            float dx = sx - x0;

            int y0 = sy_floor[oh], y1 = y1s[oh];
            float dy = sdy[oh];

            for (int ch = 0; ch < c; ++ch) {
                size_t ci00 = (size_t)x0 + (size_t)y0 * in_w + (size_t)ch * in_w * in_h;
                size_t ci10 = (size_t)x1 + (size_t)y0 * in_w + (size_t)ch * in_w * in_h;
                size_t ci01 = (size_t)x0 + (size_t)y1 * in_w + (size_t)ch * in_w * in_h;
                size_t ci11 = (size_t)x1 + (size_t)y1 * in_w + (size_t)ch * in_w * in_h;
                float val = input[ci00]*(1-dx)*(1-dy) + input[ci10]*dx*(1-dy) +
                            input[ci01]*(1-dx)*dy + input[ci11]*dx*dy;
                output[ow + (size_t)oh * out_w + (size_t)ch * out_w * out_h] = val;
            }
        }
    }
}

ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* weight, ggml_tensor* bias) {
    // ggml_norm normalizes over dimension 0. Our activations are [seq, embed],
    // so transpose to [embed, seq], normalize/apply affine over embed, then
    // transpose back.
    ggml_tensor* x_t = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor* y = ggml_norm(ctx, x_t, 1e-6f);

    ggml_tensor* w = ggml_reshape_2d(ctx, weight, ggml_nelements(weight), 1);
    ggml_tensor* b = ggml_reshape_2d(ctx, bias, ggml_nelements(bias), 1);
    y = ggml_mul(ctx, y, w);
    y = ggml_add(ctx, y, b);

    return ggml_cont(ctx, ggml_transpose(ctx, y));
}

ggml_tensor* gelu(ggml_context* ctx, ggml_tensor* x) {
    return ggml_gelu_erf(ctx, x);
}

ggml_tensor* silu(ggml_context* ctx, ggml_tensor* x) {
    return ggml_silu(ctx, x);
}

ggml_tensor* relu(ggml_context* ctx, ggml_tensor* x) {
    return ggml_relu(ctx, x);
}

ggml_tensor* self_attention(ggml_context* ctx, ggml_tensor* x,
                            const ModelLoader& ml,
                            const char* qkv_w_name, const char* qkv_b_name,
                            const char* proj_w_name, const char* proj_b_name,
                            uint32_t num_heads) {
    int seq_len = (int)x->ne[0];
    int embed_dim = (int)x->ne[1];
    int head_dim = (int)(embed_dim / num_heads);

    // QKV projection: [seq, 3*embed]
    ggml_tensor* qkv = linear(ctx, x, ml, qkv_w_name, qkv_b_name);

    // Work in [3*embed, seq] so splitting Q/K/V by feature is contiguous.
    ggml_tensor* qkv_t = ggml_cont(ctx, ggml_transpose(ctx, qkv));
    size_t col_nb = qkv_t->nb[1];
    size_t elem_size = ggml_element_size(qkv_t);

    ggml_tensor* q = ggml_view_2d(ctx, qkv_t, embed_dim, seq_len, col_nb, 0);
    ggml_tensor* k = ggml_view_2d(ctx, qkv_t, embed_dim, seq_len, col_nb,
                                  (size_t)embed_dim * elem_size);
    ggml_tensor* v = ggml_view_2d(ctx, qkv_t, embed_dim, seq_len, col_nb,
                                  (size_t)2 * embed_dim * elem_size);
    q = ggml_cont(ctx, q);
    k = ggml_cont(ctx, k);
    v = ggml_cont(ctx, v);

    // [embed, seq] -> [head_dim, seq, heads]
    ggml_tensor* q_h = ggml_reshape_3d(ctx, q, head_dim, (int)num_heads, seq_len);
    ggml_tensor* k_h = ggml_reshape_3d(ctx, k, head_dim, (int)num_heads, seq_len);
    ggml_tensor* v_h = ggml_reshape_3d(ctx, v, head_dim, (int)num_heads, seq_len);
    q_h = ggml_cont(ctx, ggml_permute(ctx, q_h, 0, 2, 1, 3));
    k_h = ggml_cont(ctx, ggml_permute(ctx, k_h, 0, 2, 1, 3));
    v_h = ggml_cont(ctx, ggml_permute(ctx, v_h, 0, 2, 1, 3));

    // Flash attention: [head_dim, seq, heads] -> [head_dim, heads, seq] (no full attn matrix)
    float scale = 1.0f / std::sqrt((float)head_dim);
    ggml_tensor* out_h = ggml_flash_attn_ext(ctx, q_h, k_h, v_h, nullptr, scale, 0.0f, 0.0f);

    // out_h: [head_dim, heads, seq] -> reshape to [embed, seq] -> transpose to [seq, embed]
    ggml_tensor* out = ggml_reshape_2d(ctx, out_h, embed_dim, seq_len);
    out = ggml_cont(ctx, ggml_transpose(ctx, out));

    out = linear(ctx, out, ml, proj_w_name, proj_b_name);
    return out;
}

ggml_tensor* mlp(ggml_context* ctx, ggml_tensor* x,
                 const ModelLoader& ml,
                 const char* fc1_w_name, const char* fc1_b_name,
                 const char* fc2_w_name, const char* fc2_b_name,
                 const std::string& act_layer) {
    ggml_tensor* y = linear(ctx, x, ml, fc1_w_name, fc1_b_name);
    y = (act_layer == "swiglu" || act_layer == "swiglufused") ? silu(ctx, y) : gelu(ctx, y);
    y = linear(ctx, y, ml, fc2_w_name, fc2_b_name);
    return y;
}

ggml_tensor* layer_scale(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gamma) {
    // Reshape gamma from [D] to [1, D] for broadcasting with [seq, D]
    ggml_tensor* g = ggml_reshape_2d(ctx, gamma, 1, ggml_nelements(gamma));
    return ggml_mul(ctx, x, g);
}

ggml_tensor* conv_transpose2d(ggml_context* ctx, ggml_tensor* x,
                                const ModelLoader& ml, const char* w_name, const char* b_name,
                                int kernel_h, int kernel_w,
                                int stride_h, int stride_w) {
    // GGUF stores ConvTranspose2d weight in ggml order: [kw, kh, cout, cin]
    // ggml_conv_transpose_2d_p0 expects: [kw, kh, cout, cin] ← matches!
    ggml_tensor* src = ml.tensor(w_name);
    assert(src && "conv_transpose2d weight not found");

    int64_t ggml_ne[4] = {src->ne[0], src->ne[1], src->ne[2], src->ne[3]};

    // Use cached transposed weight (no actual transpose needed, just passthrough)
    std::vector<float>& wdata = get_transposed_conv2d_weight(ml, w_name, ggml_ne);

    ggml_tensor* W = graph_input_tensor(ctx, GGML_TYPE_F32, 4, ggml_ne,
        wdata.data(), wdata.size() * sizeof(float));

    int stride = stride_w;
    ggml_tensor* y = ggml_conv_transpose_2d_p0(ctx, W, x, stride);

    ggml_tensor* b = clone_weight_opt(ctx, ml, b_name);
    if (b) {
        ggml_tensor* b_4d = ggml_reshape_4d(ctx, b, 1, 1, ggml_nelements(b), 1);
        y = ggml_add(ctx, y, b_4d);
    }
    return y;
}

ggml_tensor* residual_conv_unit(ggml_context* ctx, ggml_tensor* x,
                                  const ModelLoader& ml,
                                  const char* conv1_w_name, const char* conv1_b_name,
                                  const char* conv2_w_name, const char* conv2_b_name) {
    ggml_tensor* out = ggml_relu(ctx, x);
    out = conv2d(ctx, out, ml, conv1_w_name, conv1_b_name, 3, 3, 1, 1, 1, 1);
    out = ggml_relu(ctx, out);
    out = conv2d(ctx, out, ml, conv2_w_name, conv2_b_name, 3, 3, 1, 1, 1, 1);
    out = ggml_add(ctx, out, x);
    return out;
}

} // namespace da
