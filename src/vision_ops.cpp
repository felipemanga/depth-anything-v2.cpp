// Copyright 2026 depth-anything-v2.cpp contributors
// Licensed under the Apache License, Version 2.0

#include "vision_ops.hpp"
#include "backend.hpp"
#include "ggml.h"
#include "common.hpp"
#include <cassert>
#include <cmath>

namespace da {

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
    // Use GPU-resident weight tensor directly (like linear() does).
    // GGUF stores conv weights as [kw, kh, cin, cout] after ggml shape reversal,
    // which is exactly what ggml_conv_2d expects.
    ggml_tensor* W = clone_weight(ctx, ml, w_name);
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
    // Use GPU-resident weight tensor directly.
    ggml_tensor* W = clone_weight(ctx, ml, w_name);
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
