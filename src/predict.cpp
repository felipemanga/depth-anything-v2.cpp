// Copyright 2026 depth-anything-v2.cpp contributors
// Licensed under the Apache License, Version 2.0

#include "predict.hpp"
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "vision_ops.hpp"
#include "model.hpp"
#include "ggml.h"
#include "common.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace da {

static constexpr float IMAGENET_MEAN[3]  = {0.485f, 0.456f, 0.406f};
static constexpr float IMAGENET_STD[3]   = {0.229f, 0.224f, 0.225f};

// Profiling helper
static std::chrono::steady_clock::time_point g_perf_start = std::chrono::steady_clock::now();
static void perf_log(const char* section) {
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g_perf_start).count();
    DA_LOG("[PERF] %s: %.1fms", section, ms);
    g_perf_start = std::chrono::steady_clock::now();
}

// Cubic interpolation kernel (OpenCV a=-0.75)
static float cubic_kernel(float t) {
    float a = -0.75f;
    float at = std::abs(t);
    float at2 = at * at;
    float at3 = at2 * at;
    if (at <= 1.0f) return (a + 2.0f) * at3 - (a + 3.0f) * at2 + 1.0f;
    if (at < 2.0f) return a * at3 - 5.0f * a * at2 + 8.0f * a * at - 4.0f * a;
    return 0.0f;
}

// Constrain x to be a multiple of m, with optional min/max bounds.
// Matches Python's: floor to multiple, then clamp to [min_val, max_val].
static int constrain_to_multiple_of(float x, int m, int min_val = 0, int max_val = 0) {
    int y = (int)std::floor(x / (float)m) * m;
    if (min_val > 0 && y < min_val) {
        y = min_val;
    }
    if (max_val > 0 && y > max_val) {
        y = max_val;
    }
    return y;
}

PreprocessedImage preprocess(const Image& img, uint32_t target_size, uint32_t patch_size) {
    PreprocessedImage result;
    result.orig_h = img.h;
    result.orig_w = img.w;

    float scale_h = (float)target_size / (float)img.h;
    float scale_w = (float)target_size / (float)img.w;
    float scale = std::min(scale_h, scale_w);

    int new_h = constrain_to_multiple_of(scale * img.h, (int)patch_size);
    int new_w = constrain_to_multiple_of(scale * img.w, (int)patch_size);

    result.h = new_h;
    result.w = new_w;
    result.data.resize((size_t)3 * new_h * new_w);

    const float x_ratio = (float)img.w / (float)new_w;
    const float y_ratio = (float)img.h / (float)new_h;
    const uint8_t* src = img.data.data();
    float* dst = result.data.data();
    const int src_w = img.w;
    const int src_h = img.h;
    const int dst_w = new_w;
    const int dst_h = new_h;

    // Pre-allocated buffers for bicubic taps (avoids allocation in hot loop)
    alignas(16) float wx_buf[4];
    alignas(16) float wy_buf[4];
    alignas(16) int src_x[4];
    alignas(16) int src_y[4];

    // Process row-by-row — inspired by DeltaAI Blit row-pointer pattern
    for (int y = 0; y < dst_h; ++y) {
        // Pre-compute y weights & clamped source rows (4 bicubic taps)
        float sy = (y + 0.5f) * y_ratio - 0.5f;
        int y0 = (int)std::floor(sy);
        for (int j = -1; j <= 2; ++j) {
            wy_buf[j + 1] = cubic_kernel(sy - (y0 + j));
            src_y[j + 1] = std::max(0, std::min(y0 + j, src_h - 1));
        }

        // Destination row pointers (CHW layout)
        float* dst_r = dst + 0 * dst_h * dst_w + y * dst_w;
        float* dst_g = dst + 1 * dst_h * dst_w + y * dst_w;
        float* dst_b = dst + 2 * dst_h * dst_w + y * dst_w;

        for (int x = 0; x < dst_w; ++x) {
            // Pre-compute x weights & clamped source cols (4 bicubic taps)
            float sx = (x + 0.5f) * x_ratio - 0.5f;
            int x0 = (int)std::floor(sx);
            for (int i = -1; i <= 2; ++i) {
                wx_buf[i + 1] = cubic_kernel(sx - (x0 + i));
                src_x[i + 1] = std::max(0, std::min(x0 + i, src_w - 1));
            }

            // Accumulate RGB simultaneously across 4×4 bicubic neighborhood
            float vr = 0.0f, vg = 0.0f, vb = 0.0f;
            float wsum = 0.0f;

            for (int j = 0; j < 4; ++j) {
                float wy = wy_buf[j];
                const uint8_t* src_row = src + (size_t)src_y[j] * src_w * 3;
                for (int i = 0; i < 4; ++i) {
                    float w = wx_buf[i] * wy;
                    const uint8_t* px = src_row + src_x[i] * 3;
                    vr += w * px[0];
                    vg += w * px[1];
                    vb += w * px[2];
                    wsum += w;
                }
            }

            if (wsum > 1e-8f) {
                float inv = 1.0f / wsum;
                dst_r[x] = (vr * inv / 255.0f - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                dst_g[x] = (vg * inv / 255.0f - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                dst_b[x] = (vb * inv / 255.0f - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
            }
        }
    }

    return result;
}

static ggml_tensor* vit_block(ggml_context* ctx, ggml_tensor* x,
                               const ModelLoader& ml, int block_idx,
                               const DAConfig& cfg) {
    std::string prefix = "pretrained.blocks." + std::to_string(block_idx) + ".";

    ggml_tensor* norm1 = layer_norm(ctx, x,
        clone_weight(ctx, ml, (prefix + "norm1.weight").c_str()),
        clone_weight(ctx, ml, (prefix + "norm1.bias").c_str()));

    ggml_tensor* attn_out = self_attention(ctx, norm1, ml,
        (prefix + "attn.qkv.weight").c_str(), (prefix + "attn.qkv.bias").c_str(),
        (prefix + "attn.proj.weight").c_str(), (prefix + "attn.proj.bias").c_str(),
        cfg.num_heads);

    ggml_tensor* ls1 = layer_scale(ctx, attn_out,
        clone_weight(ctx, ml, (prefix + "ls1.gamma").c_str()));
    x = ggml_add(ctx, x, ls1);

    ggml_tensor* norm2 = layer_norm(ctx, x,
        clone_weight(ctx, ml, (prefix + "norm2.weight").c_str()),
        clone_weight(ctx, ml, (prefix + "norm2.bias").c_str()));

    ggml_tensor* mlp_out = mlp(ctx, norm2, ml,
        (prefix + "mlp.fc1.weight").c_str(), (prefix + "mlp.fc1.bias").c_str(),
        (prefix + "mlp.fc2.weight").c_str(), (prefix + "mlp.fc2.bias").c_str(),
        cfg.ffn_layer);

    ggml_tensor* ls2 = layer_scale(ctx, mlp_out,
        clone_weight(ctx, ml, (prefix + "ls2.gamma").c_str()));
    x = ggml_add(ctx, x, ls2);
    return x;
}

std::vector<float> predict(const ModelLoader& ml, const PreprocessedImage& input, int n_threads) {
    const DAConfig& cfg = ml.config();

    int p_h = input.h / (int)cfg.patch_size;
    int p_w = input.w / (int)cfg.patch_size;
    int num_patches = p_h * p_w;
    int embed = (int)cfg.embed_dim;

    // Convert CHW -> WHC for ggml conv2d (SIMD-accelerated)
    g_perf_start = std::chrono::steady_clock::now();
    std::vector<float> ggml_input((size_t)input.w * input.h * 3);
#if defined(__x86_64__) || defined(_M_X64)
    const int simd_n = (input.w / 4) * 4; // SSE: 4 floats per __m128
    for (int c = 0; c < 3; ++c) {
        const float* src_row = input.data.data() + c * (size_t)input.h * input.w;
        float* dst_row = ggml_input.data() + c * (size_t)input.w * input.h;
        for (int y = 0; y < input.h; ++y) {
            for (int x = 0; x < simd_n; x += 4) {
                _mm_storeu_ps(dst_row + x, _mm_loadu_ps(src_row + x));
            }
            // Scalar remainder
            for (int x = simd_n; x < input.w; ++x) {
                dst_row[x] = src_row[x];
            }
            src_row += input.w;
            dst_row += input.w;
        }
    }
#else
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < input.h; ++y) {
            for (int x = 0; x < input.w; ++x) {
                ggml_input[x + y * input.w + c * input.w * input.h] =
                    input.data[c * input.h * input.w + y * input.w + x];
            }
        }
    }
#endif

    // Pre-compute positional embeddings on host
    ensure_weights_realized(ml);
    ggml_tensor* pos_embed_raw = ml.tensor("pretrained.pos_embed");
    int N_train = (int)pos_embed_raw->ne[1];
    int sqrt_N = (int)std::round(std::sqrt((float)(N_train - 1)));

    // Get pos_embed data on host (works for both CPU and GPU backends)
    std::vector<float> pos_embed_host;
    weight_to_host_f32(ml, "pretrained.pos_embed", pos_embed_host);
    const float* pos_ptr = pos_embed_host.data();

    std::vector<float> cls_pos_data(embed);
    for (int d = 0; d < embed; ++d) {
        cls_pos_data[d] = pos_ptr[d];
    }

    std::vector<float> patch_pos_data((size_t)num_patches * embed);
    bool need_interp = (p_h != sqrt_N || p_w != sqrt_N);

    if (need_interp) {
        // Pre-compute bicubic weights & clamped indices
        float x_r = (float)sqrt_N / (float)p_w;
        float y_r = (float)sqrt_N / (float)p_h;
        struct Tap { float w; int idx; };
        std::vector<std::array<Tap, 4>> x_taps(p_w), y_taps(p_h);
        for (int x = 0; x < p_w; ++x) {
            float sx = (x + 0.5f) * x_r - 0.5f;
            int x0 = (int)std::floor(sx);
            float dx = sx - x0;
            for (int i = -1; i <= 2; ++i) {
                x_taps[x][i + 1] = { cubic_kernel(dx - i),
                                      std::max(0, std::min(x0 + i, sqrt_N - 1)) };
            }
        }
        for (int y = 0; y < p_h; ++y) {
            float sy = (y + 0.5f) * y_r - 0.5f;
            int y0 = (int)std::floor(sy);
            float dy = sy - y0;
            for (int j = -1; j <= 2; ++j) {
                y_taps[y][j + 1] = { cubic_kernel(dy - j),
                                      std::max(0, std::min(y0 + j, sqrt_N - 1)) };
            }
        }

        std::vector<float> grid_in((size_t)sqrt_N * sqrt_N);
        std::vector<float> grid_out((size_t)p_h * p_w);
        for (int d = 0; d < embed; ++d) {
            for (int y = 0; y < sqrt_N; ++y) {
                for (int x = 0; x < sqrt_N; ++x) {
                    grid_in[y * sqrt_N + x] = pos_ptr[(y * sqrt_N + x + 1) * embed + d];
                }
            }
            for (int y = 0; y < p_h; ++y) {
                const Tap* yt = &y_taps[y][0];
                for (int x = 0; x < p_w; ++x) {
                    const Tap* xt = &x_taps[x][0];
                    float val = 0.0f;
                    for (int j = 0; j < 4; ++j) {
                        float wy = yt[j].w;
                        const float* r = grid_in.data() + yt[j].idx * sqrt_N;
                        for (int i = 0; i < 4; ++i) {
                            val += r[xt[i].idx] * xt[i].w * wy;
                        }
                    }
                    grid_out[y * p_w + x] = val;
                }
            }
            for (int p = 0; p < num_patches; ++p) {
                patch_pos_data[p + d * num_patches] = grid_out[p];
            }
        }
    } else {
        for (int p = 0; p < num_patches; ++p) {
            for (int d = 0; d < embed; ++d) {
                patch_pos_data[p + d * num_patches] = pos_ptr[(p + 1) * embed + d];
            }
        }
    }

    // perf_log("cpu_prep (CHW+pos_embed)");

    // === FULL PIPELINE: ViT + DPT + Refinet + Output (single merged GPU graph) ===
    // All intermediate tensors stay on GPU — zero CPU round-trips.
    // cls token is skipped via ggml_view (offset by 1 row, zero-copy).
    int head_feat = (int)cfg.head_features;
    int head_feat_half = head_feat / 2;

    // Resize layer output sizes
    const std::vector<int32_t>& inter_layers = cfg.intermediate_layers;
    int n_inter = (int)inter_layers.size();
    std::vector<int> resized_h(n_inter), resized_w(n_inter);
    resized_h[0] = p_h * 4; resized_w[0] = p_w * 4;
    resized_h[1] = p_h * 2; resized_w[1] = p_w * 2;
    resized_h[2] = p_h;     resized_w[2] = p_w;
    resized_h[3] = (p_h - 1) / 2 + 1;
    resized_w[3] = (p_w - 1) / 2 + 1;

    uint32_t bilinear_corners = GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS;

    std::vector<float> depth_full;
    bool ok = da::run_graph(0, n_threads, [&](ggml_context* ctx) -> ggml_tensor* {
        // === ViT ENCODER ===
        ggml_tensor* x_img = graph_input_tensor(ctx, GGML_TYPE_F32, 4,
            (int64_t[]){input.w, input.h, 3, 1},
            ggml_input.data(), ggml_input.size() * sizeof(float));

        ggml_tensor* patches = conv2d(ctx, x_img, ml,
            "pretrained.patch_embed.proj.weight", "pretrained.patch_embed.proj.bias",
            (int)cfg.patch_size, (int)cfg.patch_size,
            (int)cfg.patch_size, (int)cfg.patch_size, 0, 0);
        ggml_tensor* patches_flat = ggml_reshape_2d(ctx, patches, (int64_t)num_patches, embed);

        ggml_tensor* cls_pos_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, embed);
        add_graph_input(cls_pos_input, cls_pos_data.data(), cls_pos_data.size() * sizeof(float));
        ggml_tensor* patch_pos_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t)num_patches, embed);
        add_graph_input(patch_pos_input, patch_pos_data.data(), patch_pos_data.size() * sizeof(float));

        ggml_tensor* cls_token = clone_weight(ctx, ml, "pretrained.cls_token");
        cls_token = ggml_reshape_2d(ctx, cls_token, 1, embed);
        ggml_tensor* cls_with_pos = ggml_add(ctx, cls_token, cls_pos_input);
        ggml_tensor* patches_with_pos = ggml_add(ctx, patches_flat, patch_pos_input);
        ggml_tensor* x = ggml_concat(ctx, cls_with_pos, patches_with_pos, 0);

        ggml_tensor* fnorm_w = clone_weight(ctx, ml, "pretrained.norm.weight");
        ggml_tensor* fnorm_b = clone_weight(ctx, ml, "pretrained.norm.bias");

        // Run all blocks, capture 4 intermediates as GPU tensors (cls skipped via view)
        ggml_tensor* inter_patch[4];  // patch-only tensors, cls skipped
        int inter_idx = 0;

        for (int i = 0; i < (int)cfg.depth; ++i) {
            x = vit_block(ctx, x, ml, i, cfg);

            for (int ci = 0; ci < n_inter; ++ci) {
                if (i == inter_layers[ci]) {
                    ggml_tensor* normed = layer_norm(ctx, x, fnorm_w, fnorm_b);
                    // Skip cls token (row 0) via view: offset by embed bytes, shape [num_patches, embed]
                    // normed shape: [seq, embed] = [num_patches+1, embed]
                    // Skip cls token (seq index 0) via view, then make contiguous for reshape.
                    ggml_tensor* patch_view = ggml_view_2d(ctx, normed, (int64_t)num_patches, normed->ne[1],
                        normed->nb[1], normed->nb[0]);
                    inter_patch[ci] = ggml_cont(ctx, patch_view);
                    inter_idx++;
                    break;
                }
            }
        }

        // === DPT HEAD: project + resize + layer_rn for all 4 features ===
        ggml_tensor* layer_rn[4];
        for (int i = 0; i < n_inter; ++i) {
            ggml_tensor* feat = ggml_reshape_4d(ctx, inter_patch[i], (int64_t)p_w, (int64_t)p_h, embed, 1);

            std::string proj_w = "depth_head.projects." + std::to_string(i) + ".weight";
            std::string proj_b = "depth_head.projects." + std::to_string(i) + ".bias";
            feat = conv2d(ctx, feat, ml, proj_w.c_str(), proj_b.c_str(), 1, 1, 1, 1, 0, 0);

            if (i == 0) {
                feat = conv_transpose2d(ctx, feat, ml,
                    "depth_head.resize_layers.0.weight", "depth_head.resize_layers.0.bias",
                    4, 4, 4, 4);
            } else if (i == 1) {
                feat = conv_transpose2d(ctx, feat, ml,
                    "depth_head.resize_layers.1.weight", "depth_head.resize_layers.1.bias",
                    2, 2, 2, 2);
            } else if (i == 2) {
                // Identity
            } else if (i == 3) {
                feat = conv2d(ctx, feat, ml,
                    "depth_head.resize_layers.3.weight", "depth_head.resize_layers.3.bias",
                    3, 3, 2, 2, 1, 1);
            }

            std::string lrn_w = "depth_head.scratch.layer" + std::to_string(i+1) + "_rn.weight";
            feat = conv2d(ctx, feat, ml, lrn_w.c_str(), nullptr, 3, 3, 1, 1, 1, 1);
            layer_rn[i] = feat;
        }

        // === REFINET BLOCKS ===
        ggml_tensor* rn = layer_rn[3];
        rn = residual_conv_unit(ctx, rn, ml,
            "depth_head.scratch.refinenet4.resConfUnit2.conv1.weight",
            "depth_head.scratch.refinenet4.resConfUnit2.conv1.bias",
            "depth_head.scratch.refinenet4.resConfUnit2.conv2.weight",
            "depth_head.scratch.refinenet4.resConfUnit2.conv2.bias");
        rn = ggml_interpolate(ctx, rn, resized_w[2], resized_h[2], head_feat, 1, bilinear_corners);
        rn = conv2d(ctx, rn, ml,
            "depth_head.scratch.refinenet4.out_conv.weight",
            "depth_head.scratch.refinenet4.out_conv.bias",
            1, 1, 1, 1, 0, 0);

        for (int stage = 3; stage >= 1; --stage) {
            int skip_idx = stage - 1;
            ggml_tensor* skip = layer_rn[skip_idx];
            std::string prefix = "depth_head.scratch.refinenet" + std::to_string(stage) + ".";
            ggml_tensor* res = residual_conv_unit(ctx, skip, ml,
                (prefix + "resConfUnit1.conv1.weight").c_str(),
                (prefix + "resConfUnit1.conv1.bias").c_str(),
                (prefix + "resConfUnit1.conv2.weight").c_str(),
                (prefix + "resConfUnit1.conv2.bias").c_str());
            rn = ggml_add(ctx, rn, res);
            rn = residual_conv_unit(ctx, rn, ml,
                (prefix + "resConfUnit2.conv1.weight").c_str(),
                (prefix + "resConfUnit2.conv1.bias").c_str(),
                (prefix + "resConfUnit2.conv2.weight").c_str(),
                (prefix + "resConfUnit2.conv2.bias").c_str());

            int out_h = (stage > 1) ? resized_h[stage - 2] : resized_h[0] * 2;
            int out_w = (stage > 1) ? resized_w[stage - 2] : resized_w[0] * 2;
            rn = ggml_interpolate(ctx, rn, out_w, out_h, head_feat, 1, bilinear_corners);
            rn = conv2d(ctx, rn, ml,
                (prefix + "out_conv.weight").c_str(),
                (prefix + "out_conv.bias").c_str(),
                1, 1, 1, 1, 0, 0);
        }

        // === OUTPUT CONVS + FINAL RESIZE ===
        rn = conv2d(ctx, rn, ml,
            "depth_head.scratch.output_conv1.weight",
            "depth_head.scratch.output_conv1.bias",
            3, 3, 1, 1, 1, 1);

        int pre_up_h = resized_h[0] * 2;
        int pre_up_w = resized_w[0] * 2;
        rn = ggml_interpolate(ctx, rn, pre_up_w * 2, pre_up_h * 2, head_feat_half, 1, bilinear_corners);

        rn = conv2d(ctx, rn, ml,
            "depth_head.scratch.output_conv2.0.weight",
            "depth_head.scratch.output_conv2.0.bias",
            3, 3, 1, 1, 1, 1);
        rn = ggml_relu(ctx, rn);
        rn = conv2d(ctx, rn, ml,
            "depth_head.scratch.output_conv2.2.weight",
            "depth_head.scratch.output_conv2.2.bias",
            1, 1, 1, 1, 0, 0);
        rn = ggml_relu(ctx, rn);

        rn = ggml_interpolate(ctx, rn, input.orig_w, input.orig_h, 1, 1, bilinear_corners);
        return ggml_reshape_1d(ctx, rn, (int64_t)input.orig_w * input.orig_h);
    }, depth_full);

    if (!ok) { DA_LOG("predict: full pipeline GPU failed"); return {}; }

    // perf_log("full_pipeline (GPU)");
    return depth_full;
}

std::vector<uint16_t> depth_to_uint16(const std::vector<float>& depth) {
#if defined(__x86_64__) || defined(_M_X64)
    // SIMD min/max reduction
    __m128 v_min = _mm_set1_ps(depth[0]);
    __m128 v_max = _mm_set1_ps(depth[0]);
    const size_t nvec = depth.size() / 4;
    for (size_t i = 0; i < nvec; ++i) {
        __m128 v = _mm_loadu_ps(depth.data() + i * 4);
        v_min = _mm_min_ps(v_min, v);
        v_max = _mm_max_ps(v_max, v);
    }
    // Horizontal min/max
    __m128 t = _mm_min_ps(v_min, _mm_shuffle_ps(v_min, v_min, 0x4E));
    v_min = _mm_min_ps(t, _mm_shuffle_ps(t, t, 0xB1));
    t = _mm_max_ps(v_max, _mm_shuffle_ps(v_max, v_max, 0x4E));
    v_max = _mm_max_ps(t, _mm_shuffle_ps(t, t, 0xB1));
    float d_min = _mm_cvtss_f32(v_min);
    float d_max = _mm_cvtss_f32(v_max);
#else
    float d_min = depth[0], d_max = depth[0];
    for (float v : depth) {
        d_min = std::min(d_min, v);
        d_max = std::max(d_max, v);
    }
#endif
    float range = d_max - d_min;
    if (range < 1e-6f) range = 1.0f;

    std::vector<uint16_t> result(depth.size());
    float scale = 65535.0f / range;
    float off = -d_min * scale;
#if defined(__x86_64__) || defined(_M_X64)
    __m128 v_scale = _mm_set1_ps(scale);
    __m128 v_off = _mm_set1_ps(off);
    __m128 v_zero = _mm_setzero_ps();
    __m128 v_65535 = _mm_set1_ps(65535.0f);
    size_t i = 0;
    for (size_t iv = 0; iv < nvec; ++iv) {
        __m128 v = _mm_loadu_ps(depth.data() + iv * 4);
        v = _mm_add_ps(_mm_mul_ps(v, v_scale), v_off);
        // Clamp to [0, 65535] to prevent overflow
        v = _mm_max_ps(v, v_zero);
        v = _mm_min_ps(v, v_65535);
        __m128i vi = _mm_cvtps_epi32(v);
        uint32_t tmp[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), vi);
        result[i++] = static_cast<uint16_t>(tmp[0]);
        result[i++] = static_cast<uint16_t>(tmp[1]);
        result[i++] = static_cast<uint16_t>(tmp[2]);
        result[i++] = static_cast<uint16_t>(tmp[3]);
    }
    // Scalar remainder
    for (size_t ri = nvec * 4; ri < depth.size(); ++ri) {
        float val = depth[ri] * scale + off;
        val = std::max(0.0f, std::min(65535.0f, val));
        result[ri] = static_cast<uint16_t>(val);
    }
#else
    for (size_t i = 0; i < depth.size(); ++i) {
        float val = depth[i] * scale + off;
        val = std::max(0.0f, std::min(65535.0f, val));
        result[i] = static_cast<uint16_t>(val);
    }
#endif
    return result;
}

} // namespace da
