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
    // keep_aspect_ratio=True: scale so larger dimension = target_size
    float scale = std::min(scale_h, scale_w);

    // Python: constrain_to_multiple_of(scale * dim, patch_size) -- floor to multiple
    int new_h = constrain_to_multiple_of(scale * img.h, (int)patch_size);
    int new_w = constrain_to_multiple_of(scale * img.w, (int)patch_size);

    result.h = new_h;
    result.w = new_w;
    result.data.resize((size_t)3 * new_h * new_w);

    float x_ratio = (float)img.w / (float)new_w;
    float y_ratio = (float)img.h / (float)new_h;

    // Bicubic interpolation for image resize (matches cv2.INTER_CUBIC)
    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            float sx = (x + 0.5f) * x_ratio - 0.5f;
            float sy = (y + 0.5f) * y_ratio - 0.5f;

            int x0 = (int)std::floor(sx);
            int y0 = (int)std::floor(sy);

            for (int c = 0; c < 3; ++c) {
                float val = 0.0f;
                float wsum = 0.0f;
                // Bicubic: sample 4x4 neighborhood
                for (int j = -1; j <= 2; ++j) {
                    for (int i = -1; i <= 2; ++i) {
                        int nx = std::max(0, std::min(x0 + i, img.w - 1));
                        int ny = std::max(0, std::min(y0 + j, img.h - 1));
                        float wx = cubic_kernel(sx - (x0 + i));
                        float wy = cubic_kernel(sy - (y0 + j));
                        float w = wx * wy;
                        val += w * img.data[((size_t)ny * img.w + nx) * 3 + c] / 255.0f;
                        wsum += w;
                    }
                }
                if (wsum > 1e-8f) val /= wsum; // normalize weights
                val = (val - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                result.data[c * new_h * new_w + y * new_w + x] = val;
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

// Build entire ViT encoder in one pass, capturing intermediate block outputs.
// The build lambda calls capture_graph_output() for all but the last intermediate,
// and returns the last intermediate as the main output.
static ggml_tensor* build_vit_all_blocks(ggml_context* ctx,
    const ModelLoader& ml, const DAConfig& cfg,
    int num_patches, int embed,
    const std::vector<float>& ggml_input,
    const std::vector<float>& cls_pos_data,
    const std::vector<float>& patch_pos_data,
    int input_w, int input_h,
    const std::vector<int32_t>& inter_layers,
    std::vector<std::vector<float>*> & captureDsts) {

    // Patch embedding
    ggml_tensor* x_img = graph_input_tensor(ctx, GGML_TYPE_F32, 4,
        (int64_t[]){input_w, input_h, 3, 1},
        ggml_input.data(), ggml_input.size() * sizeof(float));

    ggml_tensor* patches = conv2d(ctx, x_img, ml,
        "pretrained.patch_embed.proj.weight", "pretrained.patch_embed.proj.bias",
        (int)cfg.patch_size, (int)cfg.patch_size,
        (int)cfg.patch_size, (int)cfg.patch_size, 0, 0);

    ggml_tensor* patches_flat = ggml_reshape_2d(ctx, patches, (int64_t)num_patches, embed);

    // Positional embeddings
    ggml_tensor* cls_pos_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, embed);
    add_graph_input(cls_pos_input, cls_pos_data.data(), cls_pos_data.size() * sizeof(float));
    ggml_tensor* patch_pos_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t)num_patches, embed);
    add_graph_input(patch_pos_input, patch_pos_data.data(), patch_pos_data.size() * sizeof(float));

    // CLS + patches
    ggml_tensor* cls_token = clone_weight(ctx, ml, "pretrained.cls_token");
    cls_token = ggml_reshape_2d(ctx, cls_token, 1, embed);
    ggml_tensor* cls_with_pos = ggml_add(ctx, cls_token, cls_pos_input);
    ggml_tensor* patches_with_pos = ggml_add(ctx, patches_flat, patch_pos_input);
    ggml_tensor* x = ggml_concat(ctx, cls_with_pos, patches_with_pos, 0);

    // Final encoder norm weights (shared across all intermediate captures)
    ggml_tensor* fnorm_w = clone_weight(ctx, ml, "pretrained.norm.weight");
    ggml_tensor* fnorm_b = clone_weight(ctx, ml, "pretrained.norm.bias");

    int n_inter = (int)inter_layers.size();
    ggml_tensor* lastIntermediate = nullptr;

    // Run ALL transformer blocks, capturing intermediates as we go
    for (int i = 0; i < (int)cfg.depth; ++i) {
        x = vit_block(ctx, x, ml, i, cfg);

        // Check if this block index is in our intermediate layers list
        for (int ci = 0; ci < n_inter; ++ci) {
            if (i == inter_layers[ci]) {
                // Apply final encoder norm and capture
                ggml_tensor* normed = layer_norm(ctx, x, fnorm_w, fnorm_b);
                if (ci < n_inter - 1) {
                    // Capture all but the last intermediate
                    capture_graph_output(normed, captureDsts[ci]);
                } else {
                    // Last intermediate is the main output
                    lastIntermediate = normed;
                }
                break;
            }
        }
    }

    return lastIntermediate;
}

std::vector<float> predict(const ModelLoader& ml, const PreprocessedImage& input, int n_threads) {
    const DAConfig& cfg = ml.config();

    int p_h = input.h / (int)cfg.patch_size;
    int p_w = input.w / (int)cfg.patch_size;
    int num_patches = p_h * p_w;
    int embed = (int)cfg.embed_dim;

    // Convert CHW -> WHC for ggml conv2d
    g_perf_start = std::chrono::steady_clock::now();
    std::vector<float> ggml_input((size_t)input.w * input.h * 3);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < input.h; ++y) {
            for (int x = 0; x < input.w; ++x) {
                ggml_input[x + y * input.w + c * input.w * input.h] =
                    input.data[c * input.h * input.w + y * input.w + x];
            }
        }
    }

    // Pre-compute positional embeddings on host
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
        // Bicubic interpolate pos_embed
        std::vector<float> grid_in((size_t)sqrt_N * sqrt_N);
        std::vector<float> grid_out((size_t)p_h * p_w);
        for (int d = 0; d < embed; ++d) {
            for (int y = 0; y < sqrt_N; ++y) {
                for (int x = 0; x < sqrt_N; ++x) {
                    grid_in[y * sqrt_N + x] = pos_ptr[(y * sqrt_N + x + 1) * embed + d];
                }
            }
            // Bicubic
            float x_r = (float)sqrt_N / (float)p_w;
            float y_r = (float)sqrt_N / (float)p_h;
            for (int y = 0; y < p_h; ++y) {
                for (int x = 0; x < p_w; ++x) {
                    float sx = (x + 0.5f) * x_r - 0.5f;
                    float sy = (y + 0.5f) * y_r - 0.5f;
                    int x0 = (int)std::floor(sx);
                    int y0 = (int)std::floor(sy);
                    float dx = sx - x0;
                    float dy = sy - y0;
                    float val = 0.0f;
                    for (int j = -1; j <= 2; ++j) {
                        for (int i = -1; i <= 2; ++i) {
                            int nx = std::max(0, std::min(x0 + i, sqrt_N - 1));
                            int ny = std::max(0, std::min(y0 + j, sqrt_N - 1));
                            val += grid_in[ny * sqrt_N + nx] * cubic_kernel(dx - i) * cubic_kernel(dy - j);
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

    perf_log("cpu_prep (CHW+pos_embed)");

    // === CAPTURE INTERMEDIATE FEATURES (single-pass) ===
    // Run ViT encoder once, capturing all intermediate block outputs simultaneously
    const std::vector<int32_t>& inter_layers = cfg.intermediate_layers;
    int n_inter = (int)inter_layers.size();

    std::vector<std::vector<float>> inter_features(n_inter);
    std::vector<std::vector<float>*> captureDsts(n_inter);
    for (int i = 0; i < n_inter; ++i) captureDsts[i] = &inter_features[i];

    bool vitOk = da::run_graph(0, n_threads, [&](ggml_context* ctx) -> ggml_tensor* {
        return build_vit_all_blocks(ctx, ml, cfg,
            num_patches, embed, ggml_input, cls_pos_data, patch_pos_data,
            input.w, input.h, inter_layers, captureDsts);
    }, inter_features[n_inter - 1]);

    if (!vitOk) {
        DA_LOG("predict: ViT encoder (single-pass) failed");
        return {};
    }
    DA_LOG("predict: ViT single-pass captured %d intermediates", n_inter);

    perf_log("vit_single_pass");

    // Extract patch tokens (skip cls at row 0) from all intermediates
    for (int i = 0; i < n_inter; ++i) {
        std::vector<float> patchTokens((size_t)num_patches * embed);
        const auto& vitOut = inter_features[i];
        for (int p = 0; p < num_patches; ++p) {
            for (int d = 0; d < embed; ++d) {
                patchTokens[p + d * num_patches] = vitOut[(p + 1) + d * (num_patches + 1)];
            }
        }
        inter_features[i] = std::move(patchTokens);
    }

    perf_log("patch_token_extraction");

    // === DPT HEAD + REFINET + OUTPUT (single merged GPU graph) ===
    // All 4 layer features stay on GPU — no intermediate CPU round-trips.
    int head_feat = (int)cfg.head_features;
    int head_feat_half = head_feat / 2;

    // Resize layer output sizes
    std::vector<int> resized_h(n_inter), resized_w(n_inter);
    resized_h[0] = p_h * 4; resized_w[0] = p_w * 4;
    resized_h[1] = p_h * 2; resized_w[1] = p_w * 2;
    resized_h[2] = p_h;     resized_w[2] = p_w;
    resized_h[3] = (p_h - 1) / 2 + 1;
    resized_w[3] = (p_w - 1) / 2 + 1;

    uint32_t bilinear_corners = GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS;

    std::vector<float> depth_full;
    bool ok = da::run_graph(0, n_threads, [&](ggml_context* ctx) -> ggml_tensor* {
        // === DPT HEAD: project + resize + layer_rn for all 4 features ===
        ggml_tensor* layer_rn[4];
        for (int i = 0; i < n_inter; ++i) {
            ggml_tensor* feat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t)num_patches, embed);
            add_graph_input(feat, inter_features[i].data(), inter_features[i].size() * sizeof(float));
            feat = ggml_reshape_4d(ctx, feat, (int64_t)p_w, (int64_t)p_h, embed, 1);

            // Project: conv2d 1x1
            std::string proj_w = "depth_head.projects." + std::to_string(i) + ".weight";
            std::string proj_b = "depth_head.projects." + std::to_string(i) + ".bias";
            feat = conv2d(ctx, feat, ml, proj_w.c_str(), proj_b.c_str(), 1, 1, 1, 1, 0, 0);

            // Resize
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

            // layer_rn
            std::string lrn_w = "depth_head.scratch.layer" + std::to_string(i+1) + "_rn.weight";
            feat = conv2d(ctx, feat, ml, lrn_w.c_str(), nullptr, 3, 3, 1, 1, 1, 1);
            layer_rn[i] = feat;
        }

        // === REFINET BLOCKS ===
        // refinenet4: resConfUnit2 -> interp -> out_conv
        ggml_tensor* x = layer_rn[3];
        x = residual_conv_unit(ctx, x, ml,
            "depth_head.scratch.refinenet4.resConfUnit2.conv1.weight",
            "depth_head.scratch.refinenet4.resConfUnit2.conv1.bias",
            "depth_head.scratch.refinenet4.resConfUnit2.conv2.weight",
            "depth_head.scratch.refinenet4.resConfUnit2.conv2.bias");
        x = ggml_interpolate(ctx, x, resized_w[2], resized_h[2], head_feat, 1, bilinear_corners);
        x = conv2d(ctx, x, ml,
            "depth_head.scratch.refinenet4.out_conv.weight",
            "depth_head.scratch.refinenet4.out_conv.bias",
            1, 1, 1, 1, 0, 0);

        // refinenet3, refinenet2, refinenet1
        for (int rn = 3; rn >= 1; --rn) {
            int skip_idx = rn - 1;
            ggml_tensor* skip = layer_rn[skip_idx];

            std::string prefix = "depth_head.scratch.refinenet" + std::to_string(rn) + ".";
            ggml_tensor* res = residual_conv_unit(ctx, skip, ml,
                (prefix + "resConfUnit1.conv1.weight").c_str(),
                (prefix + "resConfUnit1.conv1.bias").c_str(),
                (prefix + "resConfUnit1.conv2.weight").c_str(),
                (prefix + "resConfUnit1.conv2.bias").c_str());
            x = ggml_add(ctx, x, res);
            x = residual_conv_unit(ctx, x, ml,
                (prefix + "resConfUnit2.conv1.weight").c_str(),
                (prefix + "resConfUnit2.conv1.bias").c_str(),
                (prefix + "resConfUnit2.conv2.weight").c_str(),
                (prefix + "resConfUnit2.conv2.bias").c_str());

            int out_h = (rn > 1) ? resized_h[rn - 2] : resized_h[0] * 2;
            int out_w = (rn > 1) ? resized_w[rn - 2] : resized_w[0] * 2;
            x = ggml_interpolate(ctx, x, out_w, out_h, head_feat, 1, bilinear_corners);
            x = conv2d(ctx, x, ml,
                (prefix + "out_conv.weight").c_str(),
                (prefix + "out_conv.bias").c_str(),
                1, 1, 1, 1, 0, 0);
        }

        // === OUTPUT CONVS + FINAL RESIZE ===
        x = conv2d(ctx, x, ml,
            "depth_head.scratch.output_conv1.weight",
            "depth_head.scratch.output_conv1.bias",
            3, 3, 1, 1, 1, 1);

        int pre_up_h = resized_h[0] * 2;
        int pre_up_w = resized_w[0] * 2;
        x = ggml_interpolate(ctx, x, pre_up_w * 2, pre_up_h * 2, head_feat_half, 1, bilinear_corners);

        x = conv2d(ctx, x, ml,
            "depth_head.scratch.output_conv2.0.weight",
            "depth_head.scratch.output_conv2.0.bias",
            3, 3, 1, 1, 1, 1);
        x = ggml_relu(ctx, x);
        x = conv2d(ctx, x, ml,
            "depth_head.scratch.output_conv2.2.weight",
            "depth_head.scratch.output_conv2.2.bias",
            1, 1, 1, 1, 0, 0);
        x = ggml_relu(ctx, x);

        x = ggml_interpolate(ctx, x, input.orig_w, input.orig_h, 1, 1, bilinear_corners);
        return ggml_reshape_1d(ctx, x, (int64_t)input.orig_w * input.orig_h);
    }, depth_full);

    if (!ok) { DA_LOG("predict: dpt+refinet+output GPU failed"); return {}; }

    perf_log("dpt+refinet+output (GPU)");
    return depth_full;
}

std::vector<float> postprocess_depth(const std::vector<float>& depth,
                                       int32_t depth_h, int32_t depth_w,
                                       int32_t target_h, int32_t target_w) {
    if (depth_h == target_h && depth_w == target_w) return depth;
    std::vector<float> result((size_t)target_w * target_h);
    // Python uses align_corners=True for final depth upscale (dpt.py:192)
    bilinear_upscale_host_corners(depth.data(), result.data(),
        depth_w, depth_h, 1, 1, target_w, target_h);
    return result;
}

std::vector<uint16_t> depth_to_uint16(const std::vector<float>& depth) {
    float d_min = depth[0], d_max = depth[0];
    for (float v : depth) {
        d_min = std::min(d_min, v);
        d_max = std::max(d_max, v);
    }
    float range = d_max - d_min;
    if (range < 1e-6f) range = 1.0f;

    std::vector<uint16_t> result(depth.size());
    for (size_t i = 0; i < depth.size(); ++i) {
        float t = (depth[i] - d_min) / range;
        result[i] = (uint16_t)(t * 65535.0f);
    }
    return result;
}

} // namespace da
