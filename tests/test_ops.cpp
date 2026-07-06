// Copyright 2026 depth-anything-v2.cpp contributors
// Licensed under the Apache License, Version 2.0

#include "model.hpp"
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "vision_ops.hpp"
#include "ggml.h"
#include "common.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <functional>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("\n=== Test: %s ===\n", name);
#define PASS() tests_passed++; printf("  PASS\n");
#define FAIL(msg) do { tests_failed++; printf("  FAIL: %s\n", msg); } while(0)
#define CHECK(cond, msg) do { if (!(cond)) FAIL(msg); } while(0)

// Print tensor shape for debugging
static void print_shape(const char* name, ggml_tensor* t) {
    printf("  %s: [%lld", name, (long long)t->ne[0]);
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] > 1) printf(", %lld", (long long)t->ne[i]);
    }
    printf("]\n");
}

// Test 1: Linear layer
// PyTorch: y = x @ W^T + b, W is [out, in]
// For vitb block 0: qkv.weight is [2304, 768]
static void test_linear(const da::ModelLoader& ml) {
    TEST("linear");

    const int in_dim = 768;
    const int out_dim = 3 * 768; // QKV
    const int seq_len = 10; // small test

    // Create test input: [seq_len, in_dim]
    std::vector<float> x_data((size_t)seq_len * in_dim);
    for (int i = 0; i < seq_len * in_dim; ++i) {
        x_data[i] = (float)(i % 7); // small repeatable values
    }
    std::vector<float> output;

    bool ok = da::run_graph(0, 4, [&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* x = da::graph_input_tensor(ctx, GGML_TYPE_F32, 2,
            (int64_t[]){seq_len, in_dim}, x_data.data(), x_data.size() * sizeof(float));

        ggml_tensor* W = da::clone_weight(ctx, ml, "pretrained.blocks.0.attn.qkv.weight");
        printf("  x:    ne=[%lld, %lld]\n", (long long)x->ne[0], (long long)x->ne[1]);
        printf("  W:    ne=[%lld, %lld]\n", (long long)W->ne[0], (long long)W->ne[1]);
        printf("  need: W->ne[0](%lld) == x->ne[1](%lld)\n", (long long)W->ne[0], (long long)x->ne[1]);

        ggml_tensor* y = da::linear(ctx, x, ml,
            "pretrained.blocks.0.attn.qkv.weight",
            "pretrained.blocks.0.attn.qkv.bias");

        print_shape("output", y);
        CHECK(y->ne[0] == seq_len, "wrong seq dim");
        CHECK(y->ne[1] == out_dim, "wrong out dim");
        return y;
    }, output);

    CHECK(ok, "graph failed");
    CHECK(output.size() == (size_t)seq_len * out_dim, "wrong output size");

    // Verify first few values are reasonable (not NaN/Inf)
    for (int i = 0; i < 5; ++i) {
        CHECK(std::isfinite(output[i]), "NaN/Inf in output");
    }
    printf("  first 3 values: %.4f %.4f %.4f\n", output[0], output[1], output[2]);
    PASS();
}

// Test 2: LayerNorm
// LN normalizes over the feature dimension
static void test_layer_norm(const da::ModelLoader& ml) {
    TEST("layer_norm");

    const int seq_len = 10;
    const int embed = 768;

    std::vector<float> x_data((size_t)seq_len * embed);
    for (int i = 0; i < seq_len * embed; ++i) {
        x_data[i] = (float)(i % 11) - 5.0f; // centered values
    }
    std::vector<float> output;

    bool ok = da::run_graph(0, 4, [&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* x = da::graph_input_tensor(ctx, GGML_TYPE_F32, 2,
            (int64_t[]){seq_len, embed}, x_data.data(), x_data.size() * sizeof(float));

        ggml_tensor* w = da::clone_weight(ctx, ml, "pretrained.blocks.0.norm1.weight");
        ggml_tensor* b = da::clone_weight(ctx, ml, "pretrained.blocks.0.norm1.bias");

        ggml_tensor* y = da::layer_norm(ctx, x, w, b);
        print_shape("output", y);
        CHECK(y->ne[0] == seq_len, "wrong seq dim");
        CHECK(y->ne[1] == embed, "wrong embed dim");
        return y;
    }, output);

    CHECK(ok, "graph failed");
    CHECK(output.size() == (size_t)seq_len * embed, "wrong output size");

    // LN output should have unit variance per position (approximately)
    // Check first position
    float mean = 0, var = 0;
    for (int i = 0; i < embed; ++i) {
        mean += output[i];
    }
    mean /= embed;
    for (int i = 0; i < embed; ++i) {
        var += (output[i] - mean) * (output[i] - mean);
    }
    var /= embed;
    printf("  first pos: mean=%.6f var=%.6f (after LN+affine)\n", mean, var);
    PASS();
}

// Test 3: Self-attention (full)
static void test_self_attention(const da::ModelLoader& ml) {
    TEST("self_attention");

    const int seq_len = 16;
    const int embed = 768;
    const uint32_t num_heads = 12;

    std::vector<float> x_data((size_t)seq_len * embed);
    for (int i = 0; i < seq_len * embed; ++i) {
        x_data[i] = (float)(i % 13) / 10.0f;
    }
    std::vector<float> output;

    bool ok = da::run_graph(0, 4, [&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* x = da::graph_input_tensor(ctx, GGML_TYPE_F32, 2,
            (int64_t[]){seq_len, embed}, x_data.data(), x_data.size() * sizeof(float));

        // First apply layer norm (as done in the actual block)
        ggml_tensor* w = da::clone_weight(ctx, ml, "pretrained.blocks.0.norm1.weight");
        ggml_tensor* b = da::clone_weight(ctx, ml, "pretrained.blocks.0.norm1.bias");
        ggml_tensor* x_norm = da::layer_norm(ctx, x, w, b);

        ggml_tensor* y = da::self_attention(ctx, x_norm, ml,
            "pretrained.blocks.0.attn.qkv.weight", "pretrained.blocks.0.attn.qkv.bias",
            "pretrained.blocks.0.attn.proj.weight", "pretrained.blocks.0.attn.proj.bias",
            num_heads);

        print_shape("output", y);
        CHECK(y->ne[0] == seq_len, "wrong seq dim");
        CHECK(y->ne[1] == embed, "wrong embed dim");
        return y;
    }, output);

    CHECK(ok, "graph failed");
    CHECK(output.size() == (size_t)seq_len * embed, "wrong output size");

    for (int i = 0; i < 5; ++i) {
        CHECK(std::isfinite(output[i]), "NaN/Inf in output");
    }
    printf("  first 3 values: %.4f %.4f %.4f\n", output[0], output[1], output[2]);
    PASS();
}

// Test 4: MLP
static void test_mlp(const da::ModelLoader& ml) {
    TEST("mlp");

    const int seq_len = 16;
    const int embed = 768;

    std::vector<float> x_data((size_t)seq_len * embed);
    for (int i = 0; i < seq_len * embed; ++i) {
        x_data[i] = (float)(i % 17) / 10.0f;
    }
    std::vector<float> output;

    bool ok = da::run_graph(0, 4, [&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* x = da::graph_input_tensor(ctx, GGML_TYPE_F32, 2,
            (int64_t[]){seq_len, embed}, x_data.data(), x_data.size() * sizeof(float));

        ggml_tensor* w = da::clone_weight(ctx, ml, "pretrained.blocks.0.norm2.weight");
        ggml_tensor* b = da::clone_weight(ctx, ml, "pretrained.blocks.0.norm2.bias");
        ggml_tensor* x_norm = da::layer_norm(ctx, x, w, b);

        ggml_tensor* y = da::mlp(ctx, x_norm, ml,
            "pretrained.blocks.0.mlp.fc1.weight", "pretrained.blocks.0.mlp.fc1.bias",
            "pretrained.blocks.0.mlp.fc2.weight", "pretrained.blocks.0.mlp.fc2.bias",
            "mlp");

        print_shape("output", y);
        CHECK(y->ne[0] == seq_len, "wrong seq dim");
        CHECK(y->ne[1] == embed, "wrong embed dim");
        return y;
    }, output);

    CHECK(ok, "graph failed");
    CHECK(output.size() == (size_t)seq_len * embed, "wrong output size");
    for (int i = 0; i < 5; ++i) {
        CHECK(std::isfinite(output[i]), "NaN/Inf in output");
    }
    PASS();
}

// Test 5: Full transformer block
static void test_vit_block(const da::ModelLoader& ml) {
    TEST("vit_block");

    const int seq_len = 16;
    const int embed = 768;

    std::vector<float> x_data((size_t)seq_len * embed);
    for (int i = 0; i < seq_len * embed; ++i) {
        x_data[i] = (float)(i % 19) / 10.0f;
    }
    std::vector<float> output;

    // Build a full block manually (same as vit_block in predict.cpp)
    bool ok = da::run_graph(0, 4, [&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* x = da::graph_input_tensor(ctx, GGML_TYPE_F32, 2,
            (int64_t[]){seq_len, embed}, x_data.data(), x_data.size() * sizeof(float));

        const char* p = "pretrained.blocks.0.";

        // Norm1 + Attention + LayerScale + Residual
        ggml_tensor* norm1 = da::layer_norm(ctx, x,
            da::clone_weight(ctx, ml, "pretrained.blocks.0.norm1.weight"),
            da::clone_weight(ctx, ml, "pretrained.blocks.0.norm1.bias"));

        ggml_tensor* attn = da::self_attention(ctx, norm1, ml,
            "pretrained.blocks.0.attn.qkv.weight", "pretrained.blocks.0.attn.qkv.bias",
            "pretrained.blocks.0.attn.proj.weight", "pretrained.blocks.0.attn.proj.bias",
            12);

        ggml_tensor* ls1 = da::layer_scale(ctx, attn,
            da::clone_weight(ctx, ml, "pretrained.blocks.0.ls1.gamma"));
        x = ggml_add(ctx, x, ls1);

        // Norm2 + MLP + LayerScale + Residual
        ggml_tensor* norm2 = da::layer_norm(ctx, x,
            da::clone_weight(ctx, ml, "pretrained.blocks.0.norm2.weight"),
            da::clone_weight(ctx, ml, "pretrained.blocks.0.norm2.bias"));

        ggml_tensor* mlp_out = da::mlp(ctx, norm2, ml,
            "pretrained.blocks.0.mlp.fc1.weight", "pretrained.blocks.0.mlp.fc1.bias",
            "pretrained.blocks.0.mlp.fc2.weight", "pretrained.blocks.0.mlp.fc2.bias",
            "mlp");

        ggml_tensor* ls2 = da::layer_scale(ctx, mlp_out,
            da::clone_weight(ctx, ml, "pretrained.blocks.0.ls2.gamma"));
        x = ggml_add(ctx, x, ls2);

        print_shape("output", x);
        CHECK(x->ne[0] == seq_len, "wrong seq dim");
        CHECK(x->ne[1] == embed, "wrong embed dim");
        return x;
    }, output);

    CHECK(ok, "graph failed");
    CHECK(output.size() == (size_t)seq_len * embed, "wrong output size");
    for (int i = 0; i < 5; ++i) {
        CHECK(std::isfinite(output[i]), "NaN/Inf in output");
    }
    printf("  first 3 values: %.4f %.4f %.4f\n", output[0], output[1], output[2]);
    PASS();
}

// Test 6: Conv2d (patch embedding)
static void test_conv2d_patch_embed(const da::ModelLoader& ml) {
    TEST("conv2d_patch_embed");

    const int h = 518;
    const int w = 784;
    const int channels = 3;
    const int patch_size = 14;
    const int embed = 768;

    // Create small test image
    std::vector<float> img_data((size_t)w * h * channels);
    for (int i = 0; i < w * h * channels; ++i) {
        img_data[i] = (float)(i % 5) / 10.0f;
    }
    std::vector<float> output;

    bool ok = da::run_graph(0, 4, [&](ggml_context* ctx) -> ggml_tensor* {
        // ggml conv2d layout: [W, H, C, N]
        ggml_tensor* x = da::graph_input_tensor(ctx, GGML_TYPE_F32, 4,
            (int64_t[]){w, h, channels, 1}, img_data.data(), img_data.size() * sizeof(float));

        ggml_tensor* y = da::conv2d(ctx, x, ml,
            "pretrained.patch_embed.proj.weight", "pretrained.patch_embed.proj.bias",
            patch_size, patch_size, patch_size, patch_size, 0, 0);

        print_shape("output", y);
        int expected_hw = h / patch_size;
        int expected_ww = w / patch_size;
        CHECK(y->ne[0] == expected_ww, "wrong width");
        CHECK(y->ne[1] == expected_hw, "wrong height");
        CHECK(y->ne[2] == embed, "wrong channels");
        return y;
    }, output);

    CHECK(ok, "graph failed");
    int expected_patches = (h / patch_size) * (w / patch_size);
    CHECK(output.size() == (size_t)expected_patches * embed, "wrong output size");
    for (int i = 0; i < 5; ++i) {
        CHECK(std::isfinite(output[i]), "NaN/Inf in output");
    }
    PASS();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        DA_LOG("Usage: %s <model.gguf>", argv[0]);
        return 1;
    }

    da::ModelLoader ml;
    if (!ml.load(argv[1])) {
        DA_LOG("Failed to load model");
        return 1;
    }
    DA_LOG("Loaded model: %s arch=%s embed=%d depth=%d",
           argv[1], ml.config().arch.c_str(),
           ml.config().embed_dim, ml.config().depth);

    // Run tests
    test_linear(ml);
    test_layer_norm(ml);
    test_self_attention(ml);
    test_mlp(ml);
    test_vit_block(ml);
    test_conv2d_patch_embed(ml);

    da::shutdown_backend();

    printf("\n==============================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
