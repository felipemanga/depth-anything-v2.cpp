// Copyright 2026 depth-anything-v2.cpp contributors
// Licensed under the Apache License, Version 2.0

#include "model.hpp"
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "image.hpp"
#include "predict.hpp"
#include "common.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include <algorithm>
#include <sys/stat.h>

static void print_usage(const char* prog) {
    std::fprintf(stderr, "Usage: %s <command> [options]\n", prog);
    std::fprintf(stderr, "\nCommands:\n");
    std::fprintf(stderr, "  predict  <model.gguf> <image.png> [-o output.png] [-d depth.png] [-t threads]\n");
    std::fprintf(stderr, "  info     <model.gguf>\n");
    std::fprintf(stderr, "  bench    <model.gguf> <image.png> [-n iterations] [-t threads]\n");
    std::fprintf(stderr, "\nOptions:\n");
    std::fprintf(stderr, "  -o <path>   Output colored depth image (default: depth_colored.png)\n");
    std::fprintf(stderr, "  -d <path>   Output raw 16-bit depth map (default: depth_raw.png)\n");
    std::fprintf(stderr, "  -t <n>      Number of threads (default: auto)\n");
    std::fprintf(stderr, "  -n <n>      Number of benchmark iterations (default: 5)\n");
}

static bool file_exists(const std::string& path) {
    struct stat buf;
    return stat(path.c_str(), &buf) == 0;
}

static int cmd_info(const std::string& model_path) {
    da::ModelLoader ml;
    if (!ml.load(model_path)) {
        DA_LOG("Failed to load model: %s", model_path.c_str());
        return 1;
    }

    const auto& cfg = ml.config();
    DA_LOG("Model: %s", model_path.c_str());
    DA_LOG("  Arch:            %s", cfg.arch.c_str());
    DA_LOG("  Embed dim:       %u", cfg.embed_dim);
    DA_LOG("  Depth:           %u blocks", cfg.depth);
    DA_LOG("  Num heads:       %u", cfg.num_heads);
    DA_LOG("  MLP ratio:       %u", cfg.mlp_ratio);
    DA_LOG("  Patch size:      %u", cfg.patch_size);
    DA_LOG("  Input size:      %u", cfg.input_size);
    DA_LOG("  Init values:     %.4f", cfg.init_values);
    DA_LOG("  FFN layer:       %s", cfg.ffn_layer.c_str());
    DA_LOG("  Head features:   %u", cfg.head_features);
    DA_LOG("  Out channels:    [%u, %u, %u, %u]",
           cfg.out_channels[0], cfg.out_channels[1],
           cfg.out_channels[2], cfg.out_channels[3]);
    DA_LOG("  Intermediate:    [%d, %d, %d, %d]",
           cfg.intermediate_layers[0], cfg.intermediate_layers[1],
           cfg.intermediate_layers[2], cfg.intermediate_layers[3]);
    return 0;
}

// Simple jet colormap
static void jet_colormap(float t, uint8_t& r, uint8_t& g, uint8_t& b) {
    t = std::max(0.0f, std::min(1.0f, t));
    r = (uint8_t)(255.0f * std::max(0.0f, std::min(1.0f, 1.0f - 4.0f * std::abs(t - 0.75f))));
    g = (uint8_t)(255.0f * std::max(0.0f, std::min(1.0f, 1.0f - 4.0f * std::abs(t - 0.50f))));
    b = (uint8_t)(255.0f * std::max(0.0f, std::min(1.0f, 1.0f - 4.0f * std::abs(t - 0.25f))));
}

static int cmd_predict(const std::string& model_path, const std::string& image_path,
                       const std::string& output_path, const std::string& depth_path,
                       int n_threads) {
    if (n_threads > 0) da::set_num_threads(n_threads);

    da::ModelLoader ml;
    if (!ml.load(model_path)) return 1;

    DA_LOG("Loading image: %s", image_path.c_str());
    da::Image img = da::load_image(image_path);
    if (img.empty()) return 1;
    DA_LOG("  Size: %dx%d", img.w, img.h);

    DA_LOG("Preprocessing...");
    da::PreprocessedImage prep = da::preprocess(img, ml.config().input_size, ml.config().patch_size);
    DA_LOG("  Preprocessed: %dx%d", prep.w, prep.h);

    DA_LOG("Running inference on %s...", da::global_backend().device_name());
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> depth = da::predict(ml, prep, n_threads);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    DA_LOG("  Inference time: %.3f s", elapsed);

    if (depth.empty()) {
        DA_LOG("Inference failed");
        return 1;
    }

    // predict() already returns depth at original image resolution
    const std::vector<float>& depth_full = depth;

    // Save raw depth
    if (!depth_path.empty()) {
        auto depth_u16 = da::depth_to_uint16(depth_full);
        da::save_depth(depth_path, depth_u16, img.w, img.h);
        DA_LOG("Saved raw depth: %s", depth_path.c_str());
    }

    // Save colored depth
    if (!output_path.empty()) {
        // Normalize to [0, 1]
        float min_d = depth_full[0], max_d = depth_full[0];
        for (float v : depth_full) {
            min_d = std::min(min_d, v);
            max_d = std::max(max_d, v);
        }
        float range = max_d - min_d;
        if (range < 1e-6f) range = 1.0f;

        std::vector<uint8_t> colored((size_t)img.w * img.h * 3);
        for (int i = 0; i < img.w * img.h; ++i) {
            float t = (depth_full[i] - min_d) / range;
            uint8_t r, g, b;
            jet_colormap(t, r, g, b);
            colored[i*3+0] = r;
            colored[i*3+1] = g;
            colored[i*3+2] = b;
        }
        da::save_image(output_path, colored, img.w, img.h);
        DA_LOG("Saved colored depth: %s", output_path.c_str());
    }

    return 0;
}

static int cmd_bench(const std::string& model_path, const std::string& image_path,
                     int n_iter, int n_threads) {
    if (n_threads > 0) da::set_num_threads(n_threads);

    da::ModelLoader ml;
    if (!ml.load(model_path)) return 1;

    da::Image img = da::load_image(image_path);
    if (img.empty()) return 1;

    da::PreprocessedImage prep = da::preprocess(img, ml.config().input_size, ml.config().patch_size);

    // Warmup
    DA_LOG("Warmup...");
    da::predict(ml, prep, n_threads);

    std::vector<double> times;
    times.reserve(n_iter);
    for (int i = 0; i < n_iter; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        da::predict(ml, prep, n_threads);
        auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }

    // Sort for median
    std::sort(times.begin(), times.end());
    double median = times[times.size() / 2];
    double mean = 0;
    for (double t : times) mean += t;
    mean /= times.size();

    DA_LOG("Benchmark: %s", da::global_backend().device_name());
    DA_LOG("  Image:     %dx%d -> %dx%d", img.w, img.h, prep.w, prep.h);
    DA_LOG("  Iterations: %d", n_iter);
    DA_LOG("  Median:    %.3f s (%.1f fps)", median, 1.0 / median);
    DA_LOG("  Mean:      %.3f s (%.1f fps)", mean, 1.0 / mean);
    DA_LOG("  Min:       %.3f s", times.front());
    DA_LOG("  Max:       %.3f s", times.back());

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    int ret = 1;
    if (command == "info" && argc >= 3) {
        ret = cmd_info(argv[2]);
    } else if (command == "predict" && argc >= 4) {
        std::string output_path = "depth_colored.png";
        std::string depth_path = "depth_raw.png";
        int n_threads = 0;
        for (int i = 4; i < argc; ++i) {
            if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) output_path = argv[++i];
            else if (std::strcmp(argv[i], "-d") == 0 && i + 1 < argc) depth_path = argv[++i];
            else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = std::atoi(argv[++i]);
        }
        ret = cmd_predict(argv[2], argv[3], output_path, depth_path, n_threads);
    } else if (command == "bench" && argc >= 4) {
        int n_iter = 5, n_threads = 0;
        for (int i = 4; i < argc; ++i) {
            if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_iter = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = std::atoi(argv[++i]);
        }
        ret = cmd_bench(argv[2], argv[3], n_iter, n_threads);
    } else {
        print_usage(argv[0]);
    }

    da::shutdown_backend();
    return ret;
}
