// Copyright 2026 depth-anything-v2.cpp contributors
// Licensed under the Apache License, Version 2.0

#pragma once
#include <functional>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace da {

// One-shot graph runner via the process-global persistent Backend.
bool run_graph(size_t mem_bytes, int n_threads,
               const std::function<ggml_tensor*(ggml_context*)>& build,
               std::vector<float>& out);

// Variant: post-extraction callback receives output data
bool run_graph(size_t mem_bytes, int n_threads,
               const std::function<ggml_tensor*(ggml_context*)>& build,
               const std::function<void(const std::vector<float>&)>& extract);

// Non-locked variant for nested calls
bool run_graph_sync(size_t mem_bytes, int n_threads,
                     const std::function<ggml_tensor*(ggml_context*)>& build,
                     std::vector<float>& out);

void set_num_threads(int n);
int  num_threads();

// Register a tensor for capture (its data will be downloaded after graph execution)
void capture_graph_output(ggml_tensor* t, std::vector<float>* dst);

class Backend;
Backend& global_backend();
void shutdown_backend();

} // namespace da
