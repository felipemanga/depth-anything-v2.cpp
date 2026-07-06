// Copyright 2026 depth-anything-v2.cpp contributors
// Licensed under the Apache License, Version 2.0

#include "model.hpp"
#include "common.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cstring>
#include <vector>
#include <utility>

namespace da {

namespace {
static uint32_t kv_u32(gguf_context* g, const char* k, uint32_t d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : (uint32_t)gguf_get_val_u32(g,id);
}
static int32_t kv_i32(gguf_context* g, const char* k, int32_t d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_i32(g,id);
}
static float kv_f32(gguf_context* g, const char* k, float d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_f32(g,id);
}
static bool kv_bool(gguf_context* g, const char* k, bool d=false){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_bool(g,id);
}
static std::string kv_str(gguf_context* g, const char* k, const char* d=""){
    int64_t id = gguf_find_key(g,k); return id<0 ? std::string(d) : std::string(gguf_get_val_str(g,id));
}
static std::vector<uint32_t> kv_u32_arr(gguf_context* g, const char* k){
    std::vector<uint32_t> out;
    int64_t id = gguf_find_key(g,k);
    if(id>=0 && gguf_get_arr_type(g,id)==GGUF_TYPE_UINT32){
        size_t n = gguf_get_arr_n(g,id);
        const uint32_t* a = (const uint32_t*)gguf_get_arr_data(g,id);
        out.assign(a, a+n);
    }
    return out;
}
static std::vector<int32_t> kv_i32_arr(gguf_context* g, const char* k){
    std::vector<int32_t> out;
    int64_t id = gguf_find_key(g,k);
    if(id>=0 && gguf_get_arr_type(g,id)==GGUF_TYPE_INT32){
        size_t n = gguf_get_arr_n(g,id);
        const int32_t* a = (const int32_t*)gguf_get_arr_data(g,id);
        out.assign(a, a+n);
    }
    return out;
}
} // namespace

ModelLoader::~ModelLoader(){
    if(weights_buf_) ggml_backend_buffer_free(weights_buf_);
    if(device_ctx_) ggml_free(device_ctx_);
    if(gguf_) gguf_free(gguf_); if(ctx_) ggml_free(ctx_);
}

bool ModelLoader::realize_weights(ggml_backend_t backend){
    if(weights_buf_) return true;
    if(!backend || !ctx_){ DA_LOG("realize_weights: null backend/ctx"); return false; }

    if (ggml_backend_is_cpu(backend)) {
        void*  base = ggml_get_mem_buffer(ctx_);
        size_t size = ggml_get_mem_size(ctx_);
        weights_buf_ = ggml_backend_cpu_buffer_from_ptr(base, size);
        if(!weights_buf_){ DA_LOG("realize_weights: buffer_from_ptr failed"); return false; }
        for(auto& kv : tensors_) kv.second->buffer = weights_buf_;
        return true;
    }

    // Device path
    const size_t n = tensors_.size();
    struct ggml_init_params dp = {
        /*.mem_size  =*/ ggml_tensor_overhead() * (n + 8),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    device_ctx_ = ggml_init(dp);
    if(!device_ctx_){ DA_LOG("realize_weights: device ctx init failed"); return false; }

    std::vector<std::pair<ggml_tensor*, const void*>> ups; ups.reserve(n);
    std::unordered_map<std::string, ggml_tensor*> devmap; devmap.reserve(n);
    for (auto& kv : tensors_) {
        ggml_tensor* s = kv.second;
        ggml_tensor* d = ggml_new_tensor(device_ctx_, s->type, GGML_MAX_DIMS, s->ne);
        ggml_set_name(d, kv.first.c_str());
        devmap.emplace(kv.first, d);
        ups.emplace_back(d, s->data);
    }
    weights_buf_ = ggml_backend_alloc_ctx_tensors(device_ctx_, backend);
    if(!weights_buf_){ DA_LOG("realize_weights: alloc_ctx_tensors failed"); return false; }
    for (auto& pr : ups)
        ggml_backend_tensor_set(pr.first, pr.second, 0, ggml_nbytes(pr.first));
    tensors_.swap(devmap);
    return true;
}

bool ModelLoader::load(const std::string& path){
    struct gguf_init_params p{ /*no_alloc*/false, /*ctx*/&ctx_ };
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if(!gguf_){ DA_LOG("gguf open failed: %s", path.c_str()); return false; }

    cfg_.arch                 = kv_str(gguf_, "depth_anything.arch");
    cfg_.embed_dim            = kv_u32(gguf_, "depth_anything.embed_dim");
    cfg_.depth                = kv_u32(gguf_, "depth_anything.depth");
    cfg_.num_heads            = kv_u32(gguf_, "depth_anything.num_heads");
    cfg_.mlp_ratio            = kv_u32(gguf_, "depth_anything.mlp_ratio", 4);
    cfg_.patch_size           = kv_u32(gguf_, "depth_anything.patch_size", 14);
    cfg_.input_size           = kv_u32(gguf_, "depth_anything.input_size", 518);
    cfg_.init_values          = kv_f32(gguf_, "depth_anything.init_values", 1.0f);
    cfg_.ffn_layer            = kv_str(gguf_, "depth_anything.ffn_layer", "mlp");
    cfg_.interpolate_offset   = kv_f32(gguf_, "depth_anything.interpolate_offset", 0.1f);
    cfg_.interpolate_antialias = kv_bool(gguf_, "depth_anything.interpolate_antialias", false);
    cfg_.head_features        = kv_u32(gguf_, "depth_anything.head_features", 256);
    cfg_.out_channels         = kv_u32_arr(gguf_, "depth_anything.out_channels");
    cfg_.intermediate_layers  = kv_i32_arr(gguf_, "depth_anything.intermediate_layers");

    // tensors
    const int64_t nt = gguf_get_n_tensors(gguf_);
    for(int64_t i=0; i<nt; ++i){
        const char* nm = gguf_get_tensor_name(gguf_, i);
        ggml_tensor* t = ggml_get_tensor(ctx_, nm);
        if(t) tensors_[nm] = t;
    }
    return cfg_.embed_dim > 0 && cfg_.depth > 0;
}

ggml_tensor* ModelLoader::tensor(const std::string& n) const {
    auto it = tensors_.find(n);
    return it == tensors_.end() ? nullptr : it->second;
}

} // namespace da
