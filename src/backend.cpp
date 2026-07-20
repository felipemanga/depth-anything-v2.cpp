// Copyright 2026 depth-anything-v2.cpp contributors
// Licensed under the Apache License, Version 2.0

#include "backend.hpp"
#include "common.hpp"
#include "ggml_graph.hpp"
#include "model.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace da {

namespace {
constexpr size_t kGraphSize = 32768;

struct PendingInput {
    ggml_tensor* tensor;
    const void*  host;
    size_t       nbytes;
};

struct PendingCapture {
    ggml_tensor*        tensor;
    std::vector<float>* dst;
};
} // namespace

struct Backend::Impl {
    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       cpu_backend = nullptr;
    ggml_gallocr_t       galloc      = nullptr;
    ggml_backend_sched_t sched       = nullptr;
    bool                 use_sched   = false;
    std::vector<PendingInput> pending;
    std::vector<PendingCapture> captures;
};

static thread_local Backend* t_active = nullptr;

Backend::Backend(int n_threads) : impl_(new Impl()) {
    const char* force = std::getenv("DA_DEVICE");
    const bool force_cpu = force && std::string(force) == "cpu";

    if (!force_cpu) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                impl_->backend = ggml_backend_dev_init(dev, nullptr);
                if (impl_->backend) {
                    device_name_ = ggml_backend_dev_name(dev);
                    impl_->use_sched = true; // Always use scheduler for GPU
                    DA_LOG("da::Backend using GPU device: %s", device_name_.c_str());
                    break;
                }
            }
        }
    }
    if (!impl_->backend) {
        impl_->backend = ggml_backend_cpu_init();
        device_name_ = "cpu";
    }
    if (!impl_->backend) {
        DA_LOG("backend init returned null");
        return;
    }
    if (impl_->use_sched) {
        impl_->cpu_backend = ggml_backend_cpu_init();
        if (!impl_->cpu_backend) {
            DA_LOG("da::Backend: CPU fallback init failed; disabling sched");
            impl_->use_sched = false;
        }
    }
    set_n_threads(n_threads);
}

Backend::~Backend() {
    if (impl_) {
        if (impl_->sched)       ggml_backend_sched_free(impl_->sched);
        if (impl_->galloc)      ggml_gallocr_free(impl_->galloc);
        if (impl_->cpu_backend) ggml_backend_free(impl_->cpu_backend);
        if (impl_->backend)     ggml_backend_free(impl_->backend);
        delete impl_;
        impl_ = nullptr;
    }
}

void Backend::set_n_threads(int n_threads) {
    n_threads_ = n_threads > 0 ? n_threads : 1;
    if (impl_ && impl_->backend && ggml_backend_is_cpu(impl_->backend)) {
        ggml_backend_cpu_set_n_threads(impl_->backend, n_threads_);
    }
    if (impl_ && impl_->cpu_backend) {
        ggml_backend_cpu_set_n_threads(impl_->cpu_backend, n_threads_);
    }
}

ggml_backend_t Backend::handle() const {
    return impl_ ? impl_->backend : nullptr;
}

void Backend::register_input(ggml_tensor* t, const void* host, size_t nbytes) {
    impl_->pending.push_back({t, host, nbytes});
}

void Backend::register_capture(ggml_tensor* t, std::vector<float>* dst) {
    impl_->captures.push_back({t, dst});
}

bool Backend::compute(const std::function<ggml_tensor*(ggml_context*)>& build,
                      std::vector<float>& out) {
    if (!impl_ || !impl_->backend) {
        DA_LOG("Backend::compute called on an uninitialised backend");
        return false;
    }

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * kGraphSize +
                            ggml_graph_overhead_custom(kGraphSize, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        DA_LOG("Backend::compute: ggml_init failed");
        return false;
    }

    impl_->pending.clear();
    impl_->captures.clear();
    Backend* prev_active = t_active;
    t_active = this;
    struct ggml_tensor* output = build(ctx);
    t_active = prev_active;

    if (!output) {
        DA_LOG("Backend::compute: build() returned null output tensor");
        impl_->pending.clear();
        impl_->captures.clear();
        ggml_free(ctx);
        return false;
    }
    ggml_set_output(output);
    for (const PendingCapture& pc : impl_->captures) ggml_set_output(pc.tensor);

    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGraphSize, false);
    for (const PendingCapture& pc : impl_->captures)
        ggml_build_forward_expand(gf, pc.tensor);
    ggml_build_forward_expand(gf, output);

    // Always use scheduler to avoid gallocr reallocation issues with GPU
    bool need_sched = impl_->use_sched; // true for GPU, forces sched path

    bool alloc_ok = false;
    if (need_sched) {
        if (!impl_->sched) {
            ggml_backend_t backs[2] = { impl_->backend, impl_->cpu_backend };
            impl_->sched = ggml_backend_sched_new(
                backs, /*bufts=*/nullptr, /*n_backends=*/2,
                /*graph_size=*/kGraphSize, /*parallel=*/false, /*op_offload=*/true);
            if (!impl_->sched) {
                DA_LOG("Backend::compute: ggml_backend_sched_new failed");
                impl_->pending.clear();
                impl_->captures.clear();
                ggml_free(ctx);
                return false;
            }
        }
        ggml_backend_sched_reset(impl_->sched);
        alloc_ok = ggml_backend_sched_alloc_graph(impl_->sched, gf);
        if (!alloc_ok) DA_LOG("Backend::compute: ggml_backend_sched_alloc_graph failed");
    } else {
        if (!impl_->galloc) {
            impl_->galloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(impl_->backend));
            if (!impl_->galloc) {
                DA_LOG("Backend::compute: ggml_gallocr_new failed");
                impl_->pending.clear();
                impl_->captures.clear();
                ggml_free(ctx);
                return false;
            }
            // Pre-reserve to avoid reallocation bug
            ggml_gallocr_reserve(impl_->galloc, gf);
        }
        alloc_ok = ggml_gallocr_alloc_graph(impl_->galloc, gf);
        if (!alloc_ok) {
            // If alloc fails, try reserve again (graph may have grown)
            if (ggml_gallocr_reserve(impl_->galloc, gf)) {
                alloc_ok = ggml_gallocr_alloc_graph(impl_->galloc, gf);
            }
            if (!alloc_ok) DA_LOG("Backend::compute: ggml_gallocr_alloc_graph failed");
        }
    }
    if (!alloc_ok) {
        impl_->pending.clear();
        impl_->captures.clear();
        ggml_free(ctx);
        return false;
    }

    for (const PendingInput& pi : impl_->pending) {
        ggml_backend_tensor_set(pi.tensor, pi.host, 0, pi.nbytes);
    }
    impl_->pending.clear();

    enum ggml_status status = need_sched
        ? ggml_backend_sched_graph_compute(impl_->sched, gf)
        : ggml_backend_graph_compute(impl_->backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        DA_LOG("Backend::compute: ggml_backend_graph_compute failed (status=%d)",
               (int)status);
        impl_->captures.clear();
        ggml_free(ctx);
        return false;
    }

    for (const PendingCapture& pc : impl_->captures) {
        size_t cn = (size_t)ggml_nelements(pc.tensor);
        pc.dst->resize(cn);
        ggml_backend_tensor_get(pc.tensor, pc.dst->data(), 0, cn * sizeof(float));
    }
    impl_->captures.clear();

    size_t n = (size_t)ggml_nelements(output);
    out.resize(n);
    ggml_backend_tensor_get(output, out.data(), 0, n * sizeof(float));

    ggml_free(ctx);
    return true;
}

void add_graph_input(ggml_tensor* t, const void* host, size_t nbytes) {
    GGML_ASSERT(t_active != nullptr &&
                "add_graph_input called outside a Backend::compute build lambda");
    ggml_set_input(t);
    t_active->register_input(t, host, nbytes);
}

ggml_tensor* graph_input_tensor(ggml_context* ctx, int type, int n_dims,
                                const int64_t* ne, const void* host,
                                size_t nbytes) {
    ggml_tensor* t = ggml_new_tensor(ctx, (ggml_type)type, n_dims, ne);
    add_graph_input(t, host, nbytes);
    return t;
}

void ensure_weights_realized(const ModelLoader& ml) {
    if (ml.weights_realized()) return;
    static std::mutex init_mutex;
    std::lock_guard<std::mutex> lock(init_mutex);
    if (ml.weights_realized()) return;
    ModelLoader& mut = const_cast<ModelLoader&>(ml);
    mut.realize_weights(global_backend().handle());
}

ggml_tensor* clone_weight(ggml_context* /*ctx*/, const ModelLoader& ml,
                          const char* name) {
    ensure_weights_realized(ml);
    ggml_tensor* src = ml.tensor(name);
    assert(src && "missing tensor");
    return src;
}

ggml_tensor* clone_weight_opt(ggml_context* ctx, const ModelLoader& ml,
                              const char* name) {
    if (!name || !ml.tensor(name)) return nullptr;
    return clone_weight(ctx, ml, name);
}

void weight_to_host_f32(const ModelLoader& ml, const char* name, std::vector<float>& out) {
    ensure_weights_realized(ml);
    ggml_tensor* t = ml.tensor(name);
    GGML_ASSERT(t && "weight_to_host_f32: missing tensor");
    GGML_ASSERT(t->type == GGML_TYPE_F32 && "weight_to_host_f32: tensor not f32");
    out.resize((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
}

} // namespace da
