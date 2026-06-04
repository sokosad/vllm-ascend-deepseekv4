#pragma once

#include <torch/extension.h>
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <cstdlib>
#include <cstdio>
#include <mutex>

struct PrefetchStreamCtx {
    bool initialized = false;
    aclrtStream stream = nullptr;
    std::mutex mtx;
};

static PrefetchStreamCtx g_pf_ctx;

static void ensure_pf_stream() {
    std::lock_guard<std::mutex> lock(g_pf_ctx.mtx);
    if (!g_pf_ctx.initialized) {
        aclError ret = aclrtCreateStream(&g_pf_ctx.stream);
        if (ret == ACL_ERROR_NONE) {
            g_pf_ctx.initialized = true;
        }
    }
}

namespace vllm_ascend {

inline void npu_prefetch_async(
    const at::Tensor& weight,
    int64_t prefetch_size) {

    if (!weight.defined() || weight.numel() == 0 || prefetch_size <= 0) {
        return;
    }

    if (weight.device().type() != at::kPrivateUse1) {
        return;
    }

    static bool log_enabled = std::getenv("VLLM_PREFETCH_LOG") != nullptr;

    void* data_ptr = weight.data_ptr();
    size_t actual_size = static_cast<size_t>(prefetch_size);
    aclrtStream compute_stream = c10_npu::getCurrentNPUStream();

    bool capturing = false;
    try {
        capturing = c10_npu::getCurrentNPUStream().isCapturing();
    } catch (...) {
        capturing = false;
    }

    if (capturing) {
        aclError ret = aclrtCmoAsync(
            data_ptr, actual_size,
            ACL_RT_CMO_TYPE_PREFETCH,
            compute_stream);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtCmoAsync capture failed, ret=", ret);
        if (log_enabled) {
            std::printf("[prefetch-c++] capture mode same stream size=%zu\n", actual_size);
            std::fflush(stdout);
        }
        return;
    }

    ensure_pf_stream();

    if (g_pf_ctx.initialized) {
        if (log_enabled) {
            std::printf("[prefetch-c++] async stream size=%zu\n", actual_size);
            std::fflush(stdout);
        }
        aclError ret = aclrtCmoAsync(
            data_ptr, actual_size,
            ACL_RT_CMO_TYPE_PREFETCH,
            g_pf_ctx.stream);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtCmoAsync async failed, ret=", ret);
    } else {
        if (log_enabled) {
            std::printf("[prefetch-c++] same stream size=%zu\n", actual_size);
            std::fflush(stdout);
        }
        aclError ret = aclrtCmoAsync(
            data_ptr, actual_size,
            ACL_RT_CMO_TYPE_PREFETCH,
            compute_stream);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtCmoAsync same failed, ret=", ret);
    }
}

}
