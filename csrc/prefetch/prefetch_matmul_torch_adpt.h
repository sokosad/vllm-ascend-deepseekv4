#pragma once

#include <torch/extension.h>
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <mutex>

struct PrefetchContext {
    bool initialized = false;
    aclrtStream prefetch_stream = nullptr;
    aclrtEvent prefetch_event = nullptr;
    std::mutex mtx;
};

static PrefetchContext g_prefetch_ctx;

static void ensure_prefetch_ctx() {
    std::lock_guard<std::mutex> lock(g_prefetch_ctx.mtx);
    if (!g_prefetch_ctx.initialized) {
        aclError ret = aclrtCreateStream(&g_prefetch_ctx.prefetch_stream);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtCreateStream failed, ret=", ret);
        ret = aclrtCreateEvent(&g_prefetch_ctx.prefetch_event);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtCreateEvent failed, ret=", ret);
        g_prefetch_ctx.initialized = true;
    }
}

namespace vllm_ascend {

inline void npu_prefetch_async(
    const at::Tensor& weight,
    int64_t prefetch_size) {

    if (!weight.defined() || weight.numel() == 0 || prefetch_size <= 0) {
        return;
    }

    ensure_prefetch_ctx();

    aclrtStream compute_stream = c10_npu::getCurrentNPUStream();

    void* data_ptr = weight.data_ptr();
    size_t actual_size = static_cast<size_t>(prefetch_size);

    aclError ret;

    ret = aclrtEventRecord(g_prefetch_ctx.prefetch_event, compute_stream);
    TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtEventRecord failed, ret=", ret);

    ret = aclrtStreamWaitEvent(g_prefetch_ctx.prefetch_stream, g_prefetch_ctx.prefetch_event);
    TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtStreamWaitEvent failed, ret=", ret);

    ret = aclrtCmoAsync(data_ptr, actual_size,
                         ACL_RT_CMO_TYPE_PREFETCH,
                         g_prefetch_ctx.prefetch_stream);
    TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtCmoAsync failed, ret=", ret);
}

}
