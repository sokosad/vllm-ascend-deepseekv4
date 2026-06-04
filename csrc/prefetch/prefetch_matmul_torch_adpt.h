#pragma once

#include <torch/extension.h>
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <cstdlib>
#include <cstdio>

namespace vllm_ascend {

inline void npu_prefetch_async(
    const at::Tensor& weight,
    int64_t prefetch_size) {

    if (!weight.defined() || weight.numel() == 0 || prefetch_size <= 0) {
        return;
    }

    static bool log_enabled = std::getenv("VLLM_PREFETCH_LOG") != nullptr;

    aclrtStream compute_stream = c10_npu::getCurrentNPUStream();

    void* data_ptr = weight.data_ptr();
    size_t actual_size = static_cast<size_t>(prefetch_size);

    if (log_enabled) {
        std::printf("[prefetch-c++] aclrtCmoAsync size=%zu\n", actual_size);
        std::fflush(stdout);
    }

    aclError ret = aclrtCmoAsync(
        data_ptr, actual_size,
        ACL_RT_CMO_TYPE_PREFETCH,
        compute_stream);

    TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtCmoAsync failed, ret=", ret);
}

}
