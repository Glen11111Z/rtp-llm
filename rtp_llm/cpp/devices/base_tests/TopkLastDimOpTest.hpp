#pragma once
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "3rdparty/trt_beam_search/topkLastDim.h"
#include <torch/torch.h>

using namespace rtp_llm;

class TopkLastDimOpTest: public DeviceTestBase {
public:
    torch::TensorOptions float_options = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
    torch::TensorOptions int_options   = torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU));

    void testTopkLastDim(int batchSize, int inputLength, int k, bool is_largest) {
        using namespace tensorrt_llm::kernels;

        // Generate random input
        auto input_tensor = torch::randn({batchSize, inputLength}, float_options);

        // Compute workspace size
        size_t workspace_size = invokeComputeTopkLastDimWorkspaceSize<float>(batchSize, inputLength, k, is_largest);

        // Allocate device buffers
        auto input_buf = tensorToBuffer(input_tensor, AllocationType::DEVICE);
        auto out_val =
            device_->allocateBuffer({DataType::TYPE_FP32, {(size_t)batchSize, (size_t)k}, AllocationType::DEVICE}, {});
        auto out_ind =
            device_->allocateBuffer({DataType::TYPE_INT32, {(size_t)batchSize, (size_t)k}, AllocationType::DEVICE}, {});
        auto workspace = device_->allocateBuffer({DataType::TYPE_UINT8, {workspace_size}, AllocationType::DEVICE}, {});

        // Run kernel on default stream
        invokeTopkLastDim<float>(batchSize,
                                 inputLength,
                                 k,
                                 is_largest,
                                 input_buf->data(),
                                 out_val->data(),
                                 out_ind->data(),
                                 workspace->data(),
                                 (cudaStream_t)0);
        device_->syncAndCheck();

        // Copy results back
        auto result_val = bufferToTensor(*out_val);
        auto result_ind = bufferToTensor(*out_ind);

        // Torch reference
        auto [ref_val, ref_ind] = torch::topk(input_tensor, k, /*dim=*/-1, /*largest=*/is_largest, /*sorted=*/true);

        // Sort both by index for stable comparison (kernel may not preserve order for equal values)
        auto [result_ind_sorted, result_perm] = result_ind.sort(-1);
        auto result_val_sorted                = result_val.gather(-1, result_perm);
        auto [ref_ind_sorted, ref_perm]       = ref_ind.sort(-1);
        auto ref_val_sorted                   = ref_val.gather(-1, ref_perm);

        assertTensorClose(result_val_sorted, ref_val_sorted);
        ASSERT_TRUE(torch::equal(result_ind_sorted.to(torch::kInt), ref_ind_sorted.to(torch::kInt)))
            << "Indices mismatch!\nGot: " << result_ind_sorted << "\nExpected: " << ref_ind_sorted;
    }
};
