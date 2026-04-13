#include "rtp_llm/cpp/devices/base_tests/TopkLastDimOpTest.hpp"
#include <gtest/gtest.h>
using namespace std;
using namespace rtp_llm;

// Explicit main() to override the one exported by module_moe_sorting.so
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

class ROCmTopkLastDimOpTest: public TopkLastDimOpTest {};

TEST_F(ROCmTopkLastDimOpTest, basicTest) {
    std::vector<int> batch_sizes   = {1, 2, 8, 32};
    std::vector<int> input_lengths = {10, 100, 1000, 7000};
    std::vector<int> ks            = {1, 2, 5, 10, 64, 128};

    for (auto batch_size : batch_sizes) {
        for (auto input_length : input_lengths) {
            for (auto k : ks) {
                if (k > input_length)
                    continue;
                std::cout << "batch_size: " << batch_size << ", input_length: " << input_length << ", k: " << k
                          << ", is_largest: true" << std::endl;
                testTopkLastDim(batch_size, input_length, k, true);
            }
        }
    }
}

TEST_F(ROCmTopkLastDimOpTest, perfTest) {
    using namespace tensorrt_llm::kernels;

    int              inputLength = 217216;
    int              k           = 1400;
    std::vector<int> batch_sizes = {512, 1024, 1536, 2048, 2560, 3072};
    int              warmup      = 3;
    int              iters       = 10;

    for (auto batchSize : batch_sizes) {
        auto   input_tensor   = torch::randn({batchSize, inputLength}, float_options);
        size_t workspace_size = invokeComputeTopkLastDimWorkspaceSize<float>(batchSize, inputLength, k, true);
        auto   input_buf      = tensorToBuffer(input_tensor, AllocationType::DEVICE);
        auto   out_val =
            device_->allocateBuffer({DataType::TYPE_FP32, {(size_t)batchSize, (size_t)k}, AllocationType::DEVICE}, {});
        auto out_ind =
            device_->allocateBuffer({DataType::TYPE_INT32, {(size_t)batchSize, (size_t)k}, AllocationType::DEVICE}, {});
        auto workspace = device_->allocateBuffer({DataType::TYPE_UINT8, {workspace_size}, AllocationType::DEVICE}, {});

        // warmup
        for (int i = 0; i < warmup; i++) {
            invokeTopkLastDim<float>(batchSize,
                                     inputLength,
                                     k,
                                     true,
                                     input_buf->data(),
                                     out_val->data(),
                                     out_ind->data(),
                                     workspace->data(),
                                     (cudaStream_t)0);
            hipDeviceSynchronize();
        }

        // timed runs
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            invokeTopkLastDim<float>(batchSize,
                                     inputLength,
                                     k,
                                     true,
                                     input_buf->data(),
                                     out_val->data(),
                                     out_ind->data(),
                                     workspace->data(),
                                     (cudaStream_t)0);
            hipDeviceSynchronize();
        }
        auto   end    = std::chrono::high_resolution_clock::now();
        double avg_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)iters;
        std::cout << "batch_size=" << batchSize << " input_length=" << inputLength << " k=" << k
                  << " avg_time=" << avg_us << "us" << std::endl;
    }
}

// Common production shapes: beam search with typical vocab sizes and beam widths
TEST_F(ROCmTopkLastDimOpTest, commonPerfTest) {
    using namespace tensorrt_llm::kernels;

    struct TestCase {
        int batch_size;
        int input_length;
        int k;
    };
    std::vector<TestCase> cases = {
        // Typical beam search: batch * beam_width rows, vocab_size cols, top-k = beam_width
        {4, 32000, 4},    // bs=1, beam=4, vocab=32k (LLaMA)
        {16, 32000, 4},   // bs=4, beam=4, vocab=32k
        {4, 128256, 4},   // bs=1, beam=4, vocab=128k (LLaMA3)
        {16, 128256, 4},  // bs=4, beam=4, vocab=128k
        {4, 151936, 4},   // bs=1, beam=4, vocab=152k (Qwen2.5)
        {16, 151936, 4},  // bs=4, beam=4, vocab=152k
        // Larger beam widths
        {8, 32000, 8},   // bs=1, beam=8, vocab=32k
        {32, 32000, 8},  // bs=4, beam=8, vocab=32k
        {8, 128256, 8},  // bs=1, beam=8, vocab=128k
        // Top-p/top-k sampling with larger k
        {1, 32000, 50},    // top-50 sampling
        {1, 128256, 50},   // top-50 sampling, large vocab
        {8, 32000, 50},    // batched top-50
        {1, 32000, 256},   // top-256 sampling
        {8, 128256, 256},  // batched top-256, large vocab
    };
    int warmup = 3;
    int iters  = 10;

    for (auto& tc : cases) {
        auto   input_tensor = torch::randn({tc.batch_size, tc.input_length}, float_options);
        size_t workspace_size =
            invokeComputeTopkLastDimWorkspaceSize<float>(tc.batch_size, tc.input_length, tc.k, true);
        auto input_buf = tensorToBuffer(input_tensor, AllocationType::DEVICE);
        auto out_val   = device_->allocateBuffer(
            {DataType::TYPE_FP32, {(size_t)tc.batch_size, (size_t)tc.k}, AllocationType::DEVICE}, {});
        auto out_ind = device_->allocateBuffer(
            {DataType::TYPE_INT32, {(size_t)tc.batch_size, (size_t)tc.k}, AllocationType::DEVICE}, {});
        auto workspace = device_->allocateBuffer({DataType::TYPE_UINT8, {workspace_size}, AllocationType::DEVICE}, {});

        for (int i = 0; i < warmup; i++) {
            invokeTopkLastDim<float>(tc.batch_size,
                                     tc.input_length,
                                     tc.k,
                                     true,
                                     input_buf->data(),
                                     out_val->data(),
                                     out_ind->data(),
                                     workspace->data(),
                                     (cudaStream_t)0);
            hipDeviceSynchronize();
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            invokeTopkLastDim<float>(tc.batch_size,
                                     tc.input_length,
                                     tc.k,
                                     true,
                                     input_buf->data(),
                                     out_val->data(),
                                     out_ind->data(),
                                     workspace->data(),
                                     (cudaStream_t)0);
            hipDeviceSynchronize();
        }
        auto   end    = std::chrono::high_resolution_clock::now();
        double avg_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)iters;
        std::cout << "batch_size=" << tc.batch_size << " input_length=" << tc.input_length << " k=" << tc.k
                  << " avg_time=" << avg_us << "us" << std::endl;
    }
}

TEST_F(ROCmTopkLastDimOpTest, smallestTest) {
    std::vector<int> batch_sizes   = {1, 4, 16};
    std::vector<int> input_lengths = {10, 500, 5000};
    std::vector<int> ks            = {1, 5, 32};

    for (auto batch_size : batch_sizes) {
        for (auto input_length : input_lengths) {
            for (auto k : ks) {
                if (k > input_length)
                    continue;
                std::cout << "batch_size: " << batch_size << ", input_length: " << input_length << ", k: " << k
                          << ", is_largest: false" << std::endl;
                testTopkLastDim(batch_size, input_length, k, false);
            }
        }
    }
}
