#include "rtp_llm/cpp/devices/base_tests/TopkLastDimOpTest.hpp"
using namespace std;
using namespace rtp_llm;

class CudaTopkLastDimOpTest: public TopkLastDimOpTest {};

TEST_F(CudaTopkLastDimOpTest, basicTest) {
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

TEST_F(CudaTopkLastDimOpTest, smallestTest) {
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
