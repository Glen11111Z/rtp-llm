#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/kernels/embedding_kernels.h"

#include <vector>

using namespace std;

namespace rtp_llm {

BufferPtr CudaDevice::embeddingLookup(const EmbeddingLookupParams& params) {
    const auto& tokens           = params.combo_tokens;
    const auto& embedding_table  = params.embedding_table;
    const auto& mask             = params.text_tokens_mask;
    const auto& position_ids     = params.position_ids;
    const auto& postition_table  = params.position_table;
    const auto& token_types      = params.token_types;
    const auto& token_type_table = params.token_type_table;

    const auto token_num   = tokens.size();
    const auto hidden_size = embedding_table.shape()[1];
    const auto data_type   = embedding_table.type();
    const auto vocab_size  = embedding_table.shape()[0];

    // Host-side validation: check token IDs are within vocab range before kernel launch.
    // NOTE: this introduces a D2H copy + sync per call; remove after debugging is done.
    {
        std::vector<int> host_tokens(token_num);
        cudaMemcpyAsync(host_tokens.data(), tokens.data<int>(),
                        token_num * sizeof(int), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // Build full prompt token dump string (for debugging when crash happens)
        auto build_full_tokens_str = [&]() {
            std::string s;
            s.reserve(token_num * 8);
            s += "[";
            for (size_t j = 0; j < token_num; ++j) {
                if (j > 0) s += ",";
                s += std::to_string(host_tokens[j]);
            }
            s += "]";
            return s;
        };

        bool has_invalid = false;
        for (size_t i = 0; i < token_num; ++i) {
            if (host_tokens[i] < 0 || (size_t)host_tokens[i] >= vocab_size) {
                RTP_LLM_LOG_ERROR(
                    "embeddingLookup: token id out of range! index=" + std::to_string(i)
                    + " token_id=" + std::to_string(host_tokens[i])
                    + " vocab_size=" + std::to_string(vocab_size)
                    + " token_num=" + std::to_string(token_num));
                has_invalid = true;
            }
        }
        if (has_invalid) {
            RTP_LLM_LOG_ERROR("embeddingLookup: full prompt tokens (token_num="
                              + std::to_string(token_num) + ", vocab_size="
                              + std::to_string(vocab_size) + ") = "
                              + build_full_tokens_str());
            throw std::runtime_error(
                "embeddingLookup: token id out of vocab range [0, "
                + std::to_string(vocab_size) + "), see ERROR log for full prompt");
        }
    }

    auto embeddings = allocateBuffer({data_type, {token_num, hidden_size}}, {"embedding"});

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                     invokeEmbeddingLookup,
                                     embeddings->data(),
                                     embedding_table.data(),
                                     params.input_embedding_scalar,
                                     postition_table.has_value() ? postition_table.value().get().data() : nullptr,
                                     token_type_table.has_value() ? token_type_table.value().get().data() : nullptr,
                                     tokens.data<int>(),
                                     position_ids.has_value() ? position_ids.value().get().data<int>() : nullptr,
                                     token_types.has_value() ? token_types.value().get().data<int>() : nullptr,
                                     mask.has_value() ? mask.value().get().data<int>() : nullptr,
                                     token_num,
                                     hidden_size,
                                     stream_);
    check_cuda_error();
    return embeddings;
}

}  // namespace rtp_llm
