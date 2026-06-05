#include "rtp_llm/cpp/models/Sampler.h"
#include <cstring>
#include <cmath>
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <unordered_set>
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <cfloat>

using namespace std;

namespace rtp_llm {

Sampler::Sampler(const SamplerInitParams& params) {}

SamplerOutput Sampler::forward(const SamplerInputs& inputs) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    RTP_LLM_PROFILE_SCOPE("sampler.forward");
    // Helper: narrow a tensor if defined, else return undefined tensor
    auto mayNarrow = [](const torch::Tensor& t, int64_t offset, int64_t size) -> torch::Tensor {
        return t.defined() ? t.narrow(0, offset, size) : torch::Tensor();
    };

    // Helper: convert optional tensor slice to std::optional<torch::Tensor>
    auto mayOptNarrow = [](const torch::Tensor& t, int64_t offset, int64_t size) -> std::optional<torch::Tensor> {
        return t.defined() ? std::optional<torch::Tensor>(t.narrow(0, offset, size)) : std::nullopt;
    };

    // =========================================================================
    // Score correction for constraint decoding:
    // Save full-vocab log_softmax BEFORE constraint masking so we can correct
    // cum_log_probs after beam search to reflect the model's true preference
    // rather than the biased probability caused by different constraint set sizes.
    // =========================================================================
    uint64_t max_seq_len   = inputs.token_ids.size(1);
    auto     num_beams_in  = inputs.num_beams_in.data_ptr<int64_t>();
    auto     num_beams_out = inputs.num_beams_out.data_ptr<int64_t>();

    bool has_num_beams = std::any_of(num_beams_in, num_beams_in + inputs.batch_size, [](auto n) { return n > 1; })
                         || std::any_of(num_beams_out, num_beams_out + inputs.batch_size, [](auto n) { return n > 1; });

    torch::Tensor full_log_softmax_saved;  // [batch_size, vocab_size] on CUDA
    if (has_num_beams && inputs.logits_processor_states_ptr != nullptr) {
        full_log_softmax_saved = at::log_softmax(inputs.logits.to(torch::kCUDA), -1);
    }

    preprocessLogits(inputs);

    // After masking: compute masked log_softmax for delta correction
    torch::Tensor masked_log_softmax_saved;  // [batch_size, vocab_size] on CUDA
    if (full_log_softmax_saved.defined()) {
        masked_log_softmax_saved = at::log_softmax(inputs.logits.to(torch::kCUDA), -1);
    }

    bool variable_num_beams = inputs.batch_size != inputs.batch_size_out;

    // allocate output tensors
    // Keep success on CUDA to avoid a blocking D2H copy: the GPU sampling kernel writes success
    // directly, and callers that need CPU access should call .cpu() explicitly.
    auto all_success =
        torch::empty({(int64_t)inputs.batch_size}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto all_beam_indices =
        has_num_beams ? torch::empty({(int64_t)inputs.batch_size_out}, torch::kInt32) : torch::Tensor();
    // Move token_ids to CUDA once so sampleGreedy writes GPU→GPU (no blocking D2H sync).
    // Callers that need CPU access should call .cpu() explicitly.
    // Use blocking transfer: on ROCm, hipMemcpyAsync from pageable memory is truly async
    // and can cause memory access faults if a kernel reads the buffer before transfer completes.
    auto inputs_token_ids_cuda = inputs.token_ids.to(torch::kCUDA);
    auto all_token_ids_out     = variable_num_beams ?
                                     torch::empty({(int64_t)inputs.batch_size_out, (int64_t)max_seq_len},
                                              torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)) :
                                     inputs_token_ids_cuda;
    auto all_cum_log_probs_out = variable_num_beams && inputs.cum_log_probs.defined() ?
                                     torch::empty({(int64_t)inputs.batch_size_out}, torch::kFloat32) :
                                     inputs.cum_log_probs;

    // Track effective beam sizes per stream for constrained decoding
    std::unordered_map<size_t, int> effective_beam_sizes;

    size_t from_batch_idx_in = 0, to_batch_idx_in = 0;
    size_t from_batch_idx_out = 0;

    while (from_batch_idx_in < inputs.batch_size) {
        auto cur_num_beams_in  = num_beams_in[from_batch_idx_in];
        auto cur_num_beams_out = num_beams_out[from_batch_idx_in];
        ++to_batch_idx_in;
        while (to_batch_idx_in < inputs.batch_size && num_beams_in[to_batch_idx_in] == cur_num_beams_in
               && num_beams_out[to_batch_idx_in] == cur_num_beams_out) {
            ++to_batch_idx_in;
        }

        // now from_batch_idx to to_batch_idx have the same beam size, sample once.
        const auto batch_size_in    = to_batch_idx_in - from_batch_idx_in;
        const auto beam_batch_size  = batch_size_in / cur_num_beams_in;
        const auto batch_size_out   = beam_batch_size * cur_num_beams_out;
        const auto to_batch_idx_out = from_batch_idx_out + batch_size_out;

        auto success           = all_success.narrow(0, from_batch_idx_in, batch_size_in);
        auto logits            = inputs.logits.narrow(0, from_batch_idx_in, batch_size_in);
        auto token_ids_in      = inputs_token_ids_cuda.narrow(0, from_batch_idx_in, batch_size_in);
        auto token_ids_out     = all_token_ids_out.narrow(0, from_batch_idx_out, batch_size_out);
        auto input_lengths     = inputs.input_lengths.narrow(0, from_batch_idx_in, batch_size_in);
        auto sequence_lengths  = inputs.sequence_lengths.narrow(0, from_batch_idx_in, batch_size_in);
        auto cum_log_probs_in  = mayNarrow(inputs.cum_log_probs, from_batch_idx_in, batch_size_in);
        auto cum_log_probs_out = mayNarrow(all_cum_log_probs_out, from_batch_idx_out, batch_size_out);

        if (cur_num_beams_in == 1 && cur_num_beams_out == 1) {
            const auto decoder_batch_size = (int64_t)inputs.sequence_lengths.size(0);
            auto       sequence_lengths_in =
                (int64_t)from_batch_idx_in < decoder_batch_size ?
                          inputs.sequence_lengths.narrow(
                        0,
                        from_batch_idx_in,
                        min((int64_t)batch_size_in, decoder_batch_size - (int64_t)from_batch_idx_in)) :
                          torch::empty({0}, torch::kInt32);

            // TODO(zhangjianning.zjn): would be better to eliminate the copy
            if (cum_log_probs_out.defined() && cum_log_probs_in.defined()) {
                cum_log_probs_out.copy_(cum_log_probs_in);
            }

            auto top_k                = inputs.top_k.narrow(0, from_batch_idx_in, batch_size_in);
            auto top_p                = inputs.top_p.narrow(0, from_batch_idx_in, batch_size_in);
            auto temperature          = inputs.temperature.narrow(0, from_batch_idx_in, batch_size_in);
            auto repetition_penalty   = mayOptNarrow(inputs.repetition_penalty, from_batch_idx_in, batch_size_in);
            auto presence_penalty     = mayOptNarrow(inputs.presence_penalty, from_batch_idx_in, batch_size_in);
            auto frequency_penalty    = mayOptNarrow(inputs.frequency_penalty, from_batch_idx_in, batch_size_in);
            auto no_repeat_ngram_size = mayOptNarrow(inputs.no_repeat_ngram_size, from_batch_idx_in, batch_size_in);
            auto all_probs            = mayOptNarrow(inputs.all_probs, from_batch_idx_in, batch_size_in);
            auto do_sample            = mayOptNarrow(inputs.do_sample, from_batch_idx_in, batch_size_in);
            auto generator            = std::vector<at::Generator>{inputs.generator.begin() + from_batch_idx_in,
                                                                   inputs.generator.begin() + from_batch_idx_in + batch_size_in};

            RTP_LLM_PROFILE_SCOPE("sampler.forward.execSampleGreedy");
            auto greedy_output = execSampleGreedy(
                {logits,
                 input_lengths,
                 sequence_lengths_in,
                 token_ids_in,
                 inputs.step,
                 top_k,
                 top_p,
                 temperature,
                 repetition_penalty,
                 no_repeat_ngram_size,
                 cum_log_probs_out.defined() ? std::optional<torch::Tensor>(cum_log_probs_out) : std::nullopt,
                 std::nullopt,  // output_log_probs
                 all_probs,
                 presence_penalty,
                 frequency_penalty,
                 do_sample,
                 generator});
            if (greedy_output.success.defined()) {
                success.copy_(greedy_output.success);
                // TODO(zhangjianning.zjn): would be better to eliminate the copy
                if (variable_num_beams) {
                    token_ids_out.copy_(token_ids_in);
                }
            } else {
                success.fill_(true);
            }
        } else {
            RTP_LLM_CHECK_WITH_INFO((batch_size_in % cur_num_beams_in == 0),
                                    "sample_batch_size[%d] must devide by current_num_beams_in[%d]");

            const size_t vocab_size      = inputs.logits.size(1);
            const size_t max_seq_len_val = inputs.token_ids.size(1);

            auto beam_indices = all_beam_indices.narrow(0, from_batch_idx_out, batch_size_out);

            // Reshape for beam search: [batch, beams, ...]
            auto logits_reshaped =
                logits.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in, (int64_t)vocab_size});
            auto token_ids_in_reshaped =
                token_ids_in.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in, (int64_t)max_seq_len_val});
            auto input_lengths_reshaped = input_lengths.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in});
            auto sequence_lengths_reshaped =
                sequence_lengths.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in});
            auto cum_log_probs_in_reshaped =
                cum_log_probs_in.defined() ?
                    cum_log_probs_in.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in}) :
                    torch::zeros({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in});

            auto logits_t           = logits_reshaped.to(torch::kCUDA);
            auto token_ids_in_t     = token_ids_in_reshaped.to(torch::kCUDA);
            auto input_lengths_t    = input_lengths_reshaped.to(torch::kCUDA);
            auto sequence_lengths_t = sequence_lengths_reshaped.to(torch::kCUDA);
            auto cum_log_probs_in_t = cum_log_probs_in_reshaped.to(torch::kCUDA);

            auto output = execSampleBeamSearch({logits_t,
                                                token_ids_in_t,
                                                input_lengths_t,
                                                sequence_lengths_t,
                                                cum_log_probs_in_t,
                                                (size_t)cur_num_beams_out});

            auto token_ids_out_reshaped =
                token_ids_out.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_out, (int64_t)max_seq_len_val});
            auto cum_log_probs_out_reshaped =
                cum_log_probs_out.defined() ?
                    cum_log_probs_out.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_out}) :
                    torch::Tensor();
            
            token_ids_out_reshaped.copy_(output.token_ids);
            if (cum_log_probs_out_reshaped.defined()) {
                cum_log_probs_out_reshaped.copy_(output.cum_log_probs);
            }
            beam_indices.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_out}).copy_(output.beam_indices);

            // =================================================================
            // Score correction: replace masked log_prob contribution with full-vocab log_prob
            // for unbiased ranking under constraint decoding.
            //
            // delta[beam] = full_log_softmax[src_beam][token] - masked_log_softmax[src_beam][token]
            // corrected_cum_log_probs[beam] = output.cum_log_probs[beam] + delta
            // =================================================================
            if (full_log_softmax_saved.defined() && output.new_token_ids.defined()
                && cum_log_probs_out_reshaped.defined()) {
                const int64_t vocab_size_val = static_cast<int64_t>(inputs.logits.size(1));

                // Slice saved log_softmax tensors for this batch range [batch_size_in, vocab]
                auto full_lsp_flat = full_log_softmax_saved.narrow(0, from_batch_idx_in, batch_size_in);
                auto masked_lsp_flat = masked_log_softmax_saved.narrow(0, from_batch_idx_in, batch_size_in);

                // Compute flat source indices: flat_src[b][i] = b * beams_in + beam_src[b][i]
                auto beam_src_gpu = output.beam_indices;  // [beam_batch, beams_out] CUDA int32
                auto batch_offsets = torch::arange((int64_t)beam_batch_size,
                                                   torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA))
                                         * (int32_t)cur_num_beams_in;
                auto flat_src = (beam_src_gpu + batch_offsets.unsqueeze(1)).flatten().to(torch::kLong);

                // Token indices
                auto flat_tokens = output.new_token_ids.flatten().to(torch::kLong);  // [beam_batch*beams_out]

                // Gather individual log_prob values using advanced indexing
                auto full_values = full_lsp_flat.index({flat_src, flat_tokens});    // [beam_batch*beams_out]
                auto masked_values = masked_lsp_flat.index({flat_src, flat_tokens}); // [beam_batch*beams_out]
                auto delta = (full_values - masked_values).cpu();  // bring to CPU

                // Apply correction to cum_log_probs_out (which is on CPU)
                auto* delta_ptr = delta.data_ptr<float>();
                auto* clp_ptr = cum_log_probs_out_reshaped.data_ptr<float>();
                const size_t n_out = beam_batch_size * cur_num_beams_out;
                for (size_t idx = 0; idx < n_out; ++idx) {
                    // Skip beams with -inf scores (completely invalid paths)
                    if (clp_ptr[idx] <= -FLT_MAX / 2) continue;
                    // Skip if delta is nan/inf (can happen with -inf logits)
                    if (!std::isfinite(delta_ptr[idx])) continue;
                    clp_ptr[idx] += delta_ptr[idx];
                }
            }
 
            // Filter out beams with -inf cum_log_probs (produced by constrained decoding masks).
            // For each batch item, compact valid beams to the front and update beam count.
            auto* cum_log_probs_data = cum_log_probs_out.data_ptr<float>();
            auto* token_ids_data     = token_ids_out.data_ptr<int32_t>();
            auto* beam_indices_data  = beam_indices.data_ptr<int32_t>();
            for (size_t batch_idx = 0; batch_idx < beam_batch_size; ++batch_idx) {
                size_t valid_beam_count = 0;
                for (size_t beam_idx = 0; beam_idx < cur_num_beams_out; ++beam_idx) {
                    const size_t src_offset = batch_idx * cur_num_beams_out + beam_idx;
                    if (cum_log_probs_data[src_offset] <=-FLT_MAX) {
                        continue;
                    }
                    if (valid_beam_count != beam_idx) {
                        const size_t dst_offset = batch_idx * cur_num_beams_out + valid_beam_count;
                        cum_log_probs_data[dst_offset] = cum_log_probs_data[src_offset];
                        beam_indices_data[dst_offset]  = beam_indices_data[src_offset];
                        std::memcpy(token_ids_data + dst_offset * max_seq_len,
                                    token_ids_data + src_offset * max_seq_len,
                                    max_seq_len * sizeof(int32_t));
                    }
                    ++valid_beam_count;
                }
                // Keep at least the first beam to avoid empty output.
                if (valid_beam_count == 0) {
                    valid_beam_count = 1;
                }
                for (size_t out_idx = 0; out_idx < cur_num_beams_out; ++out_idx) {
                    size_t global_out_idx = from_batch_idx_out + batch_idx * cur_num_beams_out + out_idx;
                    effective_beam_sizes[global_out_idx] = valid_beam_count;
                }
            }

            success.fill_(true);
        }

        // prepare for next sampling
        from_batch_idx_in  = to_batch_idx_in;
        from_batch_idx_out = to_batch_idx_out;
    }
    // TODO(xinfei.sxf) 优化copy token_ids
    SamplerOutput sampler_output;
    sampler_output.token_ids      = std::move(all_token_ids_out);
    sampler_output.cum_log_probs  = std::move(all_cum_log_probs_out);
    sampler_output.all_probs      = std::move(inputs.all_probs);
    sampler_output.beam_index     = std::move(all_beam_indices);
    sampler_output.success        = std::move(all_success);
    sampler_output.effective_beam_sizes = std::move(effective_beam_sizes);
    return sampler_output;
}

void Sampler::preprocessLogits(const SamplerInputs& inputs) {
    if (inputs.logits_processor_states_ptr != nullptr) {
        inputs.logits_processor_states_ptr->batchProcess(inputs);
    }
}

}  // namespace rtp_llm
