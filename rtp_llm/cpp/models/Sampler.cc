#include "rtp_llm/cpp/models/Sampler.h"
#include <cstring>
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <algorithm>
#include <exception>
#include <unordered_set>
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

using namespace std;

namespace rtp_llm {

Sampler::Sampler(const SamplerInitParams& params):
    fixed_max_batch_size_(params.max_batch_size > 0 && params.fixed_max_batch_size) {
    if (params.max_batch_size > 0) {
        allocateGreedySamplingBuffers(params.max_batch_size);
    }
}

void Sampler::allocateGreedySamplingBuffers(size_t max_batch_size) {
    waitGreedySamplingBufferEvents();
    max_batch_size_ = max_batch_size;
    auto pinned_i64 = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    for (auto& slot : greedy_sampling_buffer_slots_) {
        auto& buffers                = slot.buffers;
        buffers.seed_host            = torch::empty({(int64_t)max_batch_size_}, pinned_i64);
        buffers.offset_host          = torch::empty({(int64_t)max_batch_size_}, pinned_i64);
        buffers.output_ids_ptrs_host = torch::empty({(int64_t)max_batch_size_}, pinned_i64);
        buffers.max_batch_size       = max_batch_size_;
        slot.ready_event.reset();
    }
}

void Sampler::ensureGreedySamplingBuffers(size_t batch_size) {
    if (batch_size <= max_batch_size_) {
        return;
    }
    // Fixed users fail fast on impossible batch sizes. Dynamic users wait for all
    // pending slot events before rebuilding every slot together.
    RTP_LLM_CHECK_WITH_INFO(!fixed_max_batch_size_,
                            "sampler batch size [%lu] exceeds initialized max batch size [%lu]",
                            batch_size,
                            max_batch_size_);
    RTP_LLM_LOG_INFO("grow greedy sampling buffers from batch size [%lu] to [%lu]", max_batch_size_, batch_size);
    allocateGreedySamplingBuffers(batch_size);
}

void Sampler::waitGreedySamplingBufferEvents() {
    for (auto& slot : greedy_sampling_buffer_slots_) {
        if (slot.ready_event) {
            slot.ready_event->synchronize();
            slot.ready_event.reset();
        }
    }
}

GreedySamplingBuffers& Sampler::nextGreedySamplingBuffers(size_t batch_size) {
    ensureGreedySamplingBuffers(batch_size);
    auto& slot = greedy_sampling_buffer_slots_[greedy_sampling_buffer_index_];
    if (slot.ready_event) {
        slot.ready_event->synchronize();
        slot.ready_event.reset();
    }
    current_greedy_sampling_slot_ = &slot;
    greedy_sampling_buffer_index_ = (greedy_sampling_buffer_index_ + 1) % greedy_sampling_buffer_slots_.size();
    return slot.buffers;
}

void Sampler::markGreedySamplingBufferReady() {
    if (current_greedy_sampling_slot_ != nullptr) {
        auto* slot = current_greedy_sampling_slot_;
        try {
            slot->ready_event             = runtimeCreateEvent();
            current_greedy_sampling_slot_ = nullptr;
        } catch (...) {
            current_greedy_sampling_slot_ = nullptr;
            slot->ready_event.reset();
            runtimeSyncAndCheck();
            throw;
        }
    }
}

torch::Tensor Sampler::candidateTokenIdsOnLogitsDevice(const torch::Tensor& candidate_token_ids,
                                                       const torch::Device& logits_device) {
    auto candidate_token_ids_host = candidate_token_ids.to(torch::kCPU).to(torch::kLong).contiguous();
    std::vector<int64_t> candidate_token_ids_vec(candidate_token_ids_host.data_ptr<int64_t>(),
                                                 candidate_token_ids_host.data_ptr<int64_t>()
                                                     + candidate_token_ids_host.numel());
    auto logits_device_str = logits_device.str();

    if (cached_candidate_token_ids_device_tensor_.defined()
        && cached_candidate_token_ids_device_ == logits_device_str
        && cached_candidate_token_ids_host_ == candidate_token_ids_vec) {
        return cached_candidate_token_ids_device_tensor_;
    }

    cached_candidate_token_ids_host_          = std::move(candidate_token_ids_vec);
    cached_candidate_token_ids_device_        = logits_device_str;
    cached_candidate_token_ids_device_tensor_ = candidate_token_ids_host.to(logits_device);
    RTP_LLM_LOG_INFO("[PERF] sampler_candidate_token_ids_cache: refreshed=1, candidate_num=%ld, device=%s",
                     cached_candidate_token_ids_device_tensor_.numel(),
                     cached_candidate_token_ids_device_.c_str());
    return cached_candidate_token_ids_device_tensor_;
}

SamplerOutput Sampler::forward(const SamplerInputs& inputs) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    RTP_LLM_PROFILE_SCOPE("sampler.forward");
    RTP_LLM_CHECK_WITH_INFO(!forward_in_progress_.exchange(true),
                            "Sampler::forward is single-threaded and must not be called concurrently or reentrantly");
    struct SamplerForwardGuard {
        std::atomic<bool>& forward_in_progress;
        ~SamplerForwardGuard() {
            forward_in_progress.store(false);
        }
    } sampler_forward_guard{forward_in_progress_};

    // Helper: narrow a tensor if defined, else return undefined tensor
    auto mayNarrow = [](const torch::Tensor& t, int64_t offset, int64_t size) -> torch::Tensor {
        return t.defined() ? t.narrow(0, offset, size) : torch::Tensor();
    };

    // Helper: convert optional tensor slice to std::optional<torch::Tensor>
    auto mayOptNarrow = [](const torch::Tensor& t, int64_t offset, int64_t size) -> std::optional<torch::Tensor> {
        return t.defined() ? std::optional<torch::Tensor>(t.narrow(0, offset, size)) : std::nullopt;
    };

    auto tensorAllInt32Equal = [](const torch::Tensor& t, int32_t expected) -> bool {
        if (!t.defined()) {
            return true;
        }
        auto data = t.data_ptr<int32_t>();
        return std::all_of(data, data + t.numel(), [expected](int32_t value) { return value == expected; });
    };
    auto tensorAllFloatEqual = [](const torch::Tensor& t, float expected) -> bool {
        if (!t.defined()) {
            return true;
        }
        auto data = t.data_ptr<float>();
        return std::all_of(data, data + t.numel(), [expected](float value) { return value == expected; });
    };

    int64_t forward_start_us     = autil::TimeUtility::currentTimeInMicroSeconds();
    int64_t preprocess_logits_us = 0;
    int64_t prepare_output_us      = 0;
    int64_t alloc_success_us       = 0;
    int64_t alloc_beam_indices_us  = 0;
    int64_t token_ids_to_cuda_us   = 0;
    int64_t alloc_token_ids_out_us = 0;
    int64_t alloc_cum_log_probs_us = 0;
    int64_t greedy_buffer_get_us   = 0;
    int64_t greedy_exec_us         = 0;
    int64_t beam_exec_us           = 0;
    int64_t postprocess_us         = 0;
    int64_t candidate_gather_us      = 0;
    int64_t candidate_logits_cpu_us  = 0;
    int64_t candidate_cpu_select_us  = 0;
    int64_t candidate_argmax_us      = 0;
    int64_t candidate_map_us         = 0;
    size_t  greedy_group_count     = 0;
    size_t  beam_group_count       = 0;

    int64_t stage_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
    preprocessLogits(inputs);
    preprocess_logits_us = autil::TimeUtility::currentTimeInMicroSeconds() - stage_start_us;

    uint64_t max_seq_len   = inputs.token_ids.size(1);
    auto     num_beams_in  = inputs.num_beams_in.data_ptr<int64_t>();
    auto     num_beams_out = inputs.num_beams_out.data_ptr<int64_t>();

    bool has_num_beams = std::any_of(num_beams_in, num_beams_in + inputs.batch_size, [](auto n) { return n > 1; })
                         || std::any_of(num_beams_out, num_beams_out + inputs.batch_size, [](auto n) { return n > 1; });
    bool variable_num_beams = inputs.batch_size != inputs.batch_size_out;
    bool has_logits_processor = inputs.logits_processor_states_ptr != nullptr
                                && !inputs.logits_processor_states_ptr->empty();
    bool has_candidate_token_ids = inputs.candidate_token_ids.defined() && inputs.candidate_token_ids.numel() > 0;
    bool simple_greedy_fast_path = !has_num_beams && !variable_num_beams && !has_logits_processor
                                   && !inputs.return_original_all_probs && !inputs.all_probs.defined()
                                   && !inputs.cum_log_probs.defined() && tensorAllInt32Equal(inputs.top_k, 1)
                                   && tensorAllFloatEqual(inputs.repetition_penalty, 1.0f)
                                   && tensorAllFloatEqual(inputs.presence_penalty, 0.0f)
                                   && tensorAllFloatEqual(inputs.frequency_penalty, 0.0f)
                                   && tensorAllInt32Equal(inputs.no_repeat_ngram_size, 0);

    if (simple_greedy_fast_path) {
        bool   candidate_greedy_fast_path = has_candidate_token_ids && inputs.candidate_tokens_same_batch;
        size_t candidate_num              = has_candidate_token_ids ? inputs.candidate_token_ids.numel() : 0;

        int64_t prepare_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
        stage_start_us           = autil::TimeUtility::currentTimeInMicroSeconds();
        auto all_success = torch::empty({(int64_t)inputs.batch_size},
                                        torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
        all_success.fill_(true);
        alloc_success_us = autil::TimeUtility::currentTimeInMicroSeconds() - stage_start_us;
        prepare_output_us = autil::TimeUtility::currentTimeInMicroSeconds() - prepare_start_us;

        stage_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
        torch::Tensor new_token_ids;
        if (candidate_greedy_fast_path) {
            int64_t candidate_stage_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
            auto candidate_token_ids = candidateTokenIdsOnLogitsDevice(inputs.candidate_token_ids, inputs.logits.device());
            auto candidate_logits = inputs.logits.index_select(1, candidate_token_ids);
            candidate_gather_us = autil::TimeUtility::currentTimeInMicroSeconds() - candidate_stage_start_us;

            candidate_stage_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
            auto candidate_logits_cpu = candidate_logits.cpu().to(torch::kFloat32).contiguous();
            candidate_logits_cpu_us = autil::TimeUtility::currentTimeInMicroSeconds() - candidate_stage_start_us;

            candidate_stage_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
            auto candidate_token_ids_cpu = inputs.candidate_token_ids.to(torch::kCPU).to(torch::kLong).contiguous();
            auto new_token_ids_cpu = torch::empty({(int64_t)inputs.batch_size, 1}, torch::kInt32);
            const auto* candidate_logits_data = candidate_logits_cpu.data_ptr<float>();
            const auto* candidate_token_ids_data = candidate_token_ids_cpu.data_ptr<int64_t>();
            auto*       new_token_ids_data = new_token_ids_cpu.data_ptr<int32_t>();
            const auto  candidate_stride = candidate_logits_cpu.size(1);
            for (size_t batch_idx = 0; batch_idx < inputs.batch_size; ++batch_idx) {
                const auto* row = candidate_logits_data + batch_idx * candidate_stride;
                int64_t     best_candidate_idx = 0;
                float       best_score = row[0];
                for (int64_t candidate_idx = 1; candidate_idx < candidate_stride; ++candidate_idx) {
                    if (row[candidate_idx] > best_score) {
                        best_score = row[candidate_idx];
                        best_candidate_idx = candidate_idx;
                    }
                }
                new_token_ids_data[batch_idx] = static_cast<int32_t>(candidate_token_ids_data[best_candidate_idx]);
            }
            new_token_ids = new_token_ids_cpu;
            candidate_cpu_select_us = autil::TimeUtility::currentTimeInMicroSeconds() - candidate_stage_start_us;
        } else {
            candidate_num = 0;
            new_token_ids =
                torch::argmax(inputs.logits, -1).to(torch::kInt32).reshape({(int64_t)inputs.batch_size, 1});
        }
        greedy_exec_us     = autil::TimeUtility::currentTimeInMicroSeconds() - stage_start_us;
        greedy_group_count = 1;

        int64_t total_us = autil::TimeUtility::currentTimeInMicroSeconds() - forward_start_us;
        RTP_LLM_LOG_INFO(
            "[PERF] sampler_forward_detail: total_us=%ld, preprocess_logits_us=%ld, prepare_output_us=%ld, "
            "alloc_success_us=%ld, alloc_beam_indices_us=%ld, token_ids_to_cuda_us=%ld, "
            "alloc_token_ids_out_us=%ld, alloc_cum_log_probs_us=%ld, greedy_buffer_get_us=%ld, "
            "greedy_exec_us=%ld, beam_exec_us=%ld, postprocess_us=%ld, batch_size=%zu, batch_size_out=%zu, "
            "vocab_size=%zu, step=%zu, greedy_groups=%zu, beam_groups=%zu, has_logits_processor=%d, "
            "has_num_beams=%d, variable_num_beams=%d, return_original_all_probs=%d, all_probs_defined=%d, "
            "do_sample_defined=%d, top_k_defined=%d, top_p_defined=%d, temperature_defined=%d, "
            "simple_greedy_fast_path=%d, candidate_sampling=%d, candidate_num=%zu, candidate_same_batch=%d, "
            "candidate_gather_us=%ld, candidate_logits_cpu_us=%ld, candidate_cpu_select_us=%ld, "
            "candidate_argmax_us=%ld, candidate_map_us=%ld",
            total_us,
            preprocess_logits_us,
            prepare_output_us,
            alloc_success_us,
            alloc_beam_indices_us,
            token_ids_to_cuda_us,
            alloc_token_ids_out_us,
            alloc_cum_log_probs_us,
            greedy_buffer_get_us,
            greedy_exec_us,
            beam_exec_us,
            postprocess_us,
            inputs.batch_size,
            inputs.batch_size_out,
            inputs.vocab_size,
            inputs.step,
            greedy_group_count,
            beam_group_count,
            has_logits_processor,
            has_num_beams,
            variable_num_beams,
            inputs.return_original_all_probs,
            inputs.all_probs.defined(),
            inputs.do_sample.defined(),
            inputs.top_k.defined(),
            inputs.top_p.defined(),
            inputs.temperature.defined(),
            simple_greedy_fast_path,
            candidate_greedy_fast_path,
            candidate_num,
            inputs.candidate_tokens_same_batch,
            candidate_gather_us,
            candidate_logits_cpu_us,
            candidate_cpu_select_us,
            candidate_argmax_us,
            candidate_map_us);

        return SamplerOutput({std::move(new_token_ids),
                              torch::Tensor(),
                              torch::Tensor(),
                              torch::Tensor(),
                              std::move(all_success),
                              true});
    }

    int64_t prepare_start_us = autil::TimeUtility::currentTimeInMicroSeconds();

    // allocate output tensors
    // Keep success on CUDA to avoid a blocking D2H copy: the GPU sampling kernel writes success
    // directly, and callers that need CPU access should call .cpu() explicitly.
    stage_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
    auto all_success =
        torch::empty({(int64_t)inputs.batch_size}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    alloc_success_us = autil::TimeUtility::currentTimeInMicroSeconds() - stage_start_us;

    stage_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
    auto all_beam_indices =
        has_num_beams ? torch::empty({(int64_t)inputs.batch_size_out}, torch::kInt32) : torch::Tensor();
    alloc_beam_indices_us = autil::TimeUtility::currentTimeInMicroSeconds() - stage_start_us;

    // Move token_ids to CUDA once so sampleGreedy writes GPU→GPU (no blocking D2H sync).
    // Callers that need CPU access should call .cpu() explicitly.
    // Use blocking transfer: on ROCm, hipMemcpyAsync from pageable memory is truly async
    // and can cause memory access faults if a kernel reads the buffer before transfer completes.
    stage_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
    auto inputs_token_ids_cuda = inputs.token_ids.to(torch::kCUDA);
    token_ids_to_cuda_us = autil::TimeUtility::currentTimeInMicroSeconds() - stage_start_us;

    stage_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
    auto all_token_ids_out = variable_num_beams ?
                                     torch::empty({(int64_t)inputs.batch_size_out, (int64_t)max_seq_len},
                                              torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)) :
                                     inputs_token_ids_cuda;
    alloc_token_ids_out_us = autil::TimeUtility::currentTimeInMicroSeconds() - stage_start_us;

    stage_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
    auto all_cum_log_probs_out = variable_num_beams && inputs.cum_log_probs.defined() ?
                                     torch::empty({(int64_t)inputs.batch_size_out}, torch::kFloat32) :
                                     inputs.cum_log_probs;
    alloc_cum_log_probs_us = autil::TimeUtility::currentTimeInMicroSeconds() - stage_start_us;

    size_t from_batch_idx_in = 0, to_batch_idx_in = 0;
    size_t from_batch_idx_out      = 0;
    stage_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
    auto&  greedy_sampling_buffers = nextGreedySamplingBuffers(inputs.batch_size);
    greedy_buffer_get_us = autil::TimeUtility::currentTimeInMicroSeconds() - stage_start_us;
    struct GreedySamplingBufferGuard {
        Sampler* sampler = nullptr;
        ~GreedySamplingBufferGuard() {
            if (sampler == nullptr) {
                return;
            }
            try {
                sampler->markGreedySamplingBufferReady();
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("failed to record greedy sampling buffer event: %s", e.what());
            } catch (...) {
                RTP_LLM_LOG_WARNING("failed to record greedy sampling buffer event");
            }
        }
    } greedy_sampling_buffer_guard{this};
    prepare_output_us = autil::TimeUtility::currentTimeInMicroSeconds() - prepare_start_us;

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

            GreedySamplingBuffers greedy_sampling_buffer_slice{
                greedy_sampling_buffers.seed_host.narrow(0, from_batch_idx_in, batch_size_in),
                greedy_sampling_buffers.offset_host.narrow(0, from_batch_idx_in, batch_size_in),
                greedy_sampling_buffers.output_ids_ptrs_host.narrow(0, from_batch_idx_in, batch_size_in),
                batch_size_in};
            GreedySamplingBuffers* greedy_sampling_buffer_ptr = &greedy_sampling_buffer_slice;

            RTP_LLM_PROFILE_SCOPE("sampler.forward.execSampleGreedy");
            int64_t sample_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
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
                 inputs.return_original_all_probs,
                 all_probs,
                 presence_penalty,
                 frequency_penalty,
                 do_sample,
                 generator,
                 greedy_sampling_buffer_ptr});
            greedy_exec_us += autil::TimeUtility::currentTimeInMicroSeconds() - sample_start_us;
            ++greedy_group_count;
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
            RTP_LLM_LOG_DEBUG("current_num_beams_in is %d", cur_num_beams_in);
            RTP_LLM_LOG_DEBUG("current_num_beams_out is %d", cur_num_beams_out);
            RTP_LLM_LOG_DEBUG("current_beam_batch is %d", beam_batch_size);
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

            int64_t sample_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
            auto output = execSampleBeamSearch({logits_t,
                                                token_ids_in_t,
                                                input_lengths_t,
                                                sequence_lengths_t,
                                                cum_log_probs_in_t,
                                                (size_t)cur_num_beams_out});
            beam_exec_us += autil::TimeUtility::currentTimeInMicroSeconds() - sample_start_us;
            ++beam_group_count;

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

            success.fill_(true);
        }

        int64_t post_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
        // prepare for next sampling
        from_batch_idx_in  = to_batch_idx_in;
        from_batch_idx_out = to_batch_idx_out;
        postprocess_us += autil::TimeUtility::currentTimeInMicroSeconds() - post_start_us;
    }

    int64_t total_us = autil::TimeUtility::currentTimeInMicroSeconds() - forward_start_us;
    RTP_LLM_LOG_INFO(
        "[PERF] sampler_forward_detail: total_us=%ld, preprocess_logits_us=%ld, prepare_output_us=%ld, "
        "alloc_success_us=%ld, alloc_beam_indices_us=%ld, token_ids_to_cuda_us=%ld, "
        "alloc_token_ids_out_us=%ld, alloc_cum_log_probs_us=%ld, greedy_buffer_get_us=%ld, "
        "greedy_exec_us=%ld, beam_exec_us=%ld, postprocess_us=%ld, batch_size=%zu, batch_size_out=%zu, "
        "vocab_size=%zu, step=%zu, greedy_groups=%zu, beam_groups=%zu, has_logits_processor=%d, "
        "has_num_beams=%d, variable_num_beams=%d, return_original_all_probs=%d, all_probs_defined=%d, "
        "do_sample_defined=%d, top_k_defined=%d, top_p_defined=%d, temperature_defined=%d, "
        "simple_greedy_fast_path=%d, candidate_sampling=%d, candidate_num=%zu, candidate_same_batch=%d, "
        "candidate_gather_us=%ld, candidate_logits_cpu_us=%ld, candidate_cpu_select_us=%ld, "
        "candidate_argmax_us=%ld, candidate_map_us=%ld",
        total_us,
        preprocess_logits_us,
        prepare_output_us,
        alloc_success_us,
        alloc_beam_indices_us,
        token_ids_to_cuda_us,
        alloc_token_ids_out_us,
        alloc_cum_log_probs_us,
        greedy_buffer_get_us,
        greedy_exec_us,
        beam_exec_us,
        postprocess_us,
        inputs.batch_size,
        inputs.batch_size_out,
        inputs.vocab_size,
        inputs.step,
        greedy_group_count,
        beam_group_count,
        has_logits_processor,
        has_num_beams,
        variable_num_beams,
        inputs.return_original_all_probs,
        inputs.all_probs.defined(),
        inputs.do_sample.defined(),
        inputs.top_k.defined(),
        inputs.top_p.defined(),
        inputs.temperature.defined(),
        simple_greedy_fast_path,
        false,
        has_candidate_token_ids ? inputs.candidate_token_ids.numel() : 0,
        inputs.candidate_tokens_same_batch,
        candidate_gather_us,
        candidate_logits_cpu_us,
        candidate_cpu_select_us,
        candidate_argmax_us,
        candidate_map_us);

    return SamplerOutput({std::move(all_token_ids_out),
                          std::move(all_cum_log_probs_out),
                          std::move(inputs.all_probs),
                          std::move(all_beam_indices),
                          std::move(all_success),
                          false});
}

void Sampler::preprocessLogits(const SamplerInputs& inputs) {
    if (inputs.logits_processor_states_ptr != nullptr && !inputs.logits_processor_states_ptr->empty()) {
        inputs.logits_processor_states_ptr->batchProcess(inputs);
    }
}

}  // namespace rtp_llm
