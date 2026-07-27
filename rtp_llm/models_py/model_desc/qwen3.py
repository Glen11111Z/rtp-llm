import logging
import time
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    RMSNorm,
)
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        attn_configs = config.getAttentionConfigs(parallelism_config.get_attn_tp_size())
        self.self_attn = CausalAttention(
            attn_configs,
            parallelism_config,
            weights,
            config.layernorm_eps,
            quant_config,
            hw_kernel_config,
            layer_idx,
        )
        self.mlp = DenseMLP(
            config.activation_type,
            parallelism_config,
            weights,
            quant_config,
            hw_kernel_config,
        )
        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
        perf: Optional[Dict[str, int]] = None,
    ) -> torch.Tensor:
        stage_start_ns = time.perf_counter_ns()
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if perf is not None:
            perf["input_norm_us"] += (time.perf_counter_ns() - stage_start_ns) // 1000

        # Self Attention
        stage_start_ns = time.perf_counter_ns()
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
        )
        if perf is not None:
            perf["attention_us"] += (time.perf_counter_ns() - stage_start_ns) // 1000

        stage_start_ns = time.perf_counter_ns()
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if perf is not None:
            perf["post_attention_norm_us"] += (time.perf_counter_ns() - stage_start_ns) // 1000

        # Fully Connected
        stage_start_ns = time.perf_counter_ns()
        hidden_states = self.mlp(hidden_states)
        if perf is not None:
            perf["ffn_us"] += (time.perf_counter_ns() - stage_start_ns) // 1000

        stage_start_ns = time.perf_counter_ns()
        hidden_states = residual + hidden_states
        if perf is not None:
            perf["residual_us"] += (time.perf_counter_ns() - stage_start_ns) // 1000

        return hidden_states


class Qwen3Model(GptModelBase):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        quant_config: Optional[object] = None,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )

        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config,
                    parallelism_config,
                    idx,
                    weights.weights[idx],
                    quant_config,
                    py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        forward_start_ns = time.perf_counter_ns()
        perf = {
            "embedding_us": 0,
            "prepare_fmha_us": 0,
            "block_map_us": 0,
            "input_norm_us": 0,
            "attention_us": 0,
            "post_attention_norm_us": 0,
            "ffn_us": 0,
            "residual_us": 0,
            "final_norm_us": 0,
        }
        input_ids: torch.Tensor = inputs.input_ids
        stage_start_ns = time.perf_counter_ns()
        inputs_embeds = self.embed_tokens(input_ids)
        perf["embedding_us"] = (time.perf_counter_ns() - stage_start_ns) // 1000
        hidden_states = inputs_embeds
        if fmha_impl is None:
            stage_start_ns = time.perf_counter_ns()
            fmha_impl = self.prepare_fmha_impl(inputs)
            perf["prepare_fmha_us"] = (time.perf_counter_ns() - stage_start_ns) // 1000
        layer_loop_start_ns = time.perf_counter_ns()
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            stage_start_ns = time.perf_counter_ns()
            select_block_map_for_layer(inputs.attention_inputs, i)
            perf["block_map_us"] += (time.perf_counter_ns() - stage_start_ns) // 1000
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                perf=perf,
            )
        layer_loop_us = (time.perf_counter_ns() - layer_loop_start_ns) // 1000
        stage_start_ns = time.perf_counter_ns()
        hidden_states = self.norm(hidden_states)
        perf["final_norm_us"] = (time.perf_counter_ns() - stage_start_ns) // 1000
        total_us = (time.perf_counter_ns() - forward_start_ns) // 1000
        attn_inputs = inputs.attention_inputs
        ctx_batch = int(attn_inputs.prefix_lengths.numel()) if attn_inputs.prefix_lengths is not None else 0
        decode_batch = int(attn_inputs.sequence_lengths.numel()) if attn_inputs.sequence_lengths is not None else 0
        execute_tokens = int(input_ids.numel())
        max_seq_len = int(input_ids.numel())
        try:
            if attn_inputs.input_lengths is not None and attn_inputs.input_lengths.numel() > 0:
                max_seq_len = int(attn_inputs.input_lengths.max().item())
        except Exception:
            pass
        logging.info(
            "[PERF] py_model_forward_detail: "
            "total_us=%d, embedding_us=%d, prepare_fmha_us=%d, layer_loop_us=%d, "
            "block_map_us=%d, input_norm_us=%d, attention_us=%d, "
            "post_attention_norm_us=%d, ffn_us=%d, residual_us=%d, final_norm_us=%d, "
            "ctx_batch=%d, decode_batch=%d, execute_tokens=%d, max_seq_len=%d, layer_num=%d",
            total_us,
            perf["embedding_us"],
            perf["prepare_fmha_us"],
            layer_loop_us,
            perf["block_map_us"],
            perf["input_norm_us"],
            perf["attention_us"],
            perf["post_attention_norm_us"],
            perf["ffn_us"],
            perf["residual_us"],
            perf["final_norm_us"],
            ctx_batch,
            decode_batch,
            execute_tokens,
            max_seq_len,
            self.layer_num,
        )
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = [
    "Qwen3Model",
]
