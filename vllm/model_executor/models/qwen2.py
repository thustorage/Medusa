# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput
import json
from vllm._C import tensor_ops
import numpy as np
import contextlib
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils import custom_all_reduce

KVCache = List[torch.Tensor]


class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           linear_method=linear_method)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 use_sliding_window: bool = False,
                 linear_method: Optional[LinearMethodBase] = None,
                 sliding_window: Optional[int] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window if use_sliding_window else None

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
        )
        self.attn = PagedAttention(self.num_heads,
                                   self.head_dim,
                                   self.scaling,
                                   num_kv_heads=self.num_kv_heads,
                                   sliding_window=self.sliding_window)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: int,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        use_sliding_window = config.use_sliding_window and layer_idx < config.max_window_layers
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            use_sliding_window=use_sliding_window,
            linear_method=linear_method,
            sliding_window=config.sliding_window)
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            linear_method=linear_method,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(config, layer_idx, linear_method)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        early_return: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                residual,
            )
            if early_return:
                # turn off memory free inside the model before forward finish to avoid free GPU memory after end_capture
                print("early return......", flush=True)
                torch.cuda.memory_shutdown_free()
                return hidden_states
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

@contextlib.contextmanager
def _maybe_cupy_nccl():
    if cupy_utils.is_initialized() and not custom_all_reduce.is_initialized():
        with with_cupy_nccl_for_all_reduce():
            yield
    else:
        yield

class Qwen2ForCausalLM(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        linear_method: Optional[LinearMethodBase] = None,
        fast_start: bool = False,
        persist_cudagraph: bool = False,
        model_name: str = "",
        _BATCH_SIZES_TO_CAPTURE: List[int] = []
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = Qwen2Model(config, linear_method)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.sampler = Sampler(config.vocab_size)
        
        self.fast_start = fast_start
        self.persist_cudagraph = persist_cudagraph
        self.model_name = model_name
        self._BATCH_SIZES_TO_CAPTURE = _BATCH_SIZES_TO_CAPTURE
        

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        early_return: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata, early_return = early_return)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None,
                     async_load: bool = False,
                     save_tensor: bool = False,
                     gpu_cache: List[KVCache] = []):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        if save_tensor:
            params_dict = dict(self.named_parameters())
            shape_dict = {}
            for name, loaded_weight in hf_model_weights_iterator(
                    model_name_or_path, cache_dir, load_format, revision):
                if "rotary_emb.inv_freq" in name:
                    continue
                for (param_name, weight_name, shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    shape_dict[name] = param.shape
                    tensor_ops.save_tensor_prepare(param.cpu(), f"/home/zsx/raidfs-back/tensors/{self.model_name}/{name}")
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                    shape_dict[name] = param.shape
                    tensor_ops.save_tensor_prepare(param.cpu(), f"/home/zsx/raidfs-back/tensors/{self.model_name}/{name}")
                    
            tensor_ops.save_tensor_start()
            json_data = json.dumps(shape_dict)
            with open(f"/home/zsx/raidfs-back/tensors/{self.model_name}/shape_dict.json", "w") as f:
                f.write(json_data)
        else:
            # if not async_load:
            #     params_dict = dict(self.named_parameters())
            #     for name, loaded_weight in hf_model_weights_iterator(
            #             model_name_or_path, cache_dir, load_format, revision):
            #         if "rotary_emb.inv_freq" in name:
            #             continue
            #         for (param_name, weight_name, shard_id) in stacked_params_mapping:
            #             if weight_name not in name:
            #                 continue
            #             name = name.replace(weight_name, param_name)
            #             # Skip loading extra bias for GPTQ models.
            #             if name.endswith(".bias") and name not in params_dict:
            #                 continue
            #             param = params_dict[name]
            #             weight_loader = param.weight_loader
            #             weight_loader(param, loaded_weight, shard_id)
            #             break
            #         else:
            #             # Skip loading extra bias for GPTQ models.
            #             if name.endswith(".bias") and name not in params_dict:
            #                 continue
            #             param = params_dict[name]
            #             weight_loader = getattr(param, "weight_loader",
            #                                     default_weight_loader)
            #             weight_loader(param, loaded_weight)
        # ====================== load tensors ======================
            params_dict = dict(self.named_parameters())
            shape_dict = {}
            with open(f"/home/zsx/raidfs-back/tensors/{self.model_name}/shape_dict.json", "r") as file:
                shape_dict = json.load(file)
            
            for name, loaded_weight in shape_dict.items():
                if "rotary_emb.inv_freq" in name:
                    continue
                for (param_name, weight_name, shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    tensor_ops.load_tensor_async_prepare(param, f"/home/zsx/raidfs-back/tensors/{self.model_name}/{name}")
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    tensor_ops.load_tensor_async_prepare(param, f"/home/zsx/raidfs-back/tensors/{self.model_name}/{name}")
                    
            tensor_ops.load_tensor_async_start()
            if not async_load:
                tensor_ops.load_tensor_sync_all()
                
        
        if self.persist_cudagraph:
            # warmup here, tigger async module load
            _PAD_SLOT_ID = -1
            max_batch_size = max(self._BATCH_SIZES_TO_CAPTURE)
            input_tokens = torch.zeros(max_batch_size, 1, dtype=torch.long).cuda()
            input_positions = torch.zeros(max_batch_size, 1,
                                        dtype=torch.long).cuda()
            slot_mapping = torch.empty(max_batch_size, 1, dtype=torch.long).cuda()
            slot_mapping.fill_(_PAD_SLOT_ID)
            context_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
            graph_block_tables = np.zeros(
                (max(self._BATCH_SIZES_TO_CAPTURE), 256), dtype=np.int32)
            block_tables = torch.from_numpy(graph_block_tables).cuda()
 
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                for bs in self._BATCH_SIZES_TO_CAPTURE:
                    input_metadata = InputMetadata(
                        is_prompt=False,
                        slot_mapping=slot_mapping[:bs],
                        prompt_lens=None,
                        max_seq_len=None,
                        start_loc=None,
                        max_context_len=1024,
                        context_lens=context_lens[:bs],
                        block_tables=block_tables[:bs],
                        use_cuda_graph=True,
                        kv_cache_dtype="auto",
                    )
                    with _maybe_cupy_nccl():
                        self.forward(
                            input_tokens[:bs],
                            input_positions[:bs],
                            gpu_cache,
                            input_metadata,
                            early_return = True,
                        )
                    # pair with early_return = True
                    torch.cuda.memory_turnon_free()
                # not synchronize, let lazy module load async
                # torch.cuda.synchronize()