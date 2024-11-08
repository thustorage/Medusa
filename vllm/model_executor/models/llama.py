# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
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
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, List, Optional, Tuple
import contextlib

import torch
from torch import nn
from transformers import LlamaConfig

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
    VocabParallelEmbedding, ParallelLMHead, DEFAULT_VOCAB_PADDING_SIZE)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput
from vllm.config import LoRAConfig

from vllm._C import tensor_ops

import json

import numpy as np

from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils import custom_all_reduce

KVCache = List[torch.Tensor]

class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
        fast_start: bool = False,
    ) -> None:
        super().__init__()
        self.fast_start = fast_start
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


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        linear_method: Optional[LinearMethodBase] = None,
        bias: bool = False,
        fast_start: bool = False,
    ) -> None:
        super().__init__()
        self.fast_start = fast_start
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
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            linear_method=linear_method
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            linear_method=linear_method,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = PagedAttention(self.num_heads,
                                   self.head_dim,
                                   self.scaling,
                                   num_kv_heads=self.num_kv_heads)

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
    
def print_matmul_dims(module, inputs, output):
    if isinstance(module, torch.nn.modules.linear.Linear):
        input_data = inputs[0]
        weight = module.weight
        output_data = output
        print(f"Matrix multiplication dimensions: {input_data.shape} x {weight.shape} -> {output_data.shape}")

def register_hook(module):
    print(module)
    if isinstance(module, torch.nn.modules.linear.Linear):
        module.register_forward_hook(print_matmul_dims)
    if isinstance(module, torch.nn.Module):
        for child_module in module.children():
            register_hook(child_module)

class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
        fast_start: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            linear_method=linear_method,
            bias=getattr(config, "bias", False),
            fast_start = fast_start,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            linear_method=linear_method,
            fast_start = fast_start
        )
        # register_hook(self.mlp)
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


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
        fast_start: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, linear_method, fast_start = fast_start)
            for _ in range(config.num_hidden_layers)
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
            # if change this, change result_hidden_states_addr as well
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

class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
        fast_start: bool = False,
        persist_cudagraph: bool = False,
        model_name: str = "",
        _BATCH_SIZES_TO_CAPTURE: List[int] = []
    ) -> None:
        super().__init__()
        self.fast_start = fast_start
        self.persist_cudagraph = persist_cudagraph
        self.config = config
        self.linear_method = linear_method
        self.model = LlamaModel(config, linear_method, lora_config=lora_config, fast_start=fast_start)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size
        )
        self.sampler = Sampler(self.unpadded_vocab_size, config.vocab_size)
        
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
        # =================== save tensors ===================
        if save_tensor:
            params_dict = dict(self.named_parameters())
            shape_dict = {}
            
            for name, loaded_weight in hf_model_weights_iterator(
                    model_name_or_path, cache_dir, load_format, revision, fast_start = self.fast_start):
                if "rotary_emb.inv_freq" in name:
                    continue
                if ("rotary_emb.cos_cached" in name
                        or "rotary_emb.sin_cached" in name):
                    # Models trained using ColossalAI may include these tensors in
                    # the checkpoint. Skip them.
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
            #             model_name_or_path, cache_dir, load_format, revision, fast_start = self.fast_start):
            #         if "rotary_emb.inv_freq" in name:
            #             continue
            #         if ("rotary_emb.cos_cached" in name
            #                 or "rotary_emb.sin_cached" in name):
            #             # Models trained using ColossalAI may include these tensors in
            #             # the checkpoint. Skip them.
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
            
            for name, loaded_weight_shape in shape_dict.items():
                if "rotary_emb.inv_freq" in name:
                    continue
                if ("rotary_emb.cos_cached" in name
                        or "rotary_emb.sin_cached" in name):
                    # Models trained using ColossalAI may include these tensors in
                    # the checkpoint. Skip them.
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