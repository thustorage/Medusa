#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include "graph_ops.h"
#include "tensor_ops.h"
#include "test_spdk.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Cuda graph ops
  pybind11::module graph_ops = m.def_submodule("graph_ops", "Cuda graph operators");
  graph_ops.def("save_cuda_graph", &save_cuda_graph, "Save cuda graph.");

  pybind11::module tensor_ops = m.def_submodule("tensor_ops", "save/load tensor to/from file");
  tensor_ops.def("save_tensor_prepare", &save_tensor_prepare, "Save tensor to file.");
  tensor_ops.def("save_tensor_start", &save_tensor_start, "Save tensor to file start.");
  tensor_ops.def("load_tensor", &load_tensor, "Load tensor from file.");
  tensor_ops.def("load_tensor_async_prepare", &load_tensor_async_prepare, "Load tensor from file async.");
  tensor_ops.def("load_tensor_async_start", &load_tensor_async_start, "Load tensor from file async.");
  tensor_ops.def("load_tensor_sync_all", &load_tensor_sync_all, "Load tensor from file sync all.");
  tensor_ops.def("init_load_thread", &init_load_thread, "Async load process.");
  tensor_ops.def("set_model_name", &set_model_name, "Set model name.");
  tensor_ops.def("init_spdk_daemon", &init_spdk_daemon, "Init spdk daemon.");
  tensor_ops.def("fini_spdk_daemon", &fini_spdk_daemon, "Fini spdk daemon.");

  pybind11::module test_spdk = m.def_submodule("test_spdk", "test spdk");
  test_spdk.def("test", &test, "test.");

  // vLLM custom ops
  pybind11::module ops = m.def_submodule("ops", "vLLM custom operators");

  // Attention ops
  ops.def(
    "paged_attention_v1",
    &paged_attention_v1,
    "Compute the attention between an input query and the cached keys/values using PagedAttention.");
  ops.def(
    "paged_attention_v2",
    &paged_attention_v2,
    "PagedAttention V2.");

  // Activation ops
  ops.def(
    "silu_and_mul",
    &silu_and_mul,
    "Activation function used in SwiGLU.");
  ops.def(
    "gelu_new",
    &gelu_new,
    "GELU implementation used in GPT-2.");
  ops.def(
    "gelu_fast",
    &gelu_fast,
    "Approximate GELU implementation.");

  // Layernorm
  ops.def(
    "rms_norm",
    &rms_norm,
    "Apply Root Mean Square (RMS) Normalization to the input tensor.");

  ops.def(
    "fused_add_rms_norm",
    &fused_add_rms_norm,
    "In-place fused Add and RMS Normalization");

  // Rotary embedding
  ops.def(
    "rotary_embedding",
    &rotary_embedding,
    "Apply GPT-NeoX or GPT-J style rotary embedding to query and key");

  // Quantization ops
#ifndef USE_ROCM
  ops.def("awq_gemm", &awq_gemm, "Quantized GEMM for AWQ");
  ops.def("awq_dequantize", &awq_dequantize, "Dequantization for AWQ");
#endif
  ops.def("gptq_gemm", &gptq_gemm, "Quantized GEMM for GPTQ");
  ops.def("gptq_shuffle", &gptq_shuffle, "Post processing for GPTQ");
  ops.def("squeezellm_gemm", &squeezellm_gemm, "Quantized GEMM for SqueezeLLM");
  ops.def(
    "moe_align_block_size",
    &moe_align_block_size,
    "Aligning the number of tokens to be processed by each expert such that it is divisible by the block size.");

  // Cache ops
  pybind11::module cache_ops = m.def_submodule("cache_ops", "vLLM cache ops");
  cache_ops.def(
    "swap_blocks",
    &swap_blocks,
    "Swap in (out) the cache blocks from src to dst");
  cache_ops.def(
    "copy_blocks",
    &copy_blocks,
    "Copy the cache blocks from src to dst");
  cache_ops.def(
    "reshape_and_cache",
    &reshape_and_cache,
    "Reshape the key and value tensors and cache them");
  cache_ops.def(
    "gather_cached_kv",
    &gather_cached_kv,
    "Gather key and value from the cache into contiguous QKV tensors");
  cache_ops.def(
    "convert_fp8_e5m2",
    &convert_fp8_e5m2,
    "Convert the key and value cache to fp8_e5m2 data type");

  // Cuda utils
  pybind11::module cuda_utils = m.def_submodule("cuda_utils", "vLLM cuda utils");
  cuda_utils.def(
    "get_device_attribute",
    &get_device_attribute,
    "Gets the specified device attribute.");

  cuda_utils.def(
    "get_max_shared_memory_per_block_device_attribute",
    &get_max_shared_memory_per_block_device_attribute,
    "Gets the maximum shared memory per block device attribute.");

#ifndef USE_ROCM
  // Custom all-reduce kernels
  pybind11::module custom_ar = m.def_submodule("custom_ar", "custom allreduce");
  custom_ar.def("init_custom_ar", &init_custom_ar, "init_custom_ar");
  custom_ar.def("should_custom_ar", &should_custom_ar, "should_custom_ar");
  custom_ar.def("all_reduce_reg", &all_reduce_reg, "all_reduce_reg");
  custom_ar.def("all_reduce_unreg", &all_reduce_unreg, "all_reduce_unreg");
  custom_ar.def("dispose", &dispose, "dispose");
  custom_ar.def("meta_size", &meta_size, "meta_size");
  custom_ar.def("register_buffer", &register_buffer, "register_buffer");
  custom_ar.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta,
                "get_graph_buffer_ipc_meta");
  custom_ar.def("register_graph_buffers", &register_graph_buffers,
                "register_graph_buffers");
#endif

}
