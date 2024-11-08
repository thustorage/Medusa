#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAGraph.h>

#include <string>

void save_tensor_prepare(const torch::Tensor &tensor, std::string filename);

void save_tensor_start ();

void load_tensor (
  torch::Tensor tensor,
  std::string filename
);

void load_tensor_async_prepare (
  torch::Tensor tensor,
  std::string filename
);

void load_tensor_async_start ();

void load_tensor_sync_all ();

void init_load_thread ();

void set_model_name (std::string model_name);

void init_spdk_daemon();

void fini_spdk_daemon();