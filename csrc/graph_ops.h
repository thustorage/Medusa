#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAGraph.h>

void save_cuda_graph(
  at::cuda::CUDAGraph& graph
);