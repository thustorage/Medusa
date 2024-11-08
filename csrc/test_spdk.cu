#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <torch/extension.h>

#include <cuda.h>

#include "spdk/env.h"
#include "spdk_util.h"
#include "tensor_ops.h"
#include "util.h"
#include <atomic>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <vector>

void fill_write_buffer(void* buffer) {
  // open file and read to write_buffer
  std::ifstream file("/home/zsx/raidfs-back/tensors/Llama-13B/model.layers.4.mlp.down_proj.weight", std::ios::binary);
  file.read((char*)buffer, CPU_BUFFER_SIZE);
  file.close();
}

int test() {
  void* buffer_to_write = spdk_dma_malloc_socket(CPU_BUFFER_SIZE, 2UL << 20, 0, 0);
  void* buffer_to_read = spdk_dma_malloc_socket(CPU_BUFFER_SIZE, 2UL << 20, 0, 0);

  fill_write_buffer(buffer_to_write);
  memset(buffer_to_read, 0, CPU_BUFFER_SIZE);

  size_t test_size = CPU_BUFFER_SIZE;
  spdk_io(0, buffer_to_write, 65357742080, test_size, WRITE);
  spdk_io(0, buffer_to_read, 65357742080, test_size, READ);

  for (size_t i = 0; i < test_size; i++) {
    if (((char*)buffer_to_read)[i] != ((char*)buffer_to_write)[i]) {
      printf("Error at %ld\n", i);
      printf("Expected: %ld\n", ((char*)buffer_to_write)[i]);
      printf("Got: %ld\n", ((char*)buffer_to_read)[i]);
      exit(-1);
      break;
    }
  }

  printf("Test passed\n");

  spdk_dma_free(buffer_to_write);
  spdk_dma_free(buffer_to_read);

  return 0;
}