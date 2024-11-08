#pragma once
#include <c10/cuda/CUDAStream.h>
#include <rte_hash.h>
#include <spdk/log.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <string>

#define CORE_NO 0
#define THREAD_NUM 4
#define CPU_BUFFER_SIZE (1UL << 30)

constexpr size_t  block_size = (2048llu * 1024);
constexpr size_t  sector_size = 4096;

extern thread_local int thread_idx;
extern int nvme_cnt;

unsigned long inline rdtscll(void)
{
    unsigned long a, d;
    __asm__ __volatile__("rdtsc" : "=a"(a), "=d"(d));
    return a | ((unsigned long)d << 32);
}
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define round_up_divide(a, b) (((a) + (b)-1) / (b))
#define round_down_divide(a, b) ((a) / (b))
#define round_up(a, b) (round_up_divide(a, b) * (b))
#define round_down(a, b) (round_down_divide(a, b) * (b))
#if DEBUG
#define DEBUG_PRINT(...) SPDK_ERRLOG(__VA_ARGS__)
#define INFO_PRINT(...) SPDK_PRINTF(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#define INFO_PRINT(...) printf(__VA_ARGS__)
#endif
extern struct rte_hash* ptr_map;

static inline void* ptr_to_cpu_ptr(void* ptr)
{
    // if the tensor is on cpu, return the data ptr directly
    void* cpu_ptr;
    int r = rte_hash_lookup_data(ptr_map, &ptr, &cpu_ptr);
    if (r < 0) { return ptr; }
    return cpu_ptr;
}

static inline void* tensor_to_cpu_ptr(at::Tensor& tensor)
{
    // if the tensor is on cpu, return the data ptr directly
    if (tensor.is_cpu()) { return tensor.data_ptr(); }
    void* device_ptr = tensor.data_ptr();
    void* cpu_ptr;
    int r = rte_hash_lookup_data(ptr_map, &device_ptr, &cpu_ptr);
    if (r < 0) {
        printf("the dev_ptr is not registered!Error code %d\n", r);
        exit(-1);
    }
    return cpu_ptr;
}
inline  unsigned long mhash(std::string &tesnor_name){
    unsigned long hash = 5381;
    for (int i = 0; i < tesnor_name.size(); i++) {
        hash = ((hash << 5) + hash) + tesnor_name[i];
    }
    return hash;
}
