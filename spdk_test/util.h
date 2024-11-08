#pragma once
#include <rte_hash.h>
#include <spdk/log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <threads.h>

#define CORE_NO 49
#define THREAD_NUM 8
#define CPU_BUFFER_SIZE (4UL << 30)

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

inline  unsigned long mhash(std::string &tesnor_name){
    unsigned long hash = 5381;
    for (int i = 0; i < tesnor_name.size(); i++) {
        hash = ((hash << 5) + hash) + tesnor_name[i];
    }
    return hash;
}
