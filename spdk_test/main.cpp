#include "spdk_util.h"
#include <pthread.h>
#include <cuda_runtime.h>
#include <iostream>
#include "spdk/env.h"
#include <chrono>
#include "util.h"
pthread_barrier_t barrier;
void* host_buffer[THREAD_NUM];
void* host_buffer1;
void* device_ptr;
void* client(void* arg){

    thread_idx = *(int*)arg;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(56+thread_idx, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
    pthread_barrier_wait(&barrier);
    while(true){
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        spdk_io(0, host_buffer[thread_idx], 0, 1024*1024*1024, READ);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::cout<<"IO bandwidth: "<<1024*1024*1024/time_span.count()/1024/1024<<"MB/s\n";
    }
    return NULL;
}
int main(){
    pthread_barrier_init(&barrier, NULL, THREAD_NUM+1);
    spdk_init(0);
    // bind self to core 57
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(55, &set);
    cudaSetDevice(0);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
    
    cudaMallocHost(&host_buffer1, 1024*1024*1024);
    cudaMalloc(&device_ptr, 1024*1024*1024);
    for(int i = 0; i < THREAD_NUM; i++){
        host_buffer[i] = spdk_dma_malloc(1024*1024*1024, 64*1024, 0);
        pthread_t tid;
        int *arg = (int*)malloc(sizeof(int));
        *arg = i;
        pthread_create(&tid, NULL, client, arg);
    }
    pthread_barrier_wait(&barrier);
    int cnt = 0;
    while(1){
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(device_ptr, host_buffer1, 1024*1024*1024, cudaMemcpyHostToDevice);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::cout<<"PCIe bandwidth: "<<1024*1024*1024/time_span.count()/1024/1024<<"MB/s\n";
        cnt++;
    }
    return 0;
}