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
#include <chrono>

#define NUM_DISKS 4
thread_local int thread_idx;
size_t offset_max;
static int first_save = 0;
static void *host_buffer;
std::string model_name;

std::map<std::string, torch::Tensor> tensor_map;

void save_tensor_prepare(const torch::Tensor &tensor, std::string filename) {
  // some qkv may be saved multiple times
  // the last one is the correct one
  // use the map to overwrite the previous ones
  tensor_map[filename] = tensor;
}

void save_tensor_start() {
  for (auto &it : tensor_map) {
    const torch::Tensor &tensor = it.second;
    std::string filename = it.first;

    if (first_save == 0) {
      std::string offset_max_file_name = std::string("/home/zsx/vllm/model_offsets/") + model_name;
      printf("offset_max_file_name %s\n", offset_max_file_name.c_str());
      std::ifstream file(offset_max_file_name, std::ios::binary);
      if (!file.is_open()) {
        std::cerr << "Failed to open file " << offset_max_file_name << std::endl;
        exit(-1);
      }
      file >> offset_max;
      file.close();
      printf("init offset_max %lu\n", offset_max);
      spdk_init(NUM_DISKS);
      host_buffer = spdk_dma_malloc(3UL << 30, 0, NULL);
      assert(host_buffer != NULL);
      first_save = 1;
    }
    filename += ".locate";
    fflush(stderr);
    fflush(stdout);

    size_t write_size = tensor.numel() * tensor.itemsize();
    if (write_size > 3UL << 30) {
      std::cerr << "Tensor size is too large for SPDK" << std::endl;
      exit(-1);
    }

    memcpy(host_buffer, tensor.data_ptr(), write_size);
    size_t round_up_write_size = round_up(write_size, sector_size);
    std::cerr << "save tensor " << filename << " " << offset_max << " "
              << round_up_write_size << std::endl;
    spdk_io(0, host_buffer, offset_max , round_up_write_size, WRITE);
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Failed to open file" << filename << std::endl;
      fflush(stderr);
      fflush(stdout);
      exit(-1);
    }
    file << offset_max << std::endl;
    file.flush();
    file.close();

    offset_max += round_up_write_size;
    offset_max = round_up(offset_max, block_size);
    // void* data_ptr = tensor.data_ptr();

    // size_t tensor_size_bytes = tensor.numel() * tensor.itemsize();

    // file.write(static_cast<const char*>(data_ptr), tensor_size_bytes);
    printf("offset_max %lu\n", offset_max);

  }

  // read all for debug
  // printf("Read all for debug\n");
  // void *buffer_for_debug = malloc(3UL << 30);
  // for (auto &it : tensor_map) {
  //   const torch::Tensor &tensor = it.second;
  //   std::string filename = it.first;

  //   size_t write_size = tensor.numel() * tensor.itemsize();

  //   memset(host_buffer, 0, round_up(write_size, block_size));

  //   std::string offset_max_file_name = filename + ".locate";
  //   std::ifstream file(offset_max_file_name, std::ios::binary);
  //   if (!file.is_open()) {
  //     std::cerr << "Failed to open file " << offset_max_file_name << std::endl;
  //     exit(-1);
  //   }
  //   file >> offset_max;
  //   file.close();
  //   spdk_io(0, host_buffer, offset_max , round_up(write_size, block_size), READ);

  //   for (size_t j = 0; j < write_size; j++) {
  //     if (static_cast<char *>(tensor.data_ptr())[j] != static_cast<char *>(host_buffer)[j]) {
  //       std::cerr << "Read/Write Data mismatch at " << j << std::endl;
  //       std::cerr << filename << std::endl;
  //       exit(-1);
  //     }
  //   }
  // }
}

void load_tensor(torch::Tensor tensor, std::string filename) {
  printf("load tensor %s\n", filename.c_str());
  std::ifstream file(filename, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "Failed to open file" << std::endl;
    exit(-1);
  }

  void *data_ptr = tensor.data_ptr();

  size_t tensor_size_bytes = tensor.numel() * tensor.itemsize();

  void *cpu_buffer = malloc(tensor_size_bytes);
  file.read(static_cast<char *>(cpu_buffer), tensor_size_bytes);
  file.close();

  cudaMemcpy(data_ptr, cpu_buffer, tensor_size_bytes, cudaMemcpyHostToDevice);

  free(cpu_buffer);
}

std::thread async_load_thread;

std::atomic_bool async_request = false;
std::atomic_bool start_all_threads = false;
std::vector<torch::Tensor> tensors;
std::vector<std::string> filenames;

// change llm_engine.py as well, pytorch bindcore
// and change CUDAGraph as well, pytorch c++ bindcore
// 0 - 8 async load c++
// 9 - 47 for pytorch python

void bind_core(int core_no) {
  cpu_set_t cpu_mask;
  CPU_ZERO(&cpu_mask);
  CPU_SET(core_no, &cpu_mask);

  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpu_mask) != 0) {
    std::cerr << "Failed to set thread affinity mask." << std::endl;
    exit(-1);
  }
}

std::vector<torch::Tensor> all_tensors[THREAD_NUM];
std::vector<std::string> all_filenames[THREAD_NUM];
cudaStream_t all_streams[THREAD_NUM];

void *cpu_buffers[THREAD_NUM];
std::mutex mutex;
int load_tensor(void *thread_no) {
  thread_idx = *(int *)thread_no;
  cudaFree(0);
  bind_core(CORE_NO + 1 + thread_idx);
  cudaStream_t &stream = all_streams[thread_idx];

  cpu_buffers[thread_idx] =
      spdk_dma_malloc_socket(CPU_BUFFER_SIZE, 2UL << 20, 0, 0);
  assert(cpu_buffers[thread_idx] != NULL);
  // maybe unsafe, since may use the same mmap file as the other parts of SPDK
  // std::string mmap_name =
  //     std::string("/dev/hugepages/spdk114514map_") +
  //     std::to_string(thread_idx);
  // int mmap_fd = open(mmap_name.c_str(), O_RDWR, 0600);
  // if (mmap_fd == -1) {
  //   printf("mmap open failed\n");
  //   fflush(stdout);
  //   abort();
  // }

  // cpu_buffers[thread_idx] =
  //     mmap(0, CPU_BUFFER_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mmap_fd,
  //     0);
  // if (!cpu_buffers[thread_idx] || cpu_buffers[thread_idx] == MAP_FAILED) {
  //   printf("mmap failed\n");
  //   fflush(stdout);
  //   abort();
  // }
  // spdk_mem_register(cpu_buffers[thread_idx], CPU_BUFFER_SIZE);
  cudaError_t err = cudaHostRegister(cpu_buffers[thread_idx], CPU_BUFFER_SIZE,
                                     cudaHostRegisterDefault);
  printf("cpu_buffer %p, error %s\n", cpu_buffers[thread_idx],
         cudaGetErrorString(err));
  fflush(stdout);
  assert(err == cudaSuccess);

  void *cpu_buffer = cpu_buffers[thread_idx];

  while (true) {
    if (start_all_threads == true) {
      break;
    } else {
      usleep(100);
    }
  }

  std::vector<torch::Tensor> &tensors = all_tensors[thread_idx];
  std::vector<std::string> &filenames = all_filenames[thread_idx];

  uint64_t offset = 0;
  // compare buffer_for_debug and cpu_buffer for debug
  // void *buffer_for_debug = malloc(3UL << 30);
  for (size_t i = 0; i < tensors.size(); i++) {
    std::string locate_filename = filenames[i] + std::string(".locate");
    std::ifstream file(locate_filename, std::ios::binary);
    size_t file_start_offset;
    if (!file.is_open()) {
      std::cerr << "Failed to open file " << locate_filename << std::endl;
      exit(-1);
    }
    file >> file_start_offset;
    // std::cout << "start_lba " << start_offset << "for name " << filenames[i]
    //           << std::endl;
   
    file.close();

    void *data_ptr = tensors[i].data_ptr();

    size_t total_tensor_size_bytes = tensors[i].numel() * tensors[i].itemsize();

    // compare buffer_for_debug and cpu_buffer for debug
    // std::ifstream file_for_debug(filenames[i], std::ios::binary);
    // file_for_debug.read(static_cast<char *>(buffer_for_debug), total_tensor_size_bytes);
    // printf("load tensor %s\n", filenames[i].c_str());

    size_t buffer_start_offset = 0;
    while (total_tensor_size_bytes > 0) {
      size_t tensor_size_bytes = std::min(total_tensor_size_bytes, CPU_BUFFER_SIZE);
      total_tensor_size_bytes -= tensor_size_bytes;

      if (tensor_size_bytes + round_up(offset,sector_size) > CPU_BUFFER_SIZE) {
        cudaStreamSynchronize(stream);
        offset = 0;
      }
      // pass it by 32MB chunk
      size_t start = 0;
      while(start < tensor_size_bytes) {
        size_t chunk_size = std::min(32lu*1024*1024, tensor_size_bytes - start);
        spdk_io(0, static_cast<char *>(cpu_buffer) + offset + start, file_start_offset + buffer_start_offset + start,
              round_up(chunk_size, sector_size), READ);

        // compare buffer_for_debug and cpu_buffer for debug
        // for (size_t j = 0; j < chunk_size; j++) {
        //   if (static_cast<char *>(cpu_buffer)[offset + start + j] != static_cast<char *>(buffer_for_debug)[buffer_start_offset + start + j]) {
        //     std::cerr << "Data mismatch at " << buffer_start_offset + start + j << std::endl;
        //     std::cerr << filenames[i] << " start offset " << file_start_offset << std::endl;
        //     exit(-1);
        //   }
        // }

        int ret = cudaMemcpyAsync(data_ptr + buffer_start_offset + start, cpu_buffer + offset + start, chunk_size,
                              cudaMemcpyHostToDevice, stream);
        if (ret != 0) {
          std::cerr << "Failed to copy data to GPU" << std::endl;
          exit(-1);
        }
        // cudaStreamSynchronize(cudaStream_t stream)
        start += chunk_size;
      }
      // // mutex.lock();
      // spdk_io(0, static_cast<char *>(cpu_buffer) + 0, start_offset,
      //         round_up(tensor_size_bytes, sector_size), READ);
      // // file.read(static_cast<char *>(cpu_buffer) + offset, tensor_size_bytes);
      // // file.close();
      // int ret = cudaMemcpyAsync(data_ptr, cpu_buffer + 0, tensor_size_bytes,
      //                         cudaMemcpyHostToDevice, stream);
      //   if (ret != 0) {
      //     std::cerr << "Failed to copy data to GPU" << std::endl;
      //     exit(-1);
      //   }
      // cudaStreamSynchronize(stream);
      // file.read(static_cast<char*>(cpu_buffers2[i]), tensor_size_bytes);

      offset += tensor_size_bytes;
      // round offset to 4KB
      offset = round_up(offset, sector_size);

      buffer_start_offset += tensor_size_bytes;
    }
  }
  cudaStreamSynchronize(stream);
  return 0;
}

// #define CU_CALL(x)                                                             \
//   do {                                                                         \
//     CUresult result = x;                                                       \
//     if (result != CUDA_SUCCESS) {                                              \
//       printf("%s:%s:%d CUDA error: %d\n", __FILE__, __func__, __LINE__,        \
//              result);                                                          \
//       exit(-1);                                                                \
//     }                                                                          \
//   } while (0)

void async_load_process() {
  bind_core(CORE_NO);
  spdk_init(NUM_DISKS);

  std::vector<std::thread> threads;
  for (int i = 0; i < THREAD_NUM; i++) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    all_streams[i] = stream;

    int *thread_no = new int(i);
    spdk_env_thread_launch_pinned(CORE_NO + 1 + i, load_tensor, thread_no);
  }

  while (true) {
    if (async_request == true) {
      break;
    } else {
      usleep(100);
    }
  }

  printf("Start loading tensors...\n");
  // calculate duration
  std::chrono::steady_clock::time_point start_time =
      std::chrono::steady_clock::now();

  for (int i = 0; i < THREAD_NUM; i++) {
    size_t start = tensors.size() / THREAD_NUM * i;
    size_t end = tensors.size() / THREAD_NUM * (i + 1);
    if (i == THREAD_NUM - 1) {
      end = tensors.size();
    }

    all_tensors[i] = std::vector<torch::Tensor>(tensors.begin() + start,
                                                tensors.begin() + end);
    all_filenames[i] = std::vector<std::string>(filenames.begin() + start,
                                                filenames.begin() + end);
  }

  start_all_threads = true;

  spdk_env_thread_wait_all();

  std::chrono::steady_clock::time_point end_time =
      std::chrono::steady_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
  printf("Loading tensor time: %f\n", time_span.count());

  tensors.clear();
  filenames.clear();
  async_request = false;
  for (int i = 0; i < THREAD_NUM; i++) {
    cudaStreamSynchronize(all_streams[i]);
    // spdk_free(cpu_buffers[i]);
    // munmap(cpu_buffers[i], CPU_BUFFER_SIZE);
    cudaStreamDestroy(all_streams[i]);
  }
}

void init_load_thread() {
  printf("Initializing async load threads...\n");
  fflush(stdout);
  async_load_thread = std::thread([] { async_load_process(); });
}

void set_model_name(std::string model_name_) { model_name = model_name_; }

void load_tensor_async_prepare(torch::Tensor tensor, std::string filename) {
  tensors.push_back(tensor);
  filenames.push_back(filename);
}

void load_tensor_async_start() { async_request = true; }

void load_tensor_sync_all() {
  printf("Waiting for async load threads to finish...\n");
  fflush(stdout);
  while (true) {
    if (async_load_thread.joinable())
      async_load_thread.join();
    return;
  }
}

void init_spdk_daemon(){
  spdk_init(NUM_DISKS);
}

void fini_spdk_daemon(){
  for (int i = 0; i < THREAD_NUM; i++) {
    spdk_free(cpu_buffers[i]);
  }
  spdk_env_fini();
}