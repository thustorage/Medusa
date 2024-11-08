// #include <torch/extension.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <ATen/cuda/CUDAGraph.h>

// #include <cuda.h>

// #include <iostream>
// #include <string>
// #include <fstream>
// #include <thread>
// #include <vector>
// #include <atomic>
// #include <sys/mman.h>

// void save_tensor (
//   const torch::Tensor &tensor,
//   std::string filename
// ) {
//   std::ofstream file(filename, std::ios::binary);

//   if (!file.is_open()) {
//       std::cerr << "Failed to open file" << std::endl;
//       exit(-1);
//   }

//   void* data_ptr = tensor.data_ptr();

//   size_t tensor_size_bytes = tensor.numel() * tensor.itemsize();

//   file.write(static_cast<const char*>(data_ptr), tensor_size_bytes);

//   file.close();
// }

// void load_tensor (
//   torch::Tensor tensor,
//   std::string filename
// ) {
//   std::ifstream file(filename, std::ios::binary);

//   if (!file.is_open()) {
//       std::cerr << "Failed to open file" << std::endl;
//       exit(-1);
//   }

//   void* data_ptr = tensor.data_ptr();

//   size_t tensor_size_bytes = tensor.numel() * tensor.itemsize();

//   void* cpu_buffer = malloc(tensor_size_bytes);

//   file.read(static_cast<char*>(cpu_buffer), tensor_size_bytes);
//   file.close();

//   cudaMemcpy(data_ptr, cpu_buffer, tensor_size_bytes, cudaMemcpyHostToDevice);

//   free(cpu_buffer);
// }

// std::thread async_load_thread;

// std::atomic_bool async_request = false;
// std::vector<torch::Tensor> tensors;
// std::vector<std::string> filenames;

// // change llm_engine.py as well, pytorch bindcore
// // and change CUDAGraph as well, pytorch c++ bindcore
// // 0 - 8 async load c++
// // 9 - 47 for pytorch python
// #define CORE_NO 0
// #define THREAD_NUM 8
// #define CPU_BUFFER_SIZE (4UL << 30)

// void bind_core(int core_no) {
//   cpu_set_t cpu_mask;
//   CPU_ZERO(&cpu_mask);
//   CPU_SET(core_no, &cpu_mask);

//   if (sched_setaffinity(0, sizeof(cpu_set_t), &cpu_mask) != 0) {
//     std::cerr << "Failed to set thread affinity mask." << std::endl;
//     exit(-1);
//   }
// }

// void load_tensor (int thread_no, void* cpu_buffer, std::vector<torch::Tensor> tensors, std::vector<std::string> filenames) {
//   bind_core(CORE_NO + 1 + thread_no);
//   cudaStream_t stream;
//   cudaError_t result;
//   result = cudaStreamCreate(&stream);
//   uint64_t offset = 0;
//   for (size_t i = 0; i < tensors.size(); i++) {
//     std::ifstream file(filenames[i], std::ios::binary);

//     if (!file.is_open()) {
//         std::cerr << "Failed to open file" << std::endl;
//         exit(-1);
//     }

//     void* data_ptr = tensors[i].data_ptr();

//     size_t tensor_size_bytes = tensors[i].numel() * tensors[i].itemsize();

//     if (tensor_size_bytes > CPU_BUFFER_SIZE) {
//       std::cerr << "Tensor size is too large for CPU buffer" << std::endl;
//       exit(-1);
//     }

//     if (tensor_size_bytes + offset > CPU_BUFFER_SIZE) {
//       cudaStreamSynchronize(stream);
//       offset = 0;
//     }

//     std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
//     spdk.read(static_cast<char*>(cpu_buffer) + offset, tensor_size_bytes);
//     std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//     double bw = tensor_size_bytes / time_span.count();
//     std::cout << "Thread " << thread_no << " read " << tensor_size_bytes << " bytes in " << time_span.count() << " seconds, bandwidth: " << bw / 1024 / 1024 << " MB/s" << std::endl;
//     file.close();

//     int ret = cudaMemcpyAsync(data_ptr, cpu_buffer + offset, tensor_size_bytes, cudaMemcpyHostToDevice, stream);
//     if (ret != 0) {
//       std::cerr << "Failed to copy data to GPU" << std::endl;
//       exit(-1);
//     }
//     offset += tensor_size_bytes;
//   }

//   cudaStreamSynchronize(stream);

//   result = cudaStreamDestroy(stream);
// }

// #define CU_CALL(x) do { CUresult result = x; if (result != CUDA_SUCCESS) { printf("%s:%s:%d CUDA error: %d\n", __FILE__, __func__, __LINE__, result); exit(-1); } } while(0)

// void async_load_process() {
//   bind_core(CORE_NO);

//   cudaFree(0);

//   std::vector<void*> cpu_buffers;

//   for (int i = 0; i < THREAD_NUM; i++) {
//       void *cpu_buffer;
//       cudaMallocHost(&cpu_buffer, CPU_BUFFER_SIZE);
//       cpu_buffers.push_back(cpu_buffer);
//   }

//   while (true) {
//     if (async_request == true) {
//       std::vector<std::thread> threads;
//       for (int i = 0; i < THREAD_NUM; i++) {
//         size_t start = tensors.size() / THREAD_NUM * i;
//         size_t end = tensors.size() / THREAD_NUM * (i + 1);
//         if (i == THREAD_NUM - 1) {
//           end = tensors.size();
//         }
//         std::vector<torch::Tensor> tensors_slice(tensors.begin() + start, tensors.begin() + end);
//         std::vector<std::string> filenames_slice(filenames.begin() + start, filenames.begin() + end);

//         void* cpu_buffer = cpu_buffers[i];
//         threads.push_back(std::thread([i, cpu_buffer, tensors_slice, filenames_slice] {
//           load_tensor(i, cpu_buffer, tensors_slice, filenames_slice);
//         }));
//       }
//       for (auto &t: threads) {
//         t.join();
//       }
//       for (int i = 0; i < THREAD_NUM; i++) {
//         cudaFree(cpu_buffers[i]);
//       }
//       tensors.clear();
//       filenames.clear();
//       async_request = false;
//       return;
//     } else {
//       usleep(1000);
//     }
//   }
// }

// void init_load_thread() {
//   printf("Initializing async load threads...\n");
//   fflush(stdout);
//   async_load_thread = std::thread([] {
//     async_load_process();
//   });
// }

// void load_tensor_async_prepare (
//   torch::Tensor tensor,
//   std::string filename
// ) {
//   tensors.push_back(tensor);
//   filenames.push_back(filename);
// }

// void load_tensor_async_start () {
//   async_request = true;
// }

// void load_tensor_sync_all () {
//   printf("Waiting for async load threads to finish...\n");
//   fflush(stdout);
//    while (true) {
//     if (async_load_thread.joinable())
//       async_load_thread.join();
//     return;
//   }
// }