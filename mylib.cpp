#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cuda.h>

cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ) {
  cudaError_t (*lcudaLaunchKernel) ( const void*, dim3, dim3, void**, size_t, cudaStream_t ) = ( cudaError_t (*) ( const void*, dim3, dim3, void**, size_t, cudaStream_t )) dlsym (RTLD_NEXT, "cudaLaunchKernel");

  CUfunction function;
  if (cudaGetFuncBySymbol (&function, func)) {
    printf("error! cudaGetFuncBySymbol\n");
  }

  const char *name;
  if (cuFuncGetName(&name, function)) {
    printf("error! cuFuncGetName\n");
  
  }
  printf("cudaLaunchKernel hooked func.name mangled = %s\n", name);

  return lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t cudaLaunchKernelExC ( const cudaLaunchConfig_t* config, const void* func, void** args ) {
  cudaError_t (*lcudaLaunchKernelExC) ( const cudaLaunchConfig_t*, const void*, void** ) = ( cudaError_t (*) ( const cudaLaunchConfig_t*, const void*, void** )) dlsym (RTLD_NEXT, "cudaLaunchKernelExC");
  printf("cudaLaunchKernelExC hooked\n");
  return lcudaLaunchKernelExC(config, func, args);
}

cudaError_t cudaLaunchCooperativeKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ) {
  cudaError_t (*lcudaLaunchCooperativeKernel) ( const void*, dim3, dim3, void**, size_t, cudaStream_t ) = ( cudaError_t (*) ( const void*, dim3, dim3, void**, size_t, cudaStream_t )) dlsym (RTLD_NEXT, "cudaLaunchCooperativeKernel");
  printf("cudaLaunchCooperativeKernel hooked\n");
  return lcudaLaunchCooperativeKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t cudaLaunchHostFunc ( cudaStream_t stream, cudaHostFn_t fn, void* userData ) {
  cudaError_t (*lcudaLaunchHostFunc) ( cudaStream_t, cudaHostFn_t, void* ) = ( cudaError_t (*) ( cudaStream_t, cudaHostFn_t, void* )) dlsym (RTLD_NEXT, "cudaLaunchHostFunc");
  printf("cudaLaunchHostFunc hooked\n");
  return lcudaLaunchHostFunc(stream, fn, userData);
}

cudaError_t cudaGraphAddKernelNode ( cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams ) {
  cudaError_t (*lcudaGraphAddKernelNode) ( cudaGraphNode_t*, cudaGraph_t, const cudaGraphNode_t*, size_t, const cudaKernelNodeParams* ) = ( cudaError_t (*) ( cudaGraphNode_t*, cudaGraph_t, const cudaGraphNode_t*, size_t, const cudaKernelNodeParams* )) dlsym (RTLD_NEXT, "cudaGraphAddKernelNode");
  printf("cudaGraphAddKernelNode hooked\n");
  return lcudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
}
