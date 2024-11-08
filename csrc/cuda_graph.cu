#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAGraph.h>

#include <cuda.h>

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#define CU_CALL(x) do { CUresult result = x; if (result != CUDA_SUCCESS) { printf("CUDA error: %d\n", result); return; } } while(0)

struct CUDAGraph_t : public at::cuda::CUDAGraph {
  cudaGraph_t get_graph() {
    return graph_;
  }

  void set_graph(cudaGraph_t &graph) {
    graph_ = graph;
  }

  void set_graph_exec(cudaGraphExec_t &graph_exec) {
    graph_exec_ = graph_exec;
  }

  int get_capture_dev() {
    return capture_dev_;
  }
};

std::vector<uint64_t> kernelParams;

void loadKernelNodeParams(const CUDA_KERNEL_NODE_PARAMS &in, CUDA_KERNEL_NODE_PARAMS &out) {
  printf("kernel node params:\n");
  printf("ctx: %p\n", in.ctx);
  printf("extra: %p\n", in.extra);
  printf("func: %p\n", in.func);
  printf("kern: %p\n", in.kern);
  printf("kernelParams: %p\n", in.kernelParams);

  kernelParams.push_back((uint64_t)in.kernelParams);

  out = in;
  // we don't set extra field
  if (in.extra) {
    printf("extra is not nil\n");
  }
  // we don't set kern, if func is not null
  if (in.func == nullptr) {
    printf("func is nil\n");
  } else {
    out.kern = nullptr;
  }
}

void loadMemcpyNodeParams(const CUDA_MEMCPY3D &in, CUDA_MEMCPY3D &out) {
  printf("memcpy node params:\n");
  printf("srcXInBytes: %d\n", in.srcXInBytes);
  printf("srcY: %d\n", in.srcY);
  printf("srcZ: %d\n", in.srcZ);
  printf("srcLOD: %d\n", in.srcLOD);

  out = in;
}

struct CUgraphNodeParam {
  CUgraphNodeType type;

  union {
    CUDA_KERNEL_NODE_PARAMS kernel_params;
    CUDA_MEMCPY3D memcpy_params;
  } p;
};

#define MAX_NODE_AND_EDGE_NUM 600

void save_cuda_graph (
  at::cuda::CUDAGraph& g
  ) {
  CUDAGraph_t *g_ = (CUDAGraph_t*)(&g);
  const CUgraph graph_ = g_->get_graph();
  CU_CALL(cuGraphDebugDotPrint(graph_, "cuda_graph.dot", 1));

  size_t numCUGraphNodes = MAX_NODE_AND_EDGE_NUM;
  CUgraphNode nodes[MAX_NODE_AND_EDGE_NUM];
  CU_CALL(cuGraphGetNodes(graph_, nodes, &numCUGraphNodes));
  printf("cuda graph nodes: %d\n", numCUGraphNodes);

  std::vector<CUgraphNodeParam> saved_nodes_params;

  std::map<CUgraphNode, int> node_to_idx;

  for (int i = 0; i < numCUGraphNodes; i++) {
    CUgraphNode node = nodes[i];
    CUgraphNodeType type;
    CU_CALL(cuGraphNodeGetType(node, &type));
    printf("node idx: %d, node type: %d\n", i, type);

    CUgraphNodeParam param;
    param.type = type;

    node_to_idx.insert(std::make_pair(node, i));

    switch (type) {
      case CU_GRAPH_NODE_TYPE_KERNEL: {
        CUDA_KERNEL_NODE_PARAMS pNodeParams;
        CU_CALL(cuGraphKernelNodeGetParams(node, &pNodeParams));
        param.p.kernel_params = pNodeParams;
        break;
      }
      case CU_GRAPH_NODE_TYPE_MEMCPY: {
        CUDA_MEMCPY3D pNodeParams;
        CU_CALL(cuGraphMemcpyNodeGetParams(node, &pNodeParams));
        param.p.memcpy_params = pNodeParams;
        break;
      }
      default: {
        printf("error node type: %d\n", type);
        break;
      }
    }

    saved_nodes_params.push_back(param);
  }

  CUgraphNode from[MAX_NODE_AND_EDGE_NUM];
  CUgraphNode to[MAX_NODE_AND_EDGE_NUM];
  size_t savedEdges = MAX_NODE_AND_EDGE_NUM;
  CU_CALL(cuGraphGetEdges(graph_, from, to, &savedEdges));

  int saved_from_idx[MAX_NODE_AND_EDGE_NUM];
  int saved_to_idx[MAX_NODE_AND_EDGE_NUM];
  for (int i = 0; i < savedEdges; i++) {
    int from_idx = node_to_idx[from[i]];
    int to_idx = node_to_idx[to[i]];
    saved_from_idx[i] = from_idx;
    saved_to_idx[i] = to_idx;
  }

  printf("=============== saved edges: %d\n", savedEdges);

  // ===========================================================

  // Create an new CUDA graph
  // cudaGraph_t newGraph;
  // CU_CALL(cuGraphCreate(&newGraph, 0));

  // std::vector<CUgraphNode> new_nodes_vec;

  // CUcontext ctx;
  // CU_CALL(cuDevicePrimaryCtxRetain(&ctx, 0));

  // for (int i = 0; i < saved_nodes_params.size(); i++) {
  //   printf("===============================\n");
    
  //   CUgraphNodeParam param = saved_nodes_params[i];
  //   CUgraphNode newNode;

  //   switch (param.type) {
  //     case CU_GRAPH_NODE_TYPE_KERNEL: {
  //       CUDA_KERNEL_NODE_PARAMS p;
  //       loadKernelNodeParams(param.p.kernel_params, p);
  //       CU_CALL(cuGraphAddKernelNode(&newNode, newGraph, nullptr, 0, &p));
  //       new_nodes_vec.push_back(newNode);
  //       break;
  //     }
  //     case CU_GRAPH_NODE_TYPE_MEMCPY: {
  //       CUDA_MEMCPY3D p;
  //       loadMemcpyNodeParams(param.p.memcpy_params, p);
  //       CU_CALL(cuGraphAddMemcpyNode(&newNode, newGraph, nullptr, 0, &p, ctx));
  //       new_nodes_vec.push_back(newNode);
  //       break;
  //     }
  //     default: {
  //       printf("error node type: %d\n", param.type);
  //       break;
  //     }
  //   }
  // }

  // // uint64_t minKernelParams = 0;
  // // for (int i = 0; i < kernelParams.size(); i++) {
  // //   if (minKernelParams == 0 || kernelParams[i] < minKernelParams) {
  // //     minKernelParams = kernelParams[i];
  // //   }
  // // }
  // // for (int i = 0; i < kernelParams.size(); i++) {
  // //   printf("kernel idx: %d, kernel params offset: %ld\n", i, kernelParams[i] - minKernelParams);
  // // }

  // CUgraphNode saved_from[MAX_NODE_AND_EDGE_NUM];
  // CUgraphNode saved_to[MAX_NODE_AND_EDGE_NUM];

  // for (int i = 0; i < savedEdges; i++) {
  //   saved_from[i] = new_nodes_vec[saved_from_idx[i]];
  //   saved_to[i] = new_nodes_vec[saved_to_idx[i]];
  // }

  // CU_CALL(cuGraphAddDependencies(newGraph, saved_from, saved_to, savedEdges));

  // g_->set_graph(newGraph);

  // CUgraphExec newGraphExec;
  // CU_CALL(cuGraphInstantiate(&newGraphExec, newGraph, 0));
  // g_->set_graph_exec(newGraphExec);
}
