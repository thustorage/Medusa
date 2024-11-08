from vllm._C import tensor_ops
from vllm._C import test_spdk


tensor_ops.init_spdk_daemon()
test_spdk.test()

