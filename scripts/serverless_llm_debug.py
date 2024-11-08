import subprocess
import argparse

# test single batch size capture
# _BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [8 * i for i in range(1, 33)]
# test all batch size capture
_BATCH_SIZES_TO_CAPTURE = [0]
# CUDA_MODULE_LOADING=EAGER
save_tensor_command = "CUDA_VISIBLE_DEVICES=0 python examples/llm_engine_example.py --log ./breakdowns/{}_save_tensor.html --batch_size {} --model_name {} --save_tensor > ./data/{}/log_graph_{} 2>&1"
offline_command = "LD_PRELOAD=./libmylib.so CUDA_VISIBLE_DEVICES=0 python examples/llm_engine_example.py --log ./breakdowns/{}_offline.html --persist_cudagraph --batch_size {} --fast_start --model_name {} --async_load  > ./data/{}/log_graph_{} 2>&1"
log_filter_command = "python log_filter.py --batch_size {} --model {}"
online_command = "CUDA_VISIBLE_DEVICES=0 python examples/llm_engine_example.py --log ./breakdowns/{}_online.html --persist_cudagraph --batch_size {} --fast_start --model_name {} --async_load --load_graph > ./data/{}/log_loadgraph_{} 2>&1"

# test_multiple_batches
with_graph_command = "CUDA_VISIBLE_DEVICES=0 python examples/llm_engine_example.py --log ./breakdowns/{}_with_graph_command.html --persist_cudagraph --batch_size {} --fast_start --model_name {} --async_load --load_graph --test_multiple_batches > ./data/{}/log_loadgraph_{} 2>&1"
without_graph_command = "CUDA_VISIBLE_DEVICES=0 python examples/llm_engine_example.py --log ./breakdowns/{}_without_graph_command.html --batch_size {} --model_name {} --enforce-eager --test_multiple_batches > ./data/{}/log_loadgraph_{} 2>&1"

# model_names = ["Llama-13B", "Llama-7B", "Yi-6B", "Yi-9B", "Qwen-14B", "Qwen-7B", "Qwen-4B", "Qwen-1.8B", "Qwen-0.5B", "Falcon-7B"]
model_names = ["Llama-13B"]
# model_names = ["Falcon-7B"]
# model_names = ["Llama-7B"]
# model_names = ["Yi-6B"]
# model_names = ["Qwen-1.8B"]
# model_names = ["Qwen-0.5B"]

parser = argparse.ArgumentParser(
    description='Demo on using the LLMEngine class directly')
parser.add_argument('--save_tensor', action='store_true', default=False)
parser.add_argument('--offline', action='store_true', default=False)
parser.add_argument('--online', action='store_true', default=False)
# test_multiple_batches
parser.add_argument('--with_cuda_graph_cmd', action='store_true', default=False)
parser.add_argument('--without_cuda_graph', action='store_true', default=False)
args = parser.parse_args()

# args.save_tensor = True
# args.offline = True
# args.online = True

for model_name in model_names:
  # NOTE: save tensor, write each tensor to individual file
  if args.save_tensor:
    for batch_size in _BATCH_SIZES_TO_CAPTURE:
      c = save_tensor_command.format(model_name, batch_size, model_name, model_name, batch_size)
      print(c)
      result = subprocess.run(c, shell=True, capture_output=True, text=True)
      
  # NOTE: offline, save captured graph, please recompile pytorch with SERVERLESS_LOG
  if args.offline:
    for batch_size in _BATCH_SIZES_TO_CAPTURE:
      c = offline_command.format(model_name, batch_size, model_name, model_name, batch_size)
      print(c)
      result = subprocess.run(c, shell=True, capture_output=True, text=True)
      
      c = log_filter_command.format(batch_size, model_name)
      print(c)
      result = subprocess.run(c, shell=True, capture_output=True, text=True)
      print(result.stdout)
      print(result.stderr)

  # NOTE: online, load captured graph, please recompile pytorch without SERVERLESS_LOG
  if args.online:
    for batch_size in _BATCH_SIZES_TO_CAPTURE:
      c = online_command.format(model_name, batch_size, model_name, model_name, batch_size)
      print(c)
      result = subprocess.run(c, shell=True, capture_output=True, text=True) 
      
      
      orig_out_file = f"/mnt/memfs/data/{model_name}/log_graph_{batch_size}"
      new_out_file = f"/mnt/memfs/data/{model_name}/log_loadgraph_{batch_size}"
      
      orig_outputs = []
      with open(orig_out_file, "r") as f:
        for line in f:
          if line.startswith("RequestOutput"):
            orig_outputs.append(line)
            
      new_outputs = []
      with open(new_out_file, "r") as f:
        for line in f:
          if line.startswith("RequestOutput"):
            new_outputs.append(line)
      
      if len(orig_outputs) == 0 or len(new_outputs) == 0:
        print(f"========= RequestOutput size is 0!!!")
        exit(-1)

      for i in range(len(orig_outputs)):
        if orig_outputs[i] != new_outputs[i]:
          print(f"========= Output mismatch at batch_size: {batch_size} request: {i}")
          exit(-1)
    
  # NOTE: online with cuda graph, test with multiple batch sizes
  if args.with_cuda_graph_cmd:
    for batch_size in _BATCH_SIZES_TO_CAPTURE:
      c = with_graph_command.format(model_name, batch_size, model_name, model_name, batch_size)
      print(c)
      result = subprocess.run(c, shell=True, capture_output=True, text=True) 
    
  # NOTE: baseline without cuda graph, test with multiple batch sizes
  if args.without_cuda_graph:
    for batch_size in _BATCH_SIZES_TO_CAPTURE:
      c = without_graph_command.format(model_name, batch_size, model_name, model_name, batch_size)
      print(c)
      result = subprocess.run(c, shell=True, capture_output=True, text=True) 