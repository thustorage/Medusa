import subprocess
import argparse
import time
import os
import signal

# test single batch size capture
# _BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [8 * i for i in range(1, 33)]
# test all batch size capture
_BATCH_SIZES_TO_CAPTURE = [0]
# CUDA_MODULE_LOADING=EAGER
save_tensor_command = "CUDA_VISIBLE_DEVICES=0 python examples/llm_engine_example.py --log ./breakdowns/{}_save_tensor.html --batch_size {} --model_name {} --save_tensor > ./data/{}/log_graph_{} 2>&1"
offline_command = "LD_PRELOAD=./libmylib.so CUDA_VISIBLE_DEVICES=0 python examples/llm_engine_example.py --log ./breakdowns/{}_offline.html --persist_cudagraph --batch_size {} --fast_start --model_name {} --async_load > ./data/{}/log_graph_{} 2>&1"
log_filter_command = "python log_filter.py --batch_size {} --model {}"
online_command = "CUDA_VISIBLE_DEVICES=0 python examples/llm_engine_example.py --log ./breakdowns/{}_online.html --persist_cudagraph --batch_size {} --fast_start --model_name {} --async_load --load_graph > ./data/{}/log_loadgraph_{} 2>&1"

# test cuda graph acceleration
with_graph_command = "CUDA_VISIBLE_DEVICES=0 python examples/llm_engine_example.py --log ./breakdowns/{}_with_graph_command.html --batch_size {} --model_name {} --test_cuda_graph > ./data/{}/log_loadgraph_{} 2>&1"
without_graph_command = "CUDA_VISIBLE_DEVICES=0 python examples/llm_engine_example.py --log ./breakdowns/{}_without_graph_command.html --batch_size {} --model_name {} --enforce-eager --test_cuda_graph > ./data/{}/log_loadgraph_{} 2>&1"

# test overall performance
overall_sync = "CUDA_VISIBLE_DEVICES=0 python examples/llm_engine_example.py --log ./breakdowns/{}_overall_sync.html --model_name {} --test_overall_performance sync > ./data/{}/log_overall_sync_{} 2>&1"
overall_async_load = "CUDA_VISIBLE_DEVICES=0 python examples/llm_engine_example.py --log ./breakdowns/{}_overall_async_load.html --model_name {} --test_overall_performance async_load --async_load > ./data/{}/log_overall_async_load_{} 2>&1"
# NOTE: fast_start includes reordered initialization, as well as reduced in each phase(e.g., no profile run)
# NOTE: with_cuda_graph further includes two optimizations: materialzed CUDA graph && warm-up parallel
overall_without_load_graph = "CUDA_VISIBLE_DEVICES=0 python examples/llm_engine_example.py --log ./breakdowns/{}_overall_without_load_graph.html --model_name {} --test_overall_performance without_load_graph --fast_start --async_load > ./data/{}/log_overall_without_load_graph_{} 2>&1"
overall_with_load_graph = "CUDA_VISIBLE_DEVICES=0 python examples/llm_engine_example.py --log ./breakdowns/{}_overall_with_load_graph.html --model_name {} --test_overall_performance with_load_graph --fast_start --async_load --persist_cudagraph --load_graph > ./data/{}/log_overall_with_load_graph_{} 2>&1"

model_names = ["Qwen-0.5B", "Qwen-1.8B", "Qwen-4B", "Qwen-7B", "Yi-6B", "Llama-7B", "Falcon-7B", "Yi-9B", "Llama-13B", "Qwen-14B"]

parser = argparse.ArgumentParser(
    description='Demo on using the LLMEngine class directly')
parser.add_argument('--save_tensor', action='store_true', default=False)
parser.add_argument('--offline', action='store_true', default=False)
parser.add_argument('--online', action='store_true', default=False)
# test cuda graph acceleration
parser.add_argument('--with_cuda_graph_cmd', action='store_true', default=False)
parser.add_argument('--without_cuda_graph', action='store_true', default=False)
parser.add_argument('--batch_step', type=int, default=-1)
# test overall performance
parser.add_argument('--test_overall_performance', action='store_true', default=False)
# test offline performance
parser.add_argument('--offline_performance', action='store_true', default=False)
args = parser.parse_args()

# args.save_tensor = True
# args.offline = True
# args.online = True

if args.with_cuda_graph_cmd or args.without_cuda_graph:
  model_names = ["Qwen-0.5B", "Qwen-1.8B", "Qwen-4B", "Llama-7B"]
elif args.test_overall_performance:
  model_names = ["Qwen-0.5B", "Qwen-1.8B", "Qwen-4B", "Qwen-7B", "Yi-6B", "Llama-7B", "Falcon-7B", "Yi-9B", "Llama-13B", "Qwen-14B"]
elif args.offline_performance:
  model_names = ["Qwen-0.5B", "Qwen-1.8B", "Qwen-4B", "Qwen-7B", "Yi-6B", "Llama-7B", "Falcon-7B", "Yi-9B", "Llama-13B", "Qwen-14B"]

for model_name in model_names:
  # NOTE: save tensor, write each tensor to individual file
  if args.save_tensor:
    for batch_size in _BATCH_SIZES_TO_CAPTURE:
      result = subprocess.run(f"mkdir -p /home/zsx/raidfs-back/tensors/{model_name}", shell=True, capture_output=True, text=True)
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
      
  # NOTE: test overall performance
  if args.test_overall_performance:
    c = overall_sync.format(model_name, model_name, model_name, model_name)
    print(c)
    result = subprocess.run(c, shell=True, capture_output=True, text=True)
    
    c = overall_async_load.format(model_name, model_name, model_name, model_name)
    print(c)
    result = subprocess.run(c, shell=True, capture_output=True, text=True)
    
    c = overall_without_load_graph.format(model_name, model_name, model_name, model_name)
    print(c)
    result = subprocess.run(c, shell=True, capture_output=True, text=True)
    
    c = overall_with_load_graph.format(model_name, model_name, model_name, model_name)
    print(c)
    result = subprocess.run(c, shell=True, capture_output=True, text=True)
    
  # NOTE: test offline performance
  if args.offline_performance:
    total_start_time = time.perf_counter()
    for batch_size in _BATCH_SIZES_TO_CAPTURE:
      c = offline_command.format(model_name, batch_size, model_name, model_name, batch_size)
      print(c)
      result = subprocess.run(c, shell=True, capture_output=True, text=True)
      
      log_filter_start_time = time.perf_counter()
      
      c = log_filter_command.format(batch_size, model_name)
      print(c)
      result = subprocess.run(c, shell=True, capture_output=True, text=True)
      print(result.stdout)
      print(result.stderr)
    total_end_time = time.perf_counter()
    with open(f"/home/zsx/vllm/experiments/offline/{model_name}", "w") as f:
      f.write(f"offline + log_filter time: {total_end_time - total_start_time} seconds, filter time: {total_end_time - log_filter_start_time} seconds.\n")
      f.flush()
    