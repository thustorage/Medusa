import os
import re
import numpy as np
import pandas as pd

root_path = "./data"

def get_model_name(model_name):
  if model_name == "Falcon-7B":
    return "models--tiiuae--falcon-7b"
  if model_name == "Llama-7B":
    return "models--openlm-research--open_llama_7b"
  if model_name == "Llama-13B":
    return "models--openlm-research--open_llama_13b"
  if model_name == "Qwen-0.5B":
    return "models--Qwen--Qwen1.5-0.5B"
  if model_name == "Qwen-1.8B":
    return "models--Qwen--Qwen1.5-1.8B"
  if model_name == "Qwen-4B":
    return "models--Qwen--Qwen1.5-4B"
  if model_name == "Qwen-7B":
    return "models--Qwen--Qwen2-beta-7B"
  if model_name == "Qwen-14B":
    return "models--Qwen--Qwen1.5-14B"
  if model_name == "Yi-6B":
    return "models--01-ai--Yi-6B"
  if model_name == "Yi-9B":
    return "models--01-ai--Yi-9B"


# traverse all directories under ./data, open log_graph_0, match the number in "cuda graph nodes: 442" and sum them
data = {}
all_cuda_graph_nodes = 0
for dirpath, dirnames, filenames in os.walk(root_path):
  for file in filenames:
    model_name = dirpath.split("/")[-1]
    if model_name not in data:
      data[model_name] = {}
      data[model_name]["cuda_graph_nodes"] = 0
      total_size = 0
      for i_dirpath, i_dirnames, i_filenames in os.walk(f"/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/{get_model_name(model_name)}/snapshots"):
        for f in i_filenames:
          fp = os.path.join(i_dirpath, f)
          total_size += os.path.getsize(fp)
      data[model_name]["total_size(GB)"] = total_size / 1024 / 1024 / 1024
    
    if file == "log_graph_0":
      # print(os.path.join(dirpath, file))
      with open(os.path.join(dirpath, file), "r") as f:
        lines = f.readlines()
        for line in lines:
          match = re.search(r"cuda graph nodes: (\d+)", line)
          if match:
            all_cuda_graph_nodes += int(match.group(1))
            data[model_name]["cuda_graph_nodes"] += int(match.group(1))   
    
        
df = pd.DataFrame.from_dict(data, orient='index')
df.index.name = 'mode'
df = df.sort_values(['mode'])
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df.T)


# traverse all directories under ./data, open files ends up with _func_param, match the line of "addr_corresponding_malloc_idx: 96,1134,1135,886,", and count the number which is seperated by comma
# and extract the longest number of them
all_addr_corresponding_malloc_idx = 0
max_len_of_the_line = 0
for dirpath, dirnames, filenames in os.walk(root_path):
  for file in filenames:
    if file.endswith("_func_param"):
      # print(os.path.join(dirpath, file))
      with open(os.path.join(dirpath, file), "r") as f:
        lines = f.readlines()
        for line in lines:
          match = re.search(r"addr_corresponding_malloc_idx: ([-?\d,]+),", line)
          if match:
            len_of_this_line = len(match.group(1).split(","))
            max_len_of_the_line = max(max_len_of_the_line, len_of_this_line)
            all_addr_corresponding_malloc_idx += len_of_this_line
            


print("CUDA nodes number: ", all_cuda_graph_nodes)
print("Pointers number: ", all_addr_corresponding_malloc_idx)
print("Max pointers in a line: ", max_len_of_the_line)
print("Average pointers per CUDA nodes: ", all_addr_corresponding_malloc_idx / all_cuda_graph_nodes)