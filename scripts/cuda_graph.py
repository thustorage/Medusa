import subprocess
import os
import re
import pandas as pd

with_cuda_graph_cmd = "python scripts/serverless_llm.py --with_cuda_graph_cmd"
subprocess.run(with_cuda_graph_cmd, shell=True, capture_output=True, text=True) 

without_cuda_graph_cmd = "python scripts/serverless_llm.py --without_cuda_graph"
subprocess.run(without_cuda_graph_cmd, shell=True, capture_output=True, text=True)


model_names = ["Qwen-0.5B", "Qwen-1.8B", "Qwen-4B", "Llama-7B"]
dir_path = "experiments/cuda_graph"
data = {}

for filename in os.listdir(dir_path):
  with open(os.path.join(dir_path, filename), "r") as f:
    # read all lines of file
    while True:
      line = f.readline()
      if not line:
        break
  
      pattern = r"batch size (\d+) output len (\d+) time: (\d+\.\d+) milliseconds"
      match = re.search(pattern, line)
      if match:
        batch_size = int(match.group(1))
        output_len = int(match.group(2))
        time = float(match.group(3))
        
        if "_without_graph" in filename:
          model_name = filename.split("_without_graph")[0]
          without_graph = True
        else:
          model_name = filename
          without_graph = False
        if model_name in model_names:
          key = (model_name)
          
          if key not in data:
            data[key] = {"Model": model_name}
          
          if without_graph:
            data[key][f"bs{batch_size}_o{output_len}_w/o_graph_(ms)"] = time
          else:
            data[key][f"bs{batch_size}_o{output_len}_with_graph_(ms)"] = time
      else:
        print(f"Error: {filename} {line}")

df = pd.DataFrame.from_dict(data, orient='index')
df = df.reset_index(drop=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df.to_string(index=False))

