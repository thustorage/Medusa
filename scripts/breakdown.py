import subprocess
import os
import re
import pandas as pd

dir_path = "breakdowns"
data = {}

for filename in os.listdir(dir_path):
  with open(os.path.join(dir_path, filename), "r") as f:
    if filename.find("overall") == -1:
      continue
    
    if filename.find("without_load_graph") != -1:
      continue
    
    if filename.find("overall_sync") == -1:
      continue
    
    mode = filename.split('.html')[0]
    mode = mode.split('_')[0]
    # read all lines of file
    while True:
      line = f.readline()
      if not line:
        break
      
      runtime_init_pattern = r'"init_phase1",".*?"time": (\d+\.\d+)'
      runtime_time = 0
      match = re.search(runtime_init_pattern, line)
      if match:
        runtime_time += float(match.group(1))
      
      total_time_pattern = r'"init_phase2",".*?"time": (\d+\.\d+)'
      total_time = 0
      match = re.search(total_time_pattern, line)
      if match:
        total_time += float(match.group(1))
        
      
      init_model_pattern = r'"load_model",".*?"time": (\d+\.\d+)'
      init_model_time = 0
      match = re.search(init_model_pattern, line)
      if match:
        init_model_time += float(match.group(1))
        
        
      init_tokenizer_pattern = r'"_init_tokenizer",".*?"time": (\d+\.\d+)'
      init_tokenizer_time = 0
      match = re.search(init_tokenizer_pattern, line)
      if match:
        init_tokenizer_time = float(match.group(1))
      
      allocate_gpu_cache_pattern = r'"init_cache_engine",".*?"time": (\d+\.\d+)'
      profile_num_available_blocks = r'"profile_num_available_blocks",".*?"time": (\d+\.\d+)'
      init_kv_cache_time = 0
      match = re.search(profile_num_available_blocks, line)
      if match:
        init_kv_cache_time += float(match.group(1))
      match = re.search(allocate_gpu_cache_pattern, line)
      if match:
        init_kv_cache_time += float(match.group(1))
        
        
      load_weights_pattern = r'"load_weights",".*?"time": (\d+\.\d+)'
      load_tensor_sync_all_pattern = r'"PyCapsule.load_tensor_sync_all",".*?"time": (\d+\.\d+)'
        
      load_weights_total_time = 0
      prep_warmup_time = 0
   
      match = re.search(load_weights_pattern, line)
      if match:
        if filename.find("overall_with_load_graph") != -1:
          prep_warmup_time = float(match.group(1))
        load_weights_total_time += float(match.group(1))
      if filename.find("overall_sync") == -1:
        load_weights_total_time += init_tokenizer_time
        if filename.find("overall_with_load_graph") == -1:
          load_weights_total_time += init_kv_cache_time
        match = re.search(load_tensor_sync_all_pattern, line)
        if match:
          load_weights_total_time += float(match.group(1))

      
      
      capture_model_pattern = r'"capture_model",".*?"time": (\d+\.\d+)'
      capture_model_time = 0
      match = re.search(capture_model_pattern, line)
      if match:
        capture_model_time = float(match.group(1))
      
      if filename.find("overall_sync") == -1:
        match = re.search(load_tensor_sync_all_pattern, line)
        if match:
          capture_model_time -= float(match.group(1))

        
      if init_tokenizer_time != 0 and mode not in data:
        data[mode] = {}
        data[mode]["init_tokenizer"] = init_tokenizer_time
        data[mode]["load_weights"] = load_weights_total_time
        data[mode]["init_kv_cache"] = init_kv_cache_time
        data[mode]["capture_model"] = capture_model_time
        # data[mode]["prep_warmup"] = prep_warmup_time
        data[mode]["init_model"] = init_model_time
        if filename.find("overall_sync") != -1:
          data[mode]["total_load_time"] = init_tokenizer_time + load_weights_total_time + init_kv_cache_time + capture_model_time + init_model_time
        data[mode]["diff_total_time_in_file"] = total_time - data[mode]["total_load_time"]
        data[mode]["runtime_time"] = runtime_time
        
        
df = pd.DataFrame.from_dict(data, orient='index')
df.index.name = 'Model'
df = df.sort_values(by=['Model'])
df['capture_model'] = df['capture_model'] / df['total_load_time'] * 100
df['init_kv_cache'] = df['init_kv_cache'] / df['total_load_time'] * 100
df['init_tokenizer'] = df['init_tokenizer'] / df['total_load_time'] * 100
df['load_weights'] = df['load_weights'] / df['total_load_time'] * 100
df['init_model'] = df['init_model'] / df['total_load_time'] * 100

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df)

