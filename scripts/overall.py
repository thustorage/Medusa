import subprocess
import os
import re
import pandas as pd

overall_cmd = "python scripts/serverless_llm.py --test_overall_performance"
c = overall_cmd
result = subprocess.run(c, shell=True, capture_output=True, text=True)

dir_path = "breakdowns"
data = {}
filenames = {}

print("only load time", flush=True)
for filename in os.listdir(dir_path):
  if filename.find("overall") == -1:
    continue
  
  if filename.find("without_load_graph") != -1:
    continue
  
  with open(os.path.join(dir_path, filename), "r") as f:
    model_name = filename.split("_")[0]    
    mode = ("_".join(filename.split("_")[1:])).split('.html')[0]
    # read all lines of file
    while True:
      line = f.readline()
      if not line:
        break
      
      total_time_pattern = r'"init_phase2",".*?"time": (\d+\.\d+)'
      total_time = 0
      match = re.search(total_time_pattern, line)
      if match:
        total_time += float(match.group(1))
      
        
      if total_time != 0:
        if model_name not in data:
          data[model_name] = {}
        data[model_name]["Model"] = model_name
        data[model_name][mode] = total_time

df = pd.DataFrame.from_dict(data, orient='index')
df = df.reset_index(drop=True)
df['Model_name'] = df['Model'].str.split('-').str[0]
df['Model_num'] = df['Model'].str.split('-').str[1].str[:-1].astype(float)
df = df.sort_values(by=['Model_name', 'Model_num'])
df = df.drop(['Model_name', 'Model_num'], axis=1)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df.to_string(index=False), flush=True)

data = {}
print("runtime init time + load time", flush=True)
for filename in os.listdir(dir_path):
  if filename.find("overall") == -1:
    continue
  
  if filename.find("without_load_graph") != -1:
    continue
  with open(os.path.join(dir_path, filename), "r") as f:
    model_name = filename.split("_")[0]    
    mode = ("_".join(filename.split("_")[1:])).split('.html')[0]
    # read all lines of file
    while True:
      line = f.readline()
      if not line:
        break
      
      total_time_pattern = r'"init_phase1",".*?"time": (\d+\.\d+)'
      total_time = 0
      match = re.search(total_time_pattern, line)
      if match:
        total_time += float(match.group(1))

      
      total_time_pattern = r'"init_phase2",".*?"time": (\d+\.\d+)'
      match = re.search(total_time_pattern, line)
      if match:
        total_time += float(match.group(1))
      
        
      if total_time != 0:
        if model_name not in data:
          data[model_name] = {}
        data[model_name]["Model"] = model_name
        data[model_name][mode] = total_time

df = pd.DataFrame.from_dict(data, orient='index')
df = df.reset_index(drop=True)
df['Model_name'] = df['Model'].str.split('-').str[0]
df['Model_num'] = df['Model'].str.split('-').str[1].str[:-1].astype(float)
df = df.sort_values(by=['Model_name', 'Model_num'])
df = df.drop(['Model_name', 'Model_num'], axis=1)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df.to_string(index=False))