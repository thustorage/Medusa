import subprocess
import os
import re
import pandas as pd

offline_cmd = "python scripts/serverless_llm.py --offline_performance"
c = offline_cmd
print(c)
result = subprocess.run(c, shell=True, capture_output=True, text=True)
print(result.stdout)
print(result.stderr)


dir_path = "experiments/offline"
data = {}

for filename in os.listdir(dir_path):
  with open(os.path.join(dir_path, filename), "r") as f:
    model_name = filename
    # read all lines of file
    while True:
      line = f.readline()
      if not line:
        break
  
      pattern = r"offline \+ log_filter time: (\d+\.\d+) seconds, filter time: (\d+\.\d+) seconds."
      match = re.search(pattern, line)
      if match:
        offline_filter_time = float(match.group(1))
        filter_time = float(match.group(2))
        
        if model_name not in data:
          data[model_name] = {}
          data[model_name]["Model"] = model_name
          data[model_name][f"total_time"] = offline_filter_time
          data[model_name][f"analysis_stage"] = filter_time
          data[model_name][f"capture_stage"] = offline_filter_time - filter_time
        
      else:
        print(f"Error: {filename} {line}")

df = pd.DataFrame.from_dict(data, orient='index')
df = df.reset_index(drop=True)
df['Model_name'] = df['Model'].str.split('-').str[0]
df['Model_num'] = df['Model'].str.split('-').str[1].str[:-1].astype(float)
df = df.sort_values(by=['Model_name', 'Model_num'])
df = df.drop(['Model_name', 'Model_num'], axis=1)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df)

