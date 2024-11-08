# python scheduler.py --batch_size 0 --fast_start --model_name "Llama-13B" --async_load --persist_cudagraph --load_graph --qps 2.0 --run_time 30 > output.txt
import subprocess
import os
import re
import pandas as pd
import time
import signal

all_qps = [1,2,5,6,7,8,9]

model_names = ["Qwen-4B", "Llama-7B"]
# model_names = ["Llama-7B"]
# model_names = ["Qwen-4B"]

sync_cmd = 'python3 scripts/scheduler_ahead.py --batch_size 0 --model_name {} --qps {} --run_time {} > experiments/traces_throughput/qps{}/{}_sync 2>&1'
async_cmd = 'python3 scripts/scheduler_ahead.py --batch_size 0 --model_name {} --async_load --qps {} --run_time {} > experiments/traces_throughput/qps{}/{}_async 2>&1'
with_cuda_graph_cmd = 'python3 scripts/scheduler_ahead.py --batch_size 0 --model_name {} --fast_start --async_load --persist_cudagraph --load_graph --qps {} --run_time {} > experiments/traces_throughput/qps{}/{}_with_cuda_graph 2>&1'
without_cuda_graph_cmd = 'python3 scripts/scheduler_ahead.py --batch_size 0 --model_name {} --qps {} --run_time {} --enforce-eager > experiments/traces_throughput/qps{}/{}_without_cuda_graph 2>&1'


def clear_python_process():
  current_pid = os.getpid()
  try:
      python_pids = subprocess.check_output(['pgrep', 'python']).decode().splitlines()
      for pid in python_pids:
          pid = int(pid)
          if pid != current_pid:
              os.kill(pid, signal.SIGTERM)
              print(f"Killed Python process with PID: {pid}")
  except subprocess.CalledProcessError:
      print("No Python processes found.")

overall_data = {}
for qps in all_qps:
  dir_path = f"experiments/traces_throughput/qps{qps}"
  subprocess.run(f"mkdir {dir_path}", shell=True, capture_output=True, text=True)
  
  clear_python_process()
  
  for model_name in model_names:
    run_time = 120 / qps
    if qps > 1:
      run_time = 60
    c = sync_cmd.format(model_name, qps, run_time, qps, model_name)
    print(c)
    subprocess.run(c, shell=True, capture_output=True, text=True) 
    clear_python_process()

    c = async_cmd.format(model_name, qps, run_time, qps, model_name)
    print(c)
    subprocess.run(c, shell=True, capture_output=True, text=True)
    clear_python_process()

    c = with_cuda_graph_cmd.format(model_name, qps, run_time, qps, model_name)
    print(c)
    subprocess.run(c, shell=True, capture_output=True, text=True)
    clear_python_process()
    
    c = without_cuda_graph_cmd.format(model_name, qps, run_time, qps, model_name)
    print(c)
    subprocess.run(c, shell=True, capture_output=True, text=True)
    clear_python_process()

  average_throughput = 0
  for filename in os.listdir(dir_path):
    with open(os.path.join(dir_path, filename), "r") as f:
      data = {}
      TTFT_list = []
      ATT_list = []
      E2E_list = []

      # read all lines of file
      while True:
        line = f.readline()
        if not line:
          break
        
        mode = filename + "_qps_" + str(qps)
        if mode not in data:
          data[mode] = {}
          
        if line.find("Average") != -1:
          continue
        
        TTFT_pattern = r".*TTFT:  (\d+\.\d+)"
        ATT_pattern = r".*ATT:  (\d+\.\d+)"
        E2E_pattern = r".*E2E:  (\d+\.\d+)"
        
        TTFT = 0
        ATT = 0
        E2E = 0
        
        match = re.search(TTFT_pattern, line)
        if match:
          TTFT = float(match.group(1))
          TTFT_list.append(TTFT)
        
        match = re.search(ATT_pattern, line)
        if match:
          ATT = float(match.group(1))
          ATT_list.append(ATT)
          
        match = re.search(E2E_pattern, line)
        if match:
          E2E = float(match.group(1))
          E2E_list.append(E2E)
          
        total_time_pattern = r"Test total time: (\d+\.\d+)"
        match = re.search(total_time_pattern, line)
        if match:
          total_time = float(match.group(1))
          
        total_requests_pattern = r"Total requests: (\d+)"
        match = re.search(total_requests_pattern, line)
        if match:
          total_requests = int(match.group(1))
          average_throughput = total_requests / total_time
      
      data[mode]["TTFT"] = TTFT_list
      # data[mode]["ATT"] = ATT_list
      # data[mode]["E2E"] = E2E_list

      df = pd.DataFrame.from_dict(data, orient='index')
      df = df.explode(['TTFT'])
      pd.set_option('display.max_rows', None)
      pd.set_option('display.max_columns', None)
      pd.set_option('display.width', None)
      df.sort_values(['TTFT'], inplace=True)
      df['TTFT-CDF'] = df['TTFT'].rank() / len(df)
      # print(df)
      overall_data[mode] = {}
      # overall_data[mode]["TTFT_50"] = df[f'TTFT'].quantile(0.50)
      overall_data[mode]["TTFT_99"] = df[f'TTFT'].quantile(0.99)
      # overall_data[mode]["E2E_50"] = df[f'E2E'].quantile(0.50)
      # overall_data[mode]["E2E_99"] = df[f'E2E'].quantile(0.99)
      overall_data[mode]["throughput"] = average_throughput
  
df = pd.DataFrame.from_dict(overall_data, orient='index')
df.index.name = 'mode'
df = df.sort_values(['mode'])
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df)