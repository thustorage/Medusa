import multiprocessing as mp
from multiprocessing import Process, Queue
import multiprocessing
import pickle
import random
import numpy as np
import argparse
from typing import List, Tuple
from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from pyinstrument import Profiler
import torch
import time
import threading
from vllm._C import tensor_ops
import os
import sys

seed = 42
random.seed(seed)
np.random.seed(seed)

def process_requests(engine: LLMEngine,
                     recv_q,
                     send_q):
    """Continuously process a list of prompts and handle the outputs."""
    while not recv_q.empty() or engine.has_unfinished_requests():
        has_new = False
        if(not recv_q.empty()):
            has_new = True
            req = recv_q.get()
            print(req)
            payload = req["payload"]
            prompt = "TO "* (payload[0]-1)
            prompt = prompt.strip()
            sampling_params = SamplingParams(ignore_eos=True, max_tokens=payload[1])
            engine.add_request(req["request_id"], prompt, sampling_params)
    
        request_outputs: List[RequestOutput] = engine.step()
        if(has_new):
            send_q.put({"type":"first_time","data":time.time(),"request_id":req["request_id"]})
        for request_output in request_outputs:
            if request_output.finished:
                # print(request_output)
                send_q.put({"type":"last_time","data":time.time(),"request_id":request_output.request_id})
    print("trying to exit the worker")
    send_q.put({"type":"exit"})
    while True:
        has_new = False
        if(not recv_q.empty()):
            has_new = True
            req = recv_q.get()
            if(req["type"] == "exitack"):
                send_q.put({"type":"exitfin"})
                exit(0)
            print(req)
            payload = req["payload"]
            prompt = "TO "* (payload[0]-1)
            prompt = prompt.strip()
            sampling_params = SamplingParams(ignore_eos=True, max_tokens=payload[1])
            engine.add_request(req["request_id"], prompt, sampling_params)
        request_outputs: List[RequestOutput] = engine.step()
        if(has_new):
            send_q.put({"type":"first_time","data":time.time(),"request_id":req["request_id"]})
        for request_output in request_outputs:
            if request_output.finished:
                send_q.put({"type":"last_time","data":time.time(),"request_id":request_output.request_id})


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)

def get_args():
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument('--qps', type=float, default=1.5)
    parser.add_argument('--run_time', type=float, default=120)
    parser.add_argument('--test_cuda_graph', action='store_true', default=False)
    parser.add_argument('--test_overall_performance', type=str, choices=['sync', 'async_load', 'without_load_graph', 'with_load_graph'], default=None)
    args = parser.parse_args()
    
    if args.model_name == "Llama-13B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--openlm-research--open_llama_13b/snapshots/b6d7fde8392250730d24cc2fcfa3b7e5f9a03ce8/'
    elif args.model_name == "Llama-7B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--openlm-research--open_llama_7b/snapshots/6fb184ff23774c25bf84b3628e49c8b78372c7be/'
    elif args.model_name == "Yi-6B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--01-ai--Yi-6B/snapshots/b5b30297823fd4fbb8d76dc58ae44c93104faa2b/'
    elif args.model_name == "Yi-9B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--01-ai--Yi-9B/snapshots/2af40f9fdd3ad4a56ddeb52a21e3f9c720d82353/'
    elif args.model_name == "Qwen-14B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--Qwen--Qwen1.5-14B/snapshots/dce4b190d34470818e5bec2a92cb8233aaa02ca2/'
        args.max_model_len = 4096
    elif args.model_name == "Qwen-7B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--Qwen--Qwen2-beta-7B/snapshots/831096e3a59a0789a541415da25ef195ceb802fe/'
        args.max_model_len = 4096
    elif args.model_name == "Qwen-4B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--Qwen--Qwen1.5-4B/snapshots/a66363a0c24e2155c561e4b53c658b1d3965474e/'
        args.max_model_len = 4096
    elif args.model_name == "Qwen-1.8B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--Qwen--Qwen1.5-1.8B/snapshots/7846de7ed421727b318d6605a0bfab659da2c067/'
        args.max_model_len = 4096
    elif args.model_name == "Qwen-0.5B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--Qwen--Qwen1.5-0.5B/snapshots/8f445e3628f3500ee69f24e1303c9f10f5342a39/'
        args.max_model_len = 4096
    elif args.model_name == "Falcon-7B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--tiiuae--falcon-7b/snapshots/898df1396f35e447d5fe44e0a3ccaaaa69f30d36/'
    else:
        print("model_name not supported", args.model_name)
        exit(-1)
    return args

def vllm_worker(recv_q, send_q, args):
    # polling until get start message
    message = recv_q.get()
    assert message["type"] == "start"
    cuda_id = message["cuda"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    
    engine = initialize_engine(args)
    engine.init_phase2()
    print("Worker {} started".format(cuda_id),flush=True)
    process_requests(engine, recv_q, send_q)
    
def generate_requests(request_rate, run_time):
    # Generate timestamps for requests using Poisson distribution.
    time_quantum = 10
    lam = request_rate * (time_quantum / 1000)
    quantums_per_sec = 1000 / time_quantum
    print("lam:",lam,flush=True)
    arrival_times = np.random.poisson(lam=lam, size=int(run_time * quantums_per_sec))
    timestamps = []
    for i, n in enumerate(arrival_times):
        timestamps += [i * (time_quantum / 1000)] * n
    return timestamps

worker_id_g = 1
data_points = {}
class Worker:
    def __init__(self, args):
        global worker_id_g
        self.worker_id = worker_id_g
        self.send_q = Queue()
        self.recv_q = Queue()
        self.onfly_requests = 0
        self.p = Process(target=vllm_worker, args=(self.send_q, self.recv_q,args))
        self.p.start()
        worker_id_g += 1
        self.stop = False
        
    def add(self,request_id, request):
        self.send_q.put({"type":"request","payload":request,"request_id":request_id})
        self.onfly_requests += 1
    
    def poll(self):
        global data_points
        while(not self.recv_q.empty()):
            resp = self.recv_q.get()
            if(resp["type"] == "last_time"):
                self.onfly_requests -= 1
                print(resp,flush=True)
                data_points[resp["request_id"]].append(resp["data"])
            elif resp["type"] == "first_time":
                print(resp,flush=True)
                data_points[resp["request_id"]].append(resp["data"])
            elif resp["type"] == "exit":
                self.stop = True
                self.send_q.put({"type":"exitack"})
            elif resp["type"] == "exitfin":
                return True
        return False
    
    def start(self, cuda_id):
        self.send_q.put({"type":"start","cuda":cuda_id})
TOTAL_GPUS = 4
MAX_LENGTH = 2048
scale_num = 0
worker_slots = [None] * TOTAL_GPUS
worker_poll = []
MAX_ONFLY = 16

def live_checker():
    while(True):
        to_remove = []
        for idx in range(TOTAL_GPUS):
            if(worker_slots[idx] is None):
                continue
            exited = worker_slots[idx].poll()
            if(exited):
                to_remove.append(idx)
        for idx in to_remove:
            worker_slots[idx].p.kill()
            worker_slots[idx] = None
        time.sleep(1)

def drain_die_send_ack():
    for worker in stop_workers:
        send_qs[worker].put({"type":"exitack"})
        send_qs[worker].close()
        live_workers.remove(worker)
    stop_workers.clear()

def check_and_scale():
    global scale_num
    # case1. find a worker with less than onfly requests
    for i in range(TOTAL_GPUS):
        if((not worker_slots[i] is None) and worker_slots[i].onfly_requests < MAX_ONFLY and not worker_slots[i].stop):
            return worker_slots[i]
        
    # case2. create a new worker if slot available
    for i in range(TOTAL_GPUS):
        if(worker_slots[i] is None):
            
            print ("Creating new worker on GPU",i,"at time",time.time(),flush=True)
            # find a worker in pool 
            if(len(worker_poll) > 0):
                scale_num += 1
                worker_slots[i] = worker_poll.pop()
                worker_slots[i].start(i)
            else:
                print("No workers in pool")
                assert False
            return worker_slots[i]
        
    # case3. find a worker with min onfly requests
    min_onfly_requests = min([worker_slots[i].onfly_requests for i in range(TOTAL_GPUS)] and not worker_slots[i].stop)
    for i in range(TOTAL_GPUS):
        if(onfly_requests[i] == min_onfly_requests):
            return worker_slots[i]
    assert False
    
if __name__ == '__main__':
    args = get_args()
    mp.set_start_method('spawn')
    
    for i in range(10):
        worker_poll.append(Worker(args))
        
    tensor_ops.init_spdk_daemon()
    time.sleep(5)
    
    dataset = pickle.load(open('/home/zsx/vllm/scripts/sharegpt_opt_text_completion_length.pkl', 'rb'))
    dataset = [i for i in dataset if i[0]+i[1]<MAX_LENGTH]
    random.shuffle(dataset)
    timestamps = generate_requests(args.qps, args.run_time)
    dataset = dataset[:len(timestamps)]
    start = time.time()
    # background thread for checking live workers
    threading.Thread(target=live_checker, args=()).start()
    for i in range(len(dataset)):
        target_time_stamp = start + timestamps[i]
        while(time.time() < target_time_stamp):
            pass
        worker = check_and_scale()
        worker.add(i, dataset[i])
        data_points[i] = []
        data_points[i].append(time.time())
        print({"type":"request_sent","request_id":i,"time":time.time()}, flush=True)
    # drain all workers until all requests are finished
    while(True):
        all_done = True
        for i in range(TOTAL_GPUS):
            if(not worker_slots[i] is None):
                all_done = False
                break
        if(all_done):
            break
        time.sleep(1)
        
    all_first_time = []
    all_e2e_time = []
    all_token_time = []
    for data_point in data_points.values():
        # print data point to stderr
        print(data_point, "TTFT: ",data_point[1]-data_point[0], "ATT: ",data_point[2]-data_point[1], "E2E: ", data_point[2]-data_point[0], flush=True)
        all_first_time.append(data_point[1]-data_point[0])
        all_token_time.append(data_point[2]-data_point[1])
        all_e2e_time.append(data_point[2]-data_point[0])

    print("Average TTFT:",np.mean(all_first_time))
    print("Average ATT:",np.mean(all_token_time))
    print("Average E2E:",np.mean(all_e2e_time))
    print("Scale Number:",scale_num)
    print("Total requests:",len(data_points))
    print("All requests sent",flush=True)
    for p in worker_poll:
        p.p.kill()
    os.system("killall python3")
    exit(0)