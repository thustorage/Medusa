import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput

from pyinstrument import Profiler

import torch

import time

def create_test_prompts(args, batch_size, max_tokens) -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    if args.test_cuda_graph:
        return [
            ("To " * 161,
            SamplingParams(ignore_eos=True, max_tokens=max_tokens)),
        ] * batch_size
    return [
        ("To " * 67,
        SamplingParams(ignore_eos=True, max_tokens=max_tokens)),
        # ("To be or not to be,",
        #  SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
        # ("What is the meaning of life?",
        #  SamplingParams(n=2,
        #                 best_of=5,
        #                 temperature=0.8,
        #                 top_p=0.95,
        #                 frequency_penalty=0.1)),
        # ("It is only with the heart that one can see rightly",
        #  SamplingParams(n=3, best_of=3, use_beam_search=True,
        #                 temperature=0.0)),
    ] * batch_size
        


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()
        
        for request_output in request_outputs:
            if request_output.finished:
                print("\n")
                if not args.test_cuda_graph:
                    print(request_output)
                    print(request_output.outputs[0].text)


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    engine.init_phase2()
    
    if args.test_cuda_graph:
        if args.enforce_eager:
            filename = f"/home/zsx/vllm/experiments/cuda_graph/{args.model_name}_without_graph"
        else:
            filename = f"/home/zsx/vllm/experiments/cuda_graph/{args.model_name}"
        
        # warm up
        test_prompts = create_test_prompts(args, 1, 16)
        process_requests(engine, test_prompts)
        
        with open(filename, "w") as f:
            for batch_size in [1, 8]:
                for output_len in [338]:
                    start_time = time.perf_counter()
                    for _ in range(1):
                        test_prompts = create_test_prompts(args, batch_size, output_len)
                        process_requests(engine, test_prompts)
                    end_time = time.perf_counter()
                    f.write(f"batch size {batch_size} output len {output_len} time: {(end_time - start_time) * 1000} milliseconds\n")
                    f.flush()
    elif args.test_overall_performance:
        pass
    else:
        for _ in range(1):
            test_prompts = create_test_prompts(args, 1, 1)
            process_requests(engine, test_prompts)
            
    engine.stop()


if __name__ == '__main__':
    # comment when running traces
    profiler = Profiler()
    profiler.start()
    total_start_time = time.perf_counter()
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument('--test_cuda_graph', action='store_true', default=False)
    parser.add_argument('--test_overall_performance', type=str, choices=['sync', 'async_load', 'without_load_graph', 'with_load_graph'], default=None)
    args = parser.parse_args()
    
    if args.model_name == "Llama-13B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--openlm-research--open_llama_13b/snapshots/b6d7fde8392250730d24cc2fcfa3b7e5f9a03ce8/'
        # args.model = 'openlm-research/open_llama_13b' commit id: b6d7fde8392250730d24cc2fcfa3b7e5f9a03ce8#
    elif args.model_name == "Llama-7B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--openlm-research--open_llama_7b/snapshots/6fb184ff23774c25bf84b3628e49c8b78372c7be/'
        # args.model = 'openlm-research/open_llama_7b' commit id: 6fb184ff23774c25bf84b3628e49c8b78372c7be#
    elif args.model_name == "Yi-6B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--01-ai--Yi-6B/snapshots/b5b30297823fd4fbb8d76dc58ae44c93104faa2b/'
        # args.model = '01-ai/Yi-6B' commit id: b5b30297823fd4fbb8d76dc58ae44c93104faa2b#
    elif args.model_name == "Yi-9B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--01-ai--Yi-9B/snapshots/2af40f9fdd3ad4a56ddeb52a21e3f9c720d82353/'
        # args.model = '01-ai/Yi-9B' commit id: 2af40f9fdd3ad4a56ddeb52a21e3f9c720d82353#
    elif args.model_name == "Qwen-14B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--Qwen--Qwen1.5-14B/snapshots/dce4b190d34470818e5bec2a92cb8233aaa02ca2/'
        # args.model = 'Qwen/Qwen1.5-14B' commit id: dce4b190d34470818e5bec2a92cb8233aaa02ca2#
        args.max_model_len = 4096
    elif args.model_name == "Qwen-7B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--Qwen--Qwen2-beta-7B/snapshots/831096e3a59a0789a541415da25ef195ceb802fe/'
        # args.model = 'Qwen/Qwen1.5-7B' commit id: 831096e3a59a0789a541415da25ef195ceb802fe#
        args.max_model_len = 4096
    elif args.model_name == "Qwen-4B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--Qwen--Qwen1.5-4B/snapshots/a66363a0c24e2155c561e4b53c658b1d3965474e/'
        # args.model = 'Qwen/Qwen1.5-4B' commit id: a66363a0c24e2155c561e4b53c658b1d3965474e#
        args.max_model_len = 4096
    elif args.model_name == "Qwen-1.8B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--Qwen--Qwen1.5-1.8B/snapshots/7846de7ed421727b318d6605a0bfab659da2c067/'
        # args.model = 'Qwen/Qwen1.5-1.8B' commit id: 7846de7ed421727b318d6605a0bfab659da2c067#
        args.max_model_len = 4096
    elif args.model_name == "Qwen-0.5B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--Qwen--Qwen1.5-0.5B/snapshots/8f445e3628f3500ee69f24e1303c9f10f5342a39/'
        # args.model = 'Qwen/Qwen1.5-0.5B' commit id: 8f445e3628f3500ee69f24e1303c9f10f5342a39#
        args.max_model_len = 4096
    elif args.model_name == "Falcon-7B":
        args.model = '/home/zsx/raidfs-back/home/zsx/.cache/huggingface/hub/models--tiiuae--falcon-7b/snapshots/898df1396f35e447d5fe44e0a3ccaaaa69f30d36/'
        # args.model = 'tiiuae/falcon-7b' commit id: 898df1396f35e447d5fe44e0a3ccaaaa69f30d36#
    else:
        print("model_name not supported", args.model_name)
        exit(-1)
    
    main(args)
    total_end_time = time.perf_counter()
    if args.test_overall_performance:
        filename = f"/home/zsx/vllm/experiments/overall/{args.model_name}_{args.test_overall_performance}"
        with open(filename, "w") as f:
            f.write(f"Overall total time: {(total_end_time - total_start_time) * 1000} milliseconds")
            f.flush()
    profiler.stop()
    profiler.write_html(args.log, timeline=True, show_all=True)