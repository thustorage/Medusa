python scripts/serverless_llm_debug.py --save_tensor
# python scripts/serverless_llm_debug.py --offline
while true; do
    python scripts/serverless_llm_debug.py --online
    tail ./data/Llama-13B/log_loadgraph_0 -n 1
done
