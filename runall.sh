mkdir -p experiments/overall
python scripts/overall.py > results/Figure7
python scripts/breakdown.py > results/Figure2
mkdir -p experiments/cuda_graph
python scripts/cuda_graph.py > results/Figure3
python scripts/calculations.py > results/Table1
mkdir -p experiments/offline
python scripts/offline.py > results/Figure9
mkdir -p experiments/traces
mkdir -p experiments/traces/qps2
mkdir -p experiments/traces/qps10
python scripts/traces.py > results/Figure10
mkdir -p experiments/traces_throughput
python scripts/traces_throughput.py > results/Figure11
python scripts/breakdown_Qwen.py > results/Figure8_Figure1
