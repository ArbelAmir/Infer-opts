#!/usr/bin/env bash
# MNN-LLM benchmark driver (each heavy step is run through harness/locked.sh by the python helper).
# Setup (once): harness/locked.sh bash harness/mnn_fetch.sh ; harness/locked.sh bash harness/mnn_build.sh ;
#   harness/locked.sh g++ -O2 -std=c++17 harness/mnn/mnn_workload.cpp -Iengines/MNN/transformers/llm/engine/include \
#     -Iengines/MNN/include -Lengines/MNN/build -lllm -lMNN -Wl,-rpath,$PWD/engines/MNN/build -o engines/MNN/build/mnn_workload
#   (plus visual.mnn + visual.mnn.weight from the HF repo: Omni loader requires them even for text-only use)
# Usage: harness/mnn_bench.sh [--only synth|workloads] [--tag r1] [--synth-configs a,b] [--wl-configs a,b]
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 harness/mnn/mnn_bench.py "$@"
