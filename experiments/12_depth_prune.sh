#!/usr/bin/env bash
# Depth pruning without fine-tuning (drop aligned groups of 4 blocks from the 24-block 2B model), judged
# against the alternatives that save the same time: the next smaller model and a lower-bit quant.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=engines/llama.cpp/gguf-py
S=Qwen3.5-2B-Q4_K_M.gguf
for d in 12-15 16-19 4-7 20-23 8-11,16-19; do
  t=Qwen3.5-2B-Q4_K_M.drop${d//,/_}.gguf
  harness/locked.sh python3 harness/prune_layers.py models/$S models/$t --drop $d || { echo "prune $d failed"; continue; }
  harness/kld.sh eval Qwen3.5-2B-BF16.gguf $t || echo "kld $d failed"
  harness/lbench.py --name prune-$d --lever prune --model $t --args "-p 512 -n 128 -r 3" || true
  rm -f models/$t
done
harness/kld.sh eval Qwen3.5-2B-BF16.gguf Qwen3.5-0.8B-Q4_K_M.gguf || true
harness/lbench.py --name prune-ref-08b --lever prune --model Qwen3.5-0.8B-Q4_K_M.gguf --args "-p 512 -n 128 -r 3"
harness/lbench.py --name prune-ref-2b --lever prune --model $S --args "-p 512 -n 128 -r 3"
