#!/usr/bin/env bash
# Do the winning configurations still win on the largest models that fit in 15.7 GiB?
# Candidates: Qwen3.5-9B Q4_K_M (dense) and Qwen3.6-35B-A3B (MoE, ~3B active) at ~2.7-3 bits.
set -euo pipefail
cd "$(dirname "$0")/.."
for m in Qwen3.5-9B-Q4_K_M.gguf Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf; do
  t=${m%.gguf}; t=${t#Qwen3.}
  harness/lbench.py --name big-$t-stock  --lever big --model $m --args "-t 4 -p 512 -n 64 -r 2" || true
  harness/lbench.py --name big-$t-ik     --lever big --engine ik_llama.cpp --build native --model $m --ofmt json --args "-t 4 -p 512 -n 64 -r 2 -rtr 0" || true
  harness/lbench.py --name big-$t-ik-rtr --lever big --engine ik_llama.cpp --build native --model $m --ofmt json --args "-t 4 -p 512 -n 64 -r 2 -rtr 1" || true
done
