#!/usr/bin/env bash
# ik_llama.cpp (performance fork) on the same GGUF files, with and without run-time repack (-rtr).
set -euo pipefail
cd "$(dirname "$0")/.."
for m in Qwen3.5-2B-Q4_K_M.unsloth.gguf Qwen3.5-2B-Q4_0.gguf Qwen3.5-2B-Q8_0.gguf Qwen3.5-2B-IQ4_XS.gguf Qwen3.5-2B-Q6_K.gguf; do
  q=${m#Qwen3.5-2B-}; q=${q%.gguf}
  harness/lbench.py --name ik-$q      --lever engine-ik --engine ik_llama.cpp --build native --model $m --ofmt json --args "-t 4 -p 512 -n 128 -r 3 -rtr 0" || true
  harness/lbench.py --name ik-$q-rtr  --lever engine-ik --engine ik_llama.cpp --build native --model $m --ofmt json --args "-t 4 -p 512 -n 128 -r 3 -rtr 1" || true
done
