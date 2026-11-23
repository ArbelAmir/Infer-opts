#!/usr/bin/env bash
# Baseline + build lever. Baseline = stock llama.cpp, default cmake, unsloth Q4_K_M, default runtime flags.
set -euo pipefail
cd "$(dirname "$0")/.."
M=Qwen3.5-2B-Q4_K_M.unsloth.gguf
harness/bench.py --name baseline --lever baseline --build default --model $M --reps 5
for b in default nonative clang lto noomp; do
  harness/lbench.py --name build-$b --lever build --build $b --model $M --args "-p 512 -n 128 -r 5 --repack 1"
done
# AMX/repack on vs off in the default build (AMX buffer is only used when repack is on)
harness/lbench.py --name build-default-norepack --lever build-amx --build default --model $M --args "-p 512 -n 128 -r 5 --repack 0"
