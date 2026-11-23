#!/usr/bin/env bash
# Threads / pinning / polling / batch size / load mode, on the baseline model with llama-bench.
set -euo pipefail
cd "$(dirname "$0")/.."
M=Qwen3.5-2B-Q4_K_M.unsloth.gguf
L() { n=$1; shift; harness/lbench.py --name "$n" --lever threads-memory --model $M --args "$*"; }
L thr-sweep        -t 1,2,3,4,6,8 -p 512 -n 128 -r 3
L thr-strict       -t 4 --cpu-mask 0xF --cpu-strict 1 -p 512 -n 128 -r 3
L thr-strict3      -t 3 --cpu-mask 0x7 --cpu-strict 1 -p 512 -n 128 -r 3
L thr-poll         -t 4 --poll 0,100 -p 512 -n 128 -r 3
L thr-prio         -t 4 --prio 2 -p 512 -n 128 -r 3
L ubatch           -t 4 -p 2048 -n 0 -ub 64,128,256,512,1024,2048 -b 2048 -r 3
L loadmode         -t 4 -lm mmap,mlock,none -p 512 -n 128 -r 3
L depth            -t 4 -d 0,2048,8192 -p 512 -n 64 -r 3
L kvq8             -t 4 -ctk q8_0 -ctv q8_0 -p 512 -n 128 -r 3
