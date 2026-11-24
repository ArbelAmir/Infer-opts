#!/usr/bin/env bash
# Rows of experiment 09 that failed to start (port still bound), plus the stock-engine combinations.
set -uo pipefail
cd "$(dirname "$0")/.."
M=Qwen3.5-2B-Q4_K_M.unsloth.gguf; C="-c 8192 -np 1"
B() { n=$1; shift; harness/bench.py --name "$n" --reps 3 "$@" || echo "FAILED $n"; }
B baseline-r7       --lever baseline --model $M
B q4_0-amx-clang    --lever combo --build clang --model Qwen3.5-2B-Q4_0.gguf --server-args "$C"
B q8-amx-clang      --lever combo --build clang --model Qwen3.5-2B-Q8_0.gguf --server-args "$C"
B q8-amx-r2         --lever quant --model Qwen3.5-2B-Q8_0.gguf --server-args "$C"
B ikj-q4km-rtr-r2   --lever engine-ik --engine ik_llama.cpp --build native --model $M --server-args "$C --jinja -rtr"
B ikj-q4_0-rtr-r2   --lever engine-ik --engine ik_llama.cpp --build native --model Qwen3.5-2B-Q4_0.gguf --server-args "$C --jinja -rtr"
B baseline-r8       --lever baseline --model $M
