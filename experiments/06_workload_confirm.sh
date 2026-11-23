#!/usr/bin/env bash
# Workload-level confirmation of the leading single-lever changes (each vs. the baseline, same session).
set -euo pipefail
cd "$(dirname "$0")/.."
M=Qwen3.5-2B-Q4_K_M.unsloth.gguf; C="-c 8192 -np 1"
B() { n=$1; shift; harness/bench.py --name "$n" --reps 3 "$@"; }
B baseline-r2      --lever baseline --model $M
B ctx8k            --lever context  --model $M --server-args "$C"
B ctx8k-clang      --lever build    --build clang --model $M --server-args "$C"
B ctx8k-norepack   --lever build-amx --model $M --server-args "$C --no-repack"
B q8-amx           --lever quant    --model Qwen3.5-2B-Q8_0.gguf --server-args "$C"
B q4_0-amx         --lever quant    --model Qwen3.5-2B-Q4_0.gguf --server-args "$C"
B ik-q4km          --lever engine-ik --engine ik_llama.cpp --build native --model $M --server-args "$C"
B ik-q4km-rtr      --lever engine-ik --engine ik_llama.cpp --build native --model $M --server-args "$C -rtr"
B ik-q4_0-rtr      --lever engine-ik --engine ik_llama.cpp --build native --model Qwen3.5-2B-Q4_0.gguf --server-args "$C -rtr"
B baseline-r3      --lever baseline --model $M
