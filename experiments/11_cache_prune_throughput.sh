#!/usr/bin/env bash
# Not generating / not processing unneeded tokens, and throughput mode (stock llama.cpp, 2B Q4_K_M).
set -uo pipefail
cd "$(dirname "$0")/.."
M=Qwen3.5-2B-Q4_K_M.unsloth.gguf
harness/cache_bench.py  --name cache-2b --model $M
harness/prompt_prune.py --name pprune-2b --model $M --budget 0.25
harness/prompt_prune.py --name pprune-2b-b10 --model $M --budget 0.10
harness/throughput.py --name tp-c1-np1 --model $M --concurrency 1 --server-args "-c 4096 -np 1"
harness/throughput.py --name tp-c2-np2 --model $M --concurrency 2 --server-args "-c 8192 -np 2"
harness/throughput.py --name tp-c4-np4 --model $M --concurrency 4 --server-args "-c 16384 -np 4"
harness/throughput.py --name tp-c8-np8 --model $M --concurrency 8 --server-args "-c 32768 -np 8"
harness/throughput.py --name tp-c4-np4-q4_0 --model Qwen3.5-2B-Q4_0.gguf --concurrency 4 --server-args "-c 16384 -np 4"
