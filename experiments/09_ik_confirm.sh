#!/usr/bin/env bash
# ik_llama.cpp at workload level, this time with --jinja so the request's enable_thinking=false is honoured
# (the ik-* rows of experiment 06 ran with thinking ON: valid tok/s, but totals not comparable).
set -euo pipefail
cd "$(dirname "$0")/.."
M=Qwen3.5-2B-Q4_K_M.unsloth.gguf; C="-c 8192 -np 1 --jinja"
B() { n=$1; shift; harness/bench.py --name "$n" --reps 3 "$@" || echo "FAILED $n"; }
I() { n=$1; shift; B "$n" --lever engine-ik --engine ik_llama.cpp --build native "$@"; }
I ikj-q4km          --model $M --server-args "$C"
I ikj-q4km-rtr      --model $M --server-args "$C -rtr"
I ikj-q4km-rtr-thp  --model $M --server-args "$C -rtr -thp"
I ikj-q4_0-rtr      --model Qwen3.5-2B-Q4_0.gguf --server-args "$C -rtr"
I ikj-iq4xs-rtr     --model Qwen3.5-2B-IQ4_XS.gguf --server-args "$C -rtr"
B baseline-r4       --lever baseline --model $M
# combinations on stock llama.cpp
B q4_0-amx-clang    --lever combo --build clang --model Qwen3.5-2B-Q4_0.gguf --server-args "-c 8192 -np 1"
B q8-amx-clang      --lever combo --build clang --model Qwen3.5-2B-Q8_0.gguf --server-args "-c 8192 -np 1"
# n-gram / MTP drafting on top of ik -rtr (stateful -> fresh server per request)
I ikj-rtr-iso       --isolate --reps 2 --model $M --server-args "$C -rtr"
I ikj-rtr-ngram     --isolate --reps 2 --model $M --server-args "$C -rtr --spec-type ngram-simple"
I ikj-rtr-ngram-mod --isolate --reps 2 --model $M --server-args "$C -rtr --spec-type ngram-mod"
I ikj-rtr-mtp       --isolate --reps 2 --model Qwen3.5-2B-MTP-Q4_K_M.gguf --server-args "$C -rtr --spec-type mtp:n_max=2"
B baseline-r5       --lever baseline --model $M
