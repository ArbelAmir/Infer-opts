#!/usr/bin/env bash
# Large-model runs on the memory-mapped path (ik -rtr on the 12.3 GB file was OOM-killed mid-run), resumable.
set -uo pipefail
cd "$(dirname "$0")/.."
S() { n=$1; shift; [ -f logs/.done/$n ] && return 0; echo "== $n $(date +%T)"; if "$@"; then touch logs/.done/$n; else echo "FAILED $n"; fi; }
C="-c 8192 -np 1"; IK="--engine ik_llama.cpp --build native"; G=Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
S qual-35b-q2kxl-mmap harness/quality.py --name qual-35b-q2kxl-mmap --lever quality --model $G $IK --server-args "$C --jinja"
S depth-prune2 experiments/12b_depth_prune.sh
S big-35b-ik      harness/bench.py --name big-35b-q2kxl-ik --lever big-model $IK --model $G --server-args "$C --jinja" --reps 2
S big-35b-stock   harness/bench.py --name big-35b-q2kxl-stock --lever big-model --model $G --server-args "$C" --reps 2
S big-9b-ik-rtr   harness/bench.py --name big-9b-q4km-ik-rtr --lever big-model $IK --model Qwen3.5-9B-Q4_K_M.gguf --server-args "$C --jinja -rtr" --reps 2
S baseline-r10    harness/bench.py --name baseline-r10 --lever baseline --reps 3 --model Qwen3.5-2B-Q4_K_M.unsloth.gguf
echo "== QUEUE2 DONE"
