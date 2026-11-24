#!/usr/bin/env bash
# Resumable queue for the remaining measurements: each step leaves a stamp in logs/.done/ so that a
# container restart continues where it stopped (rerun this script; it is idempotent).
set -uo pipefail
cd "$(dirname "$0")/.."
S() { n=$1; shift; [ -f logs/.done/$n ] && return 0; echo "== $n $(date +%T)"; if "$@"; then touch logs/.done/$n; else echo "FAILED $n"; fi; }
M=Qwen3.5-2B-Q4_K_M.unsloth.gguf; C="-c 8192 -np 1"; ALL=chat_short,long_prompt,long_gen,code_edit,extract
B() { n=$1; shift; S $n harness/bench.py --name "$n" --reps 3 "$@"; }
IK="--lever engine-ik --engine ik_llama.cpp --build native"
OV=engines/ov-venv/bin/python
# 14: single-session comparison of the leading candidates, bracketed by baselines
B baseline-r7       --lever baseline --model $M
B q4_0-amx-clang    --lever combo --build clang --model Qwen3.5-2B-Q4_0.gguf --server-args "$C"
B q8-amx-clang      --lever combo --build clang --model Qwen3.5-2B-Q8_0.gguf --server-args "$C"
B q8-amx-r2         --lever quant --model Qwen3.5-2B-Q8_0.gguf --server-args "$C"
B ikj-q4km-rtr-r2   $IK --model $M --server-args "$C --jinja -rtr"
B ikj-q4_0-rtr-r2   $IK --model Qwen3.5-2B-Q4_0.gguf --server-args "$C --jinja -rtr"
# 13: other engines, same session
S mnn-compile harness/locked.sh g++ -O2 -std=c++17 harness/mnn/mnn_workload.cpp -Iengines/MNN/transformers/llm/engine/include \
  -Iengines/MNN/include -Lengines/MNN/build -lllm -lMNN -Wl,-rpath,$PWD/engines/MNN/build -o engines/MNN/build/mnn_workload
S ov-default-r2 $OV harness/ov_bench.py --name ov-default-r2 --workloads $ALL
S ov-f32-r2     $OV harness/ov_bench.py --name ov-f32-r2 --props '{"INFERENCE_PRECISION_HINT":"f32"}' --workloads $ALL
S mnn-r2        python3 harness/mnn/mnn_bench.py --only workloads --tag r2 --wl-configs t4-plow-mlow,t4-pnormal-mlow
B baseline-r8   --lever baseline --model $M
S qual-ov-f32   $OV harness/engine_quality.py --engine ov --name qual-ov-f32 --props '{"INFERENCE_PRECISION_HINT":"f32"}'
S qual-mnn      python3 harness/engine_quality.py --engine mnn --name qual-mnn-pnormal --mnn-precision normal
# 11: caching, prompt pruning, throughput
S cache   harness/cache_bench.py  --name cache-2b --model $M
S pprune  harness/prompt_prune.py --name pprune-2b --model $M --budget 0.25
for c in 1 2 4 8; do S tp-$c harness/throughput.py --name tp-c$c-np$c --model $M --concurrency $c --server-args "-c $((4096*c)) -np $c"; done
# 08: remaining accuracy guardrails
Q() { n=$1; m=$2; shift 2; S $n harness/quality.py --name "$n" --lever quality --model "$m" "$@"; }
Q qual-2b-q4_0      Qwen3.5-2B-Q4_0.gguf
Q qual-35b-q2kxl    Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf --engine ik_llama.cpp --build native --server-args "$C --jinja -rtr"
Q qual-9b-q4km      Qwen3.5-9B-Q4_K_M.gguf --engine ik_llama.cpp --build native --server-args "$C --jinja -rtr"
Q qual-08b-q4km     Qwen3.5-0.8B-Q4_K_M.gguf
Q qual-ik-q4km-rtr  $M --engine ik_llama.cpp --build native --server-args "$C --jinja -rtr"
# 12: depth pruning
S depth-prune experiments/12_depth_prune.sh
# 15: thinking
S think-on  harness/quality.py --name think-on --lever thinking --model $M --think on --max-tokens 2048
S think-256 harness/quality.py --name think-budget256 --lever thinking --model $M --think on --max-tokens 2048 --server-args "$C --reasoning-budget 256"
# large model at workload level (recommended engine settings)
B big-35b-q2kxl-ik-rtr $IK --model Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf --server-args "$C --jinja -rtr" --reps 2
Q qual-35b-iq3xxs   Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --engine ik_llama.cpp --build native --server-args "$C --jinja -rtr"
Q qual-2b-iq4xs     Qwen3.5-2B-IQ4_XS.gguf
B baseline-r9   --lever baseline --model $M
echo "== QUEUE DONE"
