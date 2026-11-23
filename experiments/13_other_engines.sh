#!/usr/bin/env bash
# Independent re-run of the OpenVINO GenAI and MNN numbers (all five workloads) plus the accuracy guardrail
# for their own 4-bit conversions, bracketed by a llama.cpp baseline rerun.
set -uo pipefail
cd "$(dirname "$0")/.."
ALL=chat_short,long_prompt,long_gen,code_edit,extract
harness/locked.sh g++ -O2 -std=c++17 harness/mnn/mnn_workload.cpp -Iengines/MNN/transformers/llm/engine/include \
  -Iengines/MNN/include -Lengines/MNN/build -lllm -lMNN -Wl,-rpath,$PWD/engines/MNN/build -o engines/MNN/build/mnn_workload
engines/ov-venv/bin/python harness/ov_bench.py --name ov-default-r2 --workloads $ALL
engines/ov-venv/bin/python harness/ov_bench.py --name ov-f32-r2 --props '{"INFERENCE_PRECISION_HINT":"f32"}' --workloads $ALL
python3 harness/mnn/mnn_bench.py --only workloads --tag r2 --wl-configs t4-plow-mlow,t4-pnormal-mlow
engines/ov-venv/bin/python harness/engine_quality.py --engine ov --name qual-ov-default
engines/ov-venv/bin/python harness/engine_quality.py --engine ov --name qual-ov-f32 --props '{"INFERENCE_PRECISION_HINT":"f32"}'
python3 harness/engine_quality.py --engine mnn --name qual-mnn-plow
python3 harness/engine_quality.py --engine mnn --name qual-mnn-pnormal --mnn-precision normal
harness/bench.py --name baseline-r6 --lever baseline --reps 3 --model Qwen3.5-2B-Q4_K_M.unsloth.gguf
