#!/usr/bin/env bash
# Thinking on / capped vs off (off = qual-2b-q4km-uns from experiment 08): accuracy and time per answer.
set -uo pipefail
cd "$(dirname "$0")/.."
M=Qwen3.5-2B-Q4_K_M.unsloth.gguf
harness/quality.py --name think-on        --lever thinking --model $M --think on --max-tokens 2048
harness/quality.py --name think-budget256 --lever thinking --model $M --think on --max-tokens 2048 --server-args "-c 8192 -np 1 --reasoning-budget 256"
harness/quality.py --name think-budget64  --lever thinking --model $M --think on --max-tokens 2048 --server-args "-c 8192 -np 1 --reasoning-budget 64"
