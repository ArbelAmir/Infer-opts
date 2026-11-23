#!/usr/bin/env bash
# Remainder of 08_quality.sh (the container restarted after the first three configurations).
set -uo pipefail
cd "$(dirname "$0")/.."
Q() { n=$1; m=$2; shift 2; harness/quality.py --name "$n" --lever quality --model "$m" "$@" || true; }
Q qual-2b-q4km      Qwen3.5-2B-Q4_K_M.gguf
Q qual-2b-q4_0      Qwen3.5-2B-Q4_0.gguf
Q qual-2b-q4_1      Qwen3.5-2B-Q4_1.gguf
Q qual-2b-iq4xs     Qwen3.5-2B-IQ4_XS.gguf
Q qual-08b-q4km     Qwen3.5-0.8B-Q4_K_M.gguf
Q qual-9b-q4km      Qwen3.5-9B-Q4_K_M.gguf
Q qual-35b-q2kxl    Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
Q qual-35b-iq3xxs   Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
