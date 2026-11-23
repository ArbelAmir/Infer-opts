#!/usr/bin/env bash
# Quantization lever on the small model: speed (llama-bench, AMX repack on and off) and quality (KLD vs BF16).
set -euo pipefail
cd "$(dirname "$0")/.."
QS="BF16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_K_S Q4_0 Q4_1 IQ4_XS IQ4_NL Q3_K_M Q2_K"
for q in $QS; do
  harness/lbench.py --name quant-$q --lever quant --model Qwen3.5-2B-$q.gguf --args "-p 512 -n 128 -r 3 --repack 1"
  harness/lbench.py --name quant-$q-norepack --lever quant-norepack --model Qwen3.5-2B-$q.gguf --args "-p 512 -n 128 -r 3 --repack 0"
done
harness/kld.sh base Qwen3.5-2B-BF16.gguf
for q in Q8_0 Q6_K Q5_K_M Q4_K_M Q4_K_S Q4_0 Q4_1 IQ4_XS IQ4_NL Q3_K_M Q2_K; do
  harness/kld.sh eval Qwen3.5-2B-BF16.gguf Qwen3.5-2B-$q.gguf
done
harness/kld.sh eval Qwen3.5-2B-BF16.gguf Qwen3.5-2B-Q4_K_M.unsloth.gguf
