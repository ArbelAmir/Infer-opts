#!/usr/bin/env bash
# Quantize the small model from its BF16 GGUF into every format under test (static, no imatrix),
# so all formats come from one source with one tool version.
set -euo pipefail
cd "$(dirname "$0")/.."
Q=engines/llama.cpp/build-default/bin/llama-quantize
SRC=models/Qwen3.5-2B-BF16.gguf
for t in Q8_0 Q6_K Q5_K_M Q4_K_M Q4_K_S Q4_0 Q4_1 IQ4_XS IQ4_NL Q3_K_M Q2_K; do
  out=models/Qwen3.5-2B-$t.gguf
  [ -f "$out" ] && continue
  harness/locked.sh $Q "$SRC" "$out" "$t" 4 > logs/quantize-$t.log 2>&1
  sha256sum "$out" | awk '{print $1}' > "$out.sha256"
  echo "self-quantized from unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-BF16.gguf with llama-quantize $t (no imatrix)" > "$out.source"
  ls -la "$out" | awk '{print $5, $9}'
done
