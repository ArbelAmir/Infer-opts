#!/usr/bin/env bash
# One command to reproduce the core findings on a fresh machine (Linux x86-64, cmake, a C++ compiler,
# python3, curl, unzip, flock; needs outbound access to github.com and huggingface.co + its CDN).
#
#   ./run_all.sh            # engines + models + baseline + every single-lever config + recommended config
#   QUICK=1 ./run_all.sh    # baseline and recommended configuration only
#
# Everything is appended to results/*.jsonl with full provenance; print tables with harness/summarize.py.
# Engine commits are pinned to the ones measured in REPORT.md; set LLAMA_COMMIT / IK_COMMIT=master to move.
set -euo pipefail
cd "$(dirname "$0")"
LLAMA_COMMIT="${LLAMA_COMMIT:-239242322dc24e8c15a458b1beae5064e3358cf8}"
IK_COMMIT="${IK_COMMIT:-a28f79a5ac4b8db54eddf3907dc4e607ccca460c}"
TAG="${TAG:-$(hostname -s)}"         # suffix for config names so several machines can share one results file
M=Qwen3.5-2B-Q4_K_M.unsloth.gguf
mkdir -p engines models data results logs

src() {  # src <dir> <url> <commit>
  [ -d "engines/$1/.git" ] || git clone -q "$2" "engines/$1"
  git -C "engines/$1" checkout -q "$3" 2>/dev/null || { git -C "engines/$1" fetch -q origin; git -C "engines/$1" checkout -q "$3"; }
}
build() {  # build <dir> <variant> [cmake args]; skipped when already built
  [ -x "engines/$1/build-$2/bin/llama-server" ] || TARGETS="llama-bench llama-server llama-perplexity llama-quantize" harness/build_llamacpp.sh "$@"
}

echo "== engines"
src llama.cpp    https://github.com/ggml-org/llama.cpp    "$LLAMA_COMMIT"
src ik_llama.cpp https://github.com/ikawrakow/ik_llama.cpp "$IK_COMMIT"
build llama.cpp default                       # the baseline: stock build, no options
build ik_llama.cpp native -DGGML_NATIVE=ON

echo "== models + data"
harness/fetch_model.sh unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-Q4_K_M.gguf $M
if [ ! -f data/wikitext-2-raw/wiki.valid.raw ]; then
  curl -fsSL -o data/wikitext-2-raw-v1.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
  unzip -qo data/wikitext-2-raw-v1.zip -d data
fi

B() { n=$1; shift; harness/bench.py --name "$n-$TAG" --reps 3 "$@"; }
IK="--engine ik_llama.cpp --build native"
C="-c 8192 -np 1"

echo "== baseline and recommended configuration"
B baseline     --lever baseline --model $M
B recommended  --lever recommended $IK --model $M --server-args "$C --jinja -rtr"
B recommended-ngram --lever recommended $IK --model $M --server-args "$C --jinja -rtr --spec-type ngram-simple" \
               --isolate --reps 2 --workloads code_edit,extract

if [ -z "${QUICK:-}" ]; then
  echo "== single levers (each against the baseline)"
  harness/fetch_model.sh unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-BF16.gguf
  for q in Q8_0 Q4_0; do
    [ -f models/Qwen3.5-2B-$q.gguf ] || harness/locked.sh engines/llama.cpp/build-default/bin/llama-quantize \
        models/Qwen3.5-2B-BF16.gguf models/Qwen3.5-2B-$q.gguf $q > logs/quantize-$q.log 2>&1
  done
  build llama.cpp nonative -DGGML_NATIVE=OFF
  if command -v clang >/dev/null; then CC=clang CXX=clang++ build llama.cpp clang -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++; fi
  B ctx8k        --lever context --model $M --server-args "$C"
  B nonative     --lever build --build nonative --model $M --server-args "$C"
  [ -d engines/llama.cpp/build-clang ] && B clang --lever build --build clang --model $M --server-args "$C"
  B norepack     --lever build-amx --model $M --server-args "$C --no-repack"
  B q8           --lever quant --model Qwen3.5-2B-Q8_0.gguf --server-args "$C"
  B q4_0         --lever quant --model Qwen3.5-2B-Q4_0.gguf --server-args "$C"
  B ik           --lever engine-ik $IK --model $M --server-args "$C --jinja"
  B ngram        --lever decoding --model $M --server-args "$C --spec-type ngram-simple" --isolate --reps 2
  for t in 1 2 $(nproc); do
    harness/lbench.py --name threads-$t-$TAG --lever threads --model $M --args "-t $t -p 512 -n 128 -r 3"
  done
  # quality guardrails: KL divergence against the BF16 reference, and the 50 checkable task items
  harness/kld.sh base Qwen3.5-2B-BF16.gguf
  for f in $M Qwen3.5-2B-Q8_0.gguf Qwen3.5-2B-Q4_0.gguf; do harness/kld.sh eval Qwen3.5-2B-BF16.gguf $f; done
  harness/quality.py --name qual-baseline-$TAG --model $M
  harness/quality.py --name qual-recommended-$TAG $IK --model $M --server-args "$C --jinja -rtr"
  B baseline-end --lever baseline --model $M      # drift check: should match the first baseline
fi

echo "== summary (ratios against baseline-$TAG)"
python3 harness/summarize.py --baseline "baseline-$TAG" | grep -E "config|pp t/s|$TAG" || true
