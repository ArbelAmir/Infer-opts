#!/usr/bin/env bash
# Decoding-level speedups on the small model, by workload. Every config uses a fresh server per
# measured request (--isolate) so n-gram pools cannot feed on a previous identical run.
set -euo pipefail
cd "$(dirname "$0")/.."
M=Qwen3.5-2B-Q4_K_M.unsloth.gguf; MTP=Qwen3.5-2B-MTP-Q4_K_M.gguf; D=models/Qwen3.5-0.8B-Q4_K_M.gguf
C="-c 8192 -np 1"
R() { n=$1; m=$2; shift 2; harness/bench.py --name "$n" --lever decoding --model "$m" --isolate --reps 2 --server-args "$C $*"; }
R dec-ref          $M
R dec-draft-n4     $M  -md $D --spec-type draft-simple --spec-draft-n-max 4
R dec-draft-n8     $M  -md $D --spec-type draft-simple --spec-draft-n-max 8
R dec-draft-n16p   $M  -md $D --spec-type draft-simple --spec-draft-n-max 16 --spec-draft-p-min 0.75
R dec-ngram-mod    $M  --spec-type ngram-mod
R dec-ngram-simple $M  --spec-type ngram-simple --spec-draft-n-max 64
R dec-ngram-mapk   $M  --spec-type ngram-map-k --spec-draft-n-max 64
R dec-mtp-ref      $MTP
R dec-mtp-n1       $MTP --spec-type draft-mtp --spec-draft-n-max 1
R dec-mtp-n2       $MTP --spec-type draft-mtp --spec-draft-n-max 2
R dec-mtp-n3       $MTP --spec-type draft-mtp --spec-draft-n-max 3
