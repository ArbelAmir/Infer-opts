#!/usr/bin/env bash
# Perplexity + KL divergence of a model against reference logits (wikitext-2 test, 512-token chunks).
#   kld.sh base  <reference.gguf>                 # writes data/kld-base-<ref>.bin (needs ~0.5 MB per token)
#   kld.sh eval  <reference.gguf> <model.gguf>    # appends a row to results/kld.jsonl
# Env: CHUNKS (default 20), BUILD (default default), ENGINE (default llama.cpp), EXTRA (extra perplexity args)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; root="$(dirname "$here")"; cd "$root"
mode="$1"; ref="$2"; CHUNKS="${CHUNKS:-20}"; BUILD="${BUILD:-default}"; ENGINE="${ENGINE:-llama.cpp}"
P="engines/$ENGINE/build-$BUILD/bin/llama-perplexity"
base="data/kld-base-$(basename "$ref" .gguf)-c$CHUNKS.bin"
mkdir -p logs/kld results
if [ "$mode" = base ]; then
  [ -f "$base" ] && { echo "exists: $base"; exit 0; }
  harness/locked.sh "$P" -m "models/$ref" -f data/wikitext-2-raw/wiki.test.raw -c 512 --chunks "$CHUNKS" \
      --kl-divergence-base "$base" > "logs/kld/base-$(basename "$ref" .gguf).log" 2>&1
  grep -E 'Final estimate' "logs/kld/base-$(basename "$ref" .gguf).log" || true
  exit 0
fi
model="$3"; tag="${TAG:-$(basename "$model" .gguf)}"; log="logs/kld/$tag.log"
harness/locked.sh "$P" -m "models/$model" -f data/wikitext-2-raw/wiki.test.raw -c 512 --chunks "$CHUNKS" \
    --kl-divergence-base "$base" --kl-divergence ${EXTRA:-} > "$log" 2>&1
python3 - "$log" "$model" "$ref" "$tag" "$ENGINE" "$BUILD" <<'EOF'
import json, os, re, sys, time
log, model, ref, tag, engine, build = sys.argv[1:]
t = open(log).read()
def g(pat):
    m = re.search(pat, t)
    return float(m.group(1)) if m else None
row = {"ts": time.strftime("%FT%T"), "name": tag, "model": model, "reference": ref, "engine": engine, "build": build,
       "model_sha256": open(f"models/{model}.sha256").read().strip() if os.path.exists(f"models/{model}.sha256") else None,
       "ppl": g(r"Mean PPL\(Q\)\s*:\s*([\d.]+)"), "ppl_ref": g(r"Mean PPL\(base\)\s*:\s*([\d.]+)"),
       "kld_mean": g(r"Mean\s+KLD:\s*([-\d.eE]+)"), "kld_p99": g(r"99\.0%\s+KLD:\s*([-\d.eE]+)"),
       "kld_median": g(r"Median\s+KLD:\s*([-\d.eE]+)"), "same_top_pct": g(r"Same top p:\s*([\d.]+)"),
       "raw_log": log}
open("results/kld.jsonl", "a").write(json.dumps(row, sort_keys=True) + "\n")
print(f"{tag:34s} ppl {row['ppl']}  (ref {row['ppl_ref']})  KLD mean {row['kld_mean']}  p99 {row['kld_p99']}  same-top {row['same_top_pct']}%")
EOF
