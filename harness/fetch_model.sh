#!/usr/bin/env bash
# Usage: fetch_model.sh <hf_repo> <path_in_repo> [local_name]
# Downloads a file from Hugging Face into $MODELS_DIR (default ./models), resumable, and records sha256.
set -euo pipefail
repo="$1"; path="$2"; name="${3:-$(basename "$path")}"
dir="${MODELS_DIR:-$(cd "$(dirname "$0")/.." && pwd)/models}"
mkdir -p "$dir"
out="$dir/$name"
if [ ! -f "$out.sha256" ]; then
  for i in 1 2 3 4 5; do
    curl -fsSL -C - --retry 3 -o "$out" "https://huggingface.co/$repo/resolve/main/$path" && break
    sleep $((i*5))
  done
  sha256sum "$out" | awk '{print $1}' > "$out.sha256"
  echo "$repo/$path" > "$out.source"
fi
echo "$name $(cat "$out.sha256") $(stat -c %s "$out")"
