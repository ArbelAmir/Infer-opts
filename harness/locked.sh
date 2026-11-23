#!/usr/bin/env bash
# Run any CPU-heavy command (build, quantize, perplexity, llama-bench ...) under the
# machine-wide benchmark lock so it can never overlap a measurement.
#   harness/locked.sh <command> [args...]
set -euo pipefail
LOCK="${BENCH_LOCK:-/tmp/infer-opts-bench.lock}"
if [ "${BENCH_LOCK_HELD:-0}" = "1" ]; then exec "$@"; fi
exec env BENCH_LOCK_HELD=1 flock -x "$LOCK" "$@"
