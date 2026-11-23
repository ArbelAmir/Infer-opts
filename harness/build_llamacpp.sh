#!/usr/bin/env bash
# Build a llama.cpp-family engine variant under the benchmark lock and record provenance.
#   build_llamacpp.sh <src_dir_name_under_engines> <variant_name> [extra cmake args...]
# Env: CC, CXX (optional compiler override). Output: engines/<src>/build-<variant>/ + BUILD_INFO.json
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; root="$(dirname "$here")"
src="${ENGINES_DIR:-$root/engines}/$1"; variant="$2"; shift 2
bdir="$src/build-$variant"
targets="${TARGETS:-llama-cli llama-bench llama-server llama-perplexity llama-quantize llama-imatrix}"
t0=$(date +%s)
"$here/locked.sh" bash -c "
  set -e
  cmake -S '$src' -B '$bdir' -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF $* > '$bdir.cmake.log' 2>&1
  cmake --build '$bdir' -j\$(nproc) --target $targets > '$bdir.make.log' 2>&1
"
t1=$(date +%s)
python3 - "$src" "$bdir" "$variant" "$((t1-t0))" "$@" <<'EOF'
import json, os, subprocess, sys
src, bdir, variant, secs, *args = sys.argv[1:]
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True, cwd=src).stdout.strip()
cache = {}
for line in open(os.path.join(bdir, "CMakeCache.txt")):
    if "=" in line and not line.startswith(("#", "//")):
        k, v = line.rstrip("\n").split("=", 1)
        k = k.split(":")[0]
        if k.startswith(("GGML_", "LLAMA_", "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_BUILD_TYPE",
                         "CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS", "CMAKE_INTERPROCEDURAL", "BLA_")) and v != "":
            cache[k] = v
info = {
    "engine": os.path.basename(src), "variant": variant,
    "commit": sh("git rev-parse HEAD"), "commit_date": sh("git log -1 --format=%cI"),
    "remote": sh("git remote get-url origin"),
    "cmake_args": args, "cmake_cache": cache, "build_seconds": int(secs),
    "cc_version": sh(f"{cache.get('CMAKE_C_COMPILER','cc')} --version | head -1"),
}
json.dump(info, open(os.path.join(bdir, "BUILD_INFO.json"), "w"), indent=1, sort_keys=True)
print(json.dumps({k: info[k] for k in ("engine", "variant", "commit", "cmake_args", "build_seconds")}))
EOF
