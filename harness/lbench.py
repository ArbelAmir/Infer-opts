#!/usr/bin/env python3
"""Thin wrapper around llama-bench (synthetic prompt-processing / generation speed) that holds the
benchmark lock, waits for an idle machine and records provenance. One row per llama-bench test
is appended to results/llama_bench.jsonl. Use this for wide, cheap sweeps (quant x threads x flags);
use bench.py for workload-shaped measurements.

  harness/lbench.py --name q4km-threads --build default --model X.gguf --args "-t 1,2,3,4 -p 512 -n 128 -r 5"
"""
import argparse, json, os, shlex, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--lever", default="")
    p.add_argument("--engine", default="llama.cpp")
    p.add_argument("--build", default="default")
    p.add_argument("--model", required=True)
    p.add_argument("--args", default="-p 512 -n 128 -r 5")
    p.add_argument("--prefix", default="")
    p.add_argument("--ofmt", default="jsonl", help="llama-bench output format: jsonl (stock) or json (ik_llama.cpp)")
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--out", default=os.path.join(common.RESULTS, "llama_bench.jsonl"))
    a = p.parse_args()
    build_dir = os.path.join(common.ENGINES, a.engine, f"build-{a.build}")
    model = common.model_path(a.model)
    cmd = shlex.split(a.prefix) + [os.path.join(build_dir, "bin", "llama-bench"), "-m", model, "-o", a.ofmt] + shlex.split(a.args)
    env = dict(os.environ, **dict(kv.split("=", 1) for kv in a.env))
    log_dir = os.path.join(common.LOGS, "lbench")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, a.name + ".log")
    with common.bench_lock(a.name):
        busy = common.wait_idle()
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        dt = time.time() - t0
    with open(log_path, "w") as f:
        f.write("# " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        f.write(r.stdout)
        f.write("\n# ---- stderr (tail) ----\n" + r.stderr[-6000:])
    if r.returncode != 0:
        print(r.stderr[-2000:], file=sys.stderr)
        sys.exit(f"llama-bench failed ({r.returncode}); see {log_path}")
    binfo = common.build_info(build_dir)
    if a.ofmt == "json":
        txt = r.stdout[r.stdout.index("["):r.stdout.rindex("]") + 1]
        recs = json.loads(txt)
    else:
        recs = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
    for d in recs:
        row = {
            "ts": time.strftime("%FT%T"), "name": a.name, "lever": a.lever, "engine": a.engine, "build": a.build,
            "engine_commit": binfo.get("commit"), "build_cmake_args": binfo.get("cmake_args"), "cc": binfo.get("cc_version"),
            "model": os.path.basename(model), "model_sha256": common.sha256_of(model), "model_bytes": os.path.getsize(model),
            "cmd": " ".join(shlex.quote(c) for c in cmd), "env": dict(kv.split("=", 1) for kv in a.env),
            "kind": "pp" if d.get("n_gen", 0) == 0 else ("tg" if d.get("n_prompt", 0) == 0 else "pg"),
            "n_prompt": d.get("n_prompt"), "n_gen": d.get("n_gen"), "n_threads": d.get("n_threads"),
            "n_batch": d.get("n_batch"), "n_ubatch": d.get("n_ubatch"), "n_depth": d.get("n_depth"),
            "repack": d.get("repack", d.get("run_time_repack", d.get("rtr"))), "load_mode": d.get("load_mode"), "poll": d.get("poll"),
            "cpu_mask": d.get("cpu_mask"), "cpu_strict": d.get("cpu_strict"), "type_k": d.get("type_k"),
            "avg_ts": d.get("avg_ts"), "stddev_ts": d.get("stddev_ts"), "samples_ts": d.get("samples_ts"),
            "busy_before": round(busy, 3), "raw_log": os.path.relpath(log_path, common.ROOT),
        }
        common.append_jsonl(a.out, row)
        s = sorted(row["samples_ts"] or [0])
        print(f"{a.name:26s} {row['model'][:34]:34s} t={row['n_threads']} {row['kind']} n={row['n_prompt'] or row['n_gen']:5d} "
              f"median {s[len(s)//2]:8.2f} t/s  [{s[0]:.2f}..{s[-1]:.2f}]  rp={row['repack']} ub={row['n_ubatch']}", flush=True)
    print(f"# {a.name} took {dt:.0f}s")


if __name__ == "__main__":
    main()
