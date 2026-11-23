#!/usr/bin/env python3
"""MNN-LLM benchmark driver. Every heavy step runs via harness/locked.sh (one lock hold per step).
usage: mnn_bench.py [--only synth|workloads] [--tag name]
Appends one JSON line per (config, workload) to results/engines.jsonl; raw logs in logs/mnn/<name>/.
"""
import sys, os, re, json, subprocess, time, argparse, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "harness"))
from workloads import workloads
MNN = os.path.join(ROOT, "engines/MNN")
BIN = os.path.join(MNN, "build")
MODEL_DIR = os.path.join(ROOT, "models/mnn/Qwen3.5-2B-MNN")
CFG = os.path.join(MODEL_DIR, "config.json")
RESULTS = os.path.join(ROOT, "results/engines.jsonl")
LOCK = os.path.join(ROOT, "harness/locked.sh")
WRAP = os.path.join(ROOT, "harness/mnn/runwrap.py")
commit = subprocess.check_output(["git", "-C", MNN, "rev-parse", "HEAD"], text=True).strip()
MODEL = "taobao-mnn/Qwen3.5-2B-MNN (HQQ 4-bit g64)"

# (name, threads, precision(0 normal,1 high,2 low), memory(0 normal,2 low))
SYNTH = [("t4-plow-mlow", 4, 2, 2), ("t3-plow-mlow", 3, 2, 2), ("t2-plow-mlow", 2, 2, 2),
         ("t4-pnormal-mlow", 4, 0, 2), ("t4-phigh-mlow", 4, 1, 2), ("t4-plow-mnormal", 4, 2, 0)]
PREC = {0: "normal", 1: "high", 2: "low"}; MEM = {0: "normal", 2: "low"}
WL_CONFIGS = [("t4-plow-mlow", 4, "low", "low")]  # extended by --wl-configs
WL = ["chat_short", "long_prompt", "long_gen"]

def run_locked(log, cmd):
    out = subprocess.run([LOCK, "python3", WRAP, log] + cmd, capture_output=True, text=True).stdout
    m = re.search(r"WRAP wall_s=([\d.]+) maxrss_kb=(\d+) rc=(-?\d+)", out)
    return float(m[1]), int(m[2]), int(m[3])

def emit(row):
    row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); row["engine"] = "mnn"
    row["engine_version"] = commit; row["model"] = MODEL
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))

def rng(v): return [round(min(v), 3), round(max(v), 3)] if v else None

def synth(name, t, prec, mem, tag):
    d = os.path.join(ROOT, "logs/mnn", f"{tag}-{name}-synthetic"); os.makedirs(d, exist_ok=True)
    log = os.path.join(d, "llm_bench.log")
    cmd = [os.path.join(BIN, "llm_bench"), "-m", CFG, "-a", "cpu", "-t", str(t), "-c", str(prec),
           "--memory", str(mem), "-pg", "512,128", "-rep", "3"]
    wall, rss, rc = run_locked(log, cmd)
    txt = open(log).read()
    def row_of(test):
        m = re.search(r"\|\s*" + re.escape(test) + r"\s*\|\s*([\d.]+) ± ([\d.]+)(?:<br>([\d.]+) ± ([\d.]+))?", txt)
        return m
    pp = row_of("pp512"); tg = row_of("tg128"); pg = row_of("pp512+tg128")
    cfgd = dict(threads=t, precision=PREC[prec], memory=MEM[mem], mmap=0, backend="cpu", flash_attention=1,
                note="pp from pp512 row; gen from tg128 row (1-token ctx, like llama-bench tg128); pp512+tg128 row: pp=%s gen=%s" %
                (pg and pg[1], pg and pg[3]))
    emit(dict(name=f"mnn-{name}", workload="synthetic-512-128", cmd=" ".join(cmd), config=cfgd, prompt_n=512, predicted_n=128,
              pp_tps=float(pp[1]) if pp else None, gen_tps=float(tg[1]) if tg else None, ttft_s=512 / float(pp[1]) if pp else None,
              total_s=wall, pp_tps_range=[float(pp[1]) - float(pp[2]), float(pp[1]) + float(pp[2])] if pp else None,
              gen_tps_range=[float(tg[1]) - float(tg[2]), float(tg[1]) + float(tg[2])] if tg else None,
              load_s=None, peak_rss_kb=rss, raw_log=log, rc=rc, range_kind="mean +- stddev (llm_bench)"))

def real(name, t, prec, mem, tag):
    wls = workloads()
    for w in WL:
        d = os.path.join(ROOT, "logs/mnn", f"{tag}-{name}-{w}"); os.makedirs(d, exist_ok=True)
        msgs = wls[w]["messages"]; mx = wls[w]["max_tokens"]
        assert [m["role"] for m in msgs] == ["system", "user"]
        open(os.path.join(d, "system.txt"), "w").write(msgs[0]["content"])
        open(os.path.join(d, "user.txt"), "w").write(msgs[1]["content"])
        log = os.path.join(d, "run.log")
        extra = json.dumps({"thread_num": t, "precision": prec, "memory": mem})
        cmd = [os.path.join(BIN, "mnn_workload"), CFG, os.path.join(d, "system.txt"), os.path.join(d, "user.txt"),
               str(mx), "3", os.path.join(d, "out"), extra]
        wall, rss, rc = run_locked(log, cmd)
        rs = [dict(kv.split("=") for kv in l.split()[1:]) for l in open(log) if l.startswith("RESULT")]
        meas = [r for r in rs if r["warmup"] == "0"]
        if not meas:
            emit(dict(name=f"mnn-{name}", workload=w, cmd=" ".join(cmd), config={}, rc=rc, error="no results", raw_log=log)); continue
        f = lambda k: [float(r[k]) for r in meas]
        emit(dict(name=f"mnn-{name}", workload=w, cmd=" ".join(cmd),
                  config=dict(threads=t, precision=prec, memory=mem, sampler="greedy", enable_thinking=False, warmup_runs=1, repeats=len(meas),
                              note="gen_tps=(gen_n-1)/decode_s (first token is produced in prefill); ttft=prefill_s"),
                  prompt_n=int(meas[0]["prompt_n"]), predicted_n=int(meas[0]["gen_n"]),
                  pp_tps=statistics.mean(f("pp_tps")), gen_tps=statistics.mean(f("gen_tps")), ttft_s=statistics.mean(f("prefill_s")),
                  total_s=statistics.mean(f("total_s")), pp_tps_range=rng(f("pp_tps")), gen_tps_range=rng(f("gen_tps")),
                  load_s=float(meas[0]["load_s"]), peak_rss_kb=rss, raw_log=log, rc=rc, process_wall_s=wall,
                  output_file=os.path.join(d, "out.run1.txt")))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["synth", "workloads"]); ap.add_argument("--tag", default="r1")
    ap.add_argument("--synth-configs", default=",".join(c[0] for c in SYNTH))
    ap.add_argument("--wl-configs", default="t4-plow-mlow")
    a = ap.parse_args()
    if a.only != "workloads":
        for c in SYNTH:
            if c[0] in a.synth_configs.split(","): synth(*c, a.tag)
    if a.only != "synth":
        for c in SYNTH:
            if c[0] in a.wl_configs.split(","): real(c[0], c[1], PREC[c[2]], MEM[c[3]], a.tag)
