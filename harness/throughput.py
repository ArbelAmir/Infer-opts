#!/usr/bin/env python3
"""Throughput mode: N concurrent requests against a server with N slots (continuous batching).

Sends `--concurrency` identical-shape requests at once (distinct prompts so nothing is shared),
repeats the wave `--reps` times after a warm-up wave, and appends one row to results/throughput.jsonl:
aggregate generated tokens/s over the wave's wall time, and per-request latency.

  harness/throughput.py --name tp-np4 --model X.gguf --concurrency 4 --server-args "-c 16384 -np 4"
"""
import argparse, json, os, shlex, statistics, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from bench import Server, chat_stream

TOPICS = ["the printing press", "the steam engine", "the telescope", "vaccination", "the electric telegraph",
          "the bicycle", "the compass", "photography"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--engine", default="llama.cpp")
    p.add_argument("--build", default="default")
    p.add_argument("--model", required=True)
    p.add_argument("--server-args", default="")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--out", default=os.path.join(common.RESULTS, "throughput.jsonl"))
    a = p.parse_args()
    build_dir = os.path.join(common.ENGINES, a.engine, f"build-{a.build}")
    model = common.model_path(a.model)
    log_dir = os.path.join(common.LOGS, a.name)
    os.makedirs(log_dir, exist_ok=True)

    def body(i):
        return {"messages": [{"role": "user", "content": f"Write a detailed essay of about 400 words on the history and impact of {TOPICS[i % len(TOPICS)]}."}],
                "max_tokens": a.max_tokens, "stream": True, "temperature": 0, "seed": 42, "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False}}

    waves = []
    with common.bench_lock(a.name):
        common.wait_idle()
        srv = Server(os.path.join(build_dir, "bin", "llama-server"), model, a.server_args, a.port, os.path.join(log_dir, "server.log"))
        srv.start()
        try:
            for rep in range(a.reps + 1):
                res = [None] * a.concurrency

                def work(i):
                    res[i] = chat_stream(a.port, body(i))

                t0 = time.perf_counter()
                th = [threading.Thread(target=work, args=(i,)) for i in range(a.concurrency)]
                [t.start() for t in th]
                [t.join() for t in th]
                wall = time.perf_counter() - t0
                toks = sum(r["timings"].get("predicted_n", 0) for r in res)
                w = {"wall_s": wall, "tokens": toks, "agg_tps": toks / wall,
                     "latency_s": [r["total_s"] for r in res], "ttft_s": [r["ttft_s"] for r in res],
                     "per_req_gen_tps": [r["timings"].get("predicted_per_second") for r in res]}
                print(f"  {a.name} wave{rep} c={a.concurrency} agg {w['agg_tps']:.2f} t/s  latency {statistics.median(w['latency_s']):.1f}s"
                      f"{'  (warmup)' if rep == 0 else ''}", flush=True)
                if rep:
                    waves.append(w)
        finally:
            mem = srv.stop()
    json.dump(waves, open(os.path.join(log_dir, "waves.json"), "w"), indent=1)
    binfo = common.build_info(build_dir)
    row = {"ts": time.strftime("%FT%T"), "name": a.name, "engine": a.engine, "build": a.build, "engine_commit": binfo.get("commit"),
           "model": os.path.basename(model), "model_sha256": common.sha256_of(model),
           "cmd": " ".join(shlex.quote(c) for c in srv.cmd), "concurrency": a.concurrency, "max_tokens": a.max_tokens,
           "agg_tps": statistics.median(w["agg_tps"] for w in waves), "agg_tps_range": [min(w["agg_tps"] for w in waves), max(w["agg_tps"] for w in waves)],
           "latency_s": statistics.median(l for w in waves for l in w["latency_s"]),
           "ttft_s": statistics.median(t for w in waves for t in w["ttft_s"]),
           "per_req_gen_tps": statistics.median(t for w in waves for t in w["per_req_gen_tps"]),
           "peak_rss_kb": mem.get("VmHWM"), "raw_log": os.path.relpath(os.path.join(log_dir, "waves.json"), common.ROOT)}
    common.append_jsonl(a.out, row)
    print(f"{a.name}: c={a.concurrency} aggregate {row['agg_tps']:.2f} t/s, per-request {row['per_req_gen_tps']:.2f} t/s, latency {row['latency_s']:.1f}s, rss {row['peak_rss_kb']/1048576:.2f}G")


if __name__ == "__main__":
    main()
