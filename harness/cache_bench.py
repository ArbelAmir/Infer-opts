#!/usr/bin/env python3
"""Prompt/KV caching for repeated prefixes: the same ~3.3k-token context followed by different questions
(a RAG/agent-loop shape). Measures time-to-first-token and total time of each follow-up question with
cache_prompt on vs off, on one server. Appends rows to results/cache.jsonl.

  harness/cache_bench.py --name cache-2b --model X.gguf --server-args "-c 8192 -np 1"
"""
import argparse, json, os, shlex, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from bench import Server, chat_stream
from workloads import long_context, SYSTEM

QUESTIONS = [
    "What colour are European lobsters in life? One short sentence.",
    "Between which two places does M-82 travel? One short sentence.",
    "How long can Homarus gammarus grow, in centimetres? One short sentence.",
    "Which national forest does M-82 run along? One short sentence.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--engine", default="llama.cpp")
    p.add_argument("--build", default="default")
    p.add_argument("--model", required=True)
    p.add_argument("--server-args", default="-c 8192 -np 1")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--out", default=os.path.join(common.RESULTS, "cache.jsonl"))
    a = p.parse_args()
    build_dir = os.path.join(common.ENGINES, a.engine, f"build-{a.build}")
    model = common.model_path(a.model)
    log_dir = os.path.join(common.LOGS, a.name)
    os.makedirs(log_dir, exist_ok=True)
    ctx = long_context()

    def body(q, cache):
        # context first, question last: the shared prefix is everything up to the question
        return {"messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": "Read the following two articles and then answer the question.\n\n" + ctx + "\n\nQuestion: " + q}],
                "max_tokens": 48, "stream": True, "temperature": 0, "seed": 42, "cache_prompt": cache,
                "chat_template_kwargs": {"enable_thinking": False}}

    out = {}
    with common.bench_lock(a.name):
        common.wait_idle()
        srv = Server(os.path.join(build_dir, "bin", "llama-server"), model, a.server_args, a.port, os.path.join(log_dir, "server.log"))
        srv.start()
        try:
            chat_stream(a.port, body(QUESTIONS[0], False))  # warm-up, nothing cached
            for mode, cache in (("cache_off", False), ("cache_on", True)):
                if cache:
                    chat_stream(a.port, body(QUESTIONS[0], True))  # prime the cache with the shared prefix
                runs = []
                for q in QUESTIONS[1:]:
                    r = chat_stream(a.port, body(q, cache))
                    t = r["timings"]
                    runs.append({"q": q, "ttft_s": r["ttft_s"], "total_s": r["total_s"], "prompt_n": t.get("prompt_n"),
                                 "cache_n": t.get("cache_n"), "answer": r["content"][:200]})
                    print(f"  {a.name} {mode:9s} ttft {r['ttft_s']:6.2f}s total {r['total_s']:6.2f}s processed {t.get('prompt_n')} cached {t.get('cache_n')}", flush=True)
                out[mode] = runs
        finally:
            srv.stop()
    json.dump(out, open(os.path.join(log_dir, "cache.json"), "w"), indent=1)
    binfo = common.build_info(build_dir)
    for mode, runs in out.items():
        common.append_jsonl(a.out, {
            "ts": time.strftime("%FT%T"), "name": a.name, "mode": mode, "engine": a.engine, "build": a.build,
            "engine_commit": binfo.get("commit"), "model": os.path.basename(model), "model_sha256": common.sha256_of(model),
            "cmd": " ".join(shlex.quote(c) for c in srv.cmd),
            "ttft_s": statistics.median(r["ttft_s"] for r in runs), "total_s": statistics.median(r["total_s"] for r in runs),
            "ttft_range": [min(r["ttft_s"] for r in runs), max(r["ttft_s"] for r in runs)],
            "prompt_n_processed": statistics.median(r["prompt_n"] for r in runs),
            "cache_n": statistics.median((r["cache_n"] or 0) for r in runs),
            "raw_log": os.path.relpath(os.path.join(log_dir, "cache.json"), common.ROOT)})


if __name__ == "__main__":
    main()
