#!/usr/bin/env python3
"""Prompt pruning on the long-context workload: answer 12 checkable questions over a ~3.3k-token context
with (a) the full context, (b) query-aware paragraph selection (BM25, keep ~budget of the characters),
(c) naive head truncation to the same budget. Reports accuracy, prompt tokens and time per question.
Appends rows to results/prompt_prune.jsonl.

  harness/prompt_prune.py --name pp-2b --model X.gguf --budget 0.25
"""
import argparse, json, math, os, re, shlex, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from bench import Server, chat_stream
from workloads import long_context, SYSTEM

QA = [
    ("To what maximum length, in centimetres, may Homarus gammarus grow?", ["60"]),
    ("What colour are European lobsters in life?", ["blue"]),
    ("For up to how long do female European lobsters carry their eggs before hatching?", ["year"]),
    ("Which species is Homarus gammarus closely related to?", ["americanus", "american lobster"]),
    ("What is the name of the moulting process by which lobsters shed their exoskeleton?", ["ecdysis"]),
    ("What is the eastern end point city of M-82?", ["howard city"]),
    ("In 2009, what was the peak daily traffic volume measured on M-82 west of Fremont?", ["16,532", "16532", "16 @,@ 532"]),
    ("In 2009, what was the lowest daily traffic volume (the traffic nadir) on M-82?", ["4,018", "4018", "4 @,@ 018"]),
    ("By which year had the first usage of the M-82 designation appeared in the Upper Peninsula?", ["1919"]),
    ("With which highway does M-82 run concurrently over the Muskegon River in Newaygo?", ["m-37", "m @-@ 37", "m‑37", "m 37"]),
    ("At which exit number on US 131 is the east end of M-82?", ["118"]),
    ("Which national forest does M-82 run along the southern edge of?", ["manistee"]),
]
WORD = re.compile(r"[a-z0-9]+")


def bm25_select(paras, query, budget_chars):
    docs = [WORD.findall(p.lower()) for p in paras]
    q = set(WORD.findall(query.lower()))
    n = len(docs)
    avg = sum(map(len, docs)) / n
    df = {t: sum(1 for d in docs if t in d) for t in q}
    scores = []
    for d in docs:
        s = 0.0
        for t in q:
            f = d.count(t)
            if f:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(d) / avg))
        scores.append(s)
    chosen, used = set(), 0
    for i in sorted(range(n), key=lambda i: -scores[i]):
        if used + len(paras[i]) > budget_chars and chosen:
            continue
        chosen.add(i)
        used += len(paras[i])
        if used >= budget_chars:
            break
    return "\n".join(paras[i] for i in sorted(chosen))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--engine", default="llama.cpp")
    p.add_argument("--build", default="default")
    p.add_argument("--model", required=True)
    p.add_argument("--server-args", default="-c 8192 -np 1")
    p.add_argument("--budget", type=float, default=0.25)
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--out", default=os.path.join(common.RESULTS, "prompt_prune.jsonl"))
    a = p.parse_args()
    build_dir = os.path.join(common.ENGINES, a.engine, f"build-{a.build}")
    model = common.model_path(a.model)
    log_dir = os.path.join(common.LOGS, a.name)
    os.makedirs(log_dir, exist_ok=True)
    ctx = long_context()
    paras = [x for x in ctx.split("\n") if x.strip()]
    budget = int(len(ctx) * a.budget)
    variants = {"full": lambda q: ctx, "bm25": lambda q: bm25_select(paras, q, budget), "head": lambda q: ctx[:budget]}
    out = {v: [] for v in variants}
    with common.bench_lock(a.name):
        common.wait_idle()
        srv = Server(os.path.join(build_dir, "bin", "llama-server"), model, a.server_args, a.port, os.path.join(log_dir, "server.log"))
        srv.start()
        try:
            for v, fn in variants.items():
                for q, answers in QA:
                    body = {"messages": [{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": "Answer the question using the text below. Reply with a short phrase.\n\n" + fn(q) + "\n\nQuestion: " + q}],
                            "max_tokens": 48, "stream": True, "temperature": 0, "seed": 42, "cache_prompt": False,
                            "chat_template_kwargs": {"enable_thinking": False}}
                    r = chat_stream(a.port, body)
                    ok = any(x in r["content"].lower() for x in answers)
                    out[v].append({"q": q, "ok": ok, "answer": r["content"][:200], "prompt_n": r["timings"].get("prompt_n"),
                                   "total_s": r["total_s"], "ttft_s": r["ttft_s"]})
                    print(f"  {a.name} {v:5s} {'OK ' if ok else 'BAD'} {r['timings'].get('prompt_n')}tok {r['total_s']:.1f}s  {r['content'][:60]!r}", flush=True)
        finally:
            srv.stop()
    json.dump(out, open(os.path.join(log_dir, "prompt_prune.json"), "w"), indent=1)
    binfo = common.build_info(build_dir)
    for v, runs in out.items():
        row = {"ts": time.strftime("%FT%T"), "name": a.name, "variant": v, "budget": a.budget if v != "full" else 1.0,
               "engine": a.engine, "build": a.build, "engine_commit": binfo.get("commit"), "model": os.path.basename(model),
               "model_sha256": common.sha256_of(model), "cmd": " ".join(shlex.quote(c) for c in srv.cmd),
               "n_ok": sum(r["ok"] for r in runs), "n": len(runs), "prompt_n": statistics.median(r["prompt_n"] for r in runs),
               "total_s": statistics.median(r["total_s"] for r in runs), "ttft_s": statistics.median(r["ttft_s"] for r in runs),
               "raw_log": os.path.relpath(os.path.join(log_dir, "prompt_prune.json"), common.ROOT)}
        common.append_jsonl(a.out, row)
        print(f"{a.name} {v:5s}: {row['n_ok']}/{row['n']} correct, median prompt {row['prompt_n']} tok, median total {row['total_s']:.1f}s")


if __name__ == "__main__":
    main()
