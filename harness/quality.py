#!/usr/bin/env python3
"""Task-accuracy guardrail: run the fixed checkable items against one server configuration.

Appends one summary row to results/quality.jsonl (accuracy overall and per category, tokens generated,
wall time per item) and keeps every reply in logs/<name>/quality.json.

  harness/quality.py --name q-q4km --model Qwen3.5-2B-Q4_K_M.gguf --server-args "-c 8192 -np 1"
  harness/quality.py --name q-think --model X.gguf --think on --max-tokens 2048
"""
import argparse, json, os, re, shlex, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from bench import Server, chat_stream
from quality_items import items

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def check(item, reply):
    lines = [l for l in reply.strip().splitlines() if re.search(r"answer\s*:", l, re.I)]
    if not lines:
        return False, "no_answer_line"
    last = re.split(r"answer\s*:", lines[-1], flags=re.I)[-1].strip().strip("*`. ")
    if not last:  # "Answer:" on its own line, value on the next
        tail = reply.strip().splitlines()
        last = tail[-1].strip().strip("*`. ")
    if item["kind"] == "num":
        nums = NUM.findall(last.replace(",", ""))
        if not nums:
            return False, last[:60]
        try:
            return abs(float(nums[-1]) - float(item["answer"])) < 1e-6, last[:60]
        except ValueError:
            return False, last[:60]
    return item["answer"].lower() in last.lower(), last[:60]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--lever", default="")
    p.add_argument("--engine", default="llama.cpp")
    p.add_argument("--build", default="default")
    p.add_argument("--binary", default="")
    p.add_argument("--model", required=True)
    p.add_argument("--server-args", default="-c 8192 -np 1")
    p.add_argument("--body", default="")
    p.add_argument("--think", choices=["on", "off"], default="off")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--categories", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--out", default=os.path.join(common.RESULTS, "quality.jsonl"))
    a = p.parse_args()
    build_dir = os.path.join(common.ENGINES, a.engine, f"build-{a.build}")
    binary = a.binary or os.path.join(build_dir, "bin", "llama-server")
    model = common.model_path(a.model)
    log_dir = os.path.join(common.LOGS, a.name)
    os.makedirs(log_dir, exist_ok=True)
    its = items()
    if a.categories:
        its = [i for i in its if i["category"] in a.categories.split(",")]
    if a.limit:
        its = its[: a.limit]
    extra = json.loads(a.body) if a.body else {}
    out = []
    with common.bench_lock(a.name):
        common.wait_idle()
        srv = Server(binary, model, a.server_args, a.port, os.path.join(log_dir, "server.log"))
        srv.start()
        try:
            for it in its:
                body = {"messages": [{"role": "user", "content": it["prompt"]}], "max_tokens": a.max_tokens,
                        "stream": True, "temperature": 0, "seed": 42, "cache_prompt": False,
                        "chat_template_kwargs": {"enable_thinking": a.think == "on"}}
                body.update(extra)
                r = chat_stream(a.port, body)
                ok, got = check(it, r["content"])
                t = r["timings"]
                out.append({"id": it["id"], "category": it["category"], "ok": ok, "got": got, "want": it["answer"],
                            "total_s": r["total_s"], "predicted_n": t.get("predicted_n"), "prompt_n": t.get("prompt_n"),
                            "reasoning_chars": len(r["reasoning"]), "finish": r["finish"],
                            "content": r["content"], "reasoning": r["reasoning"]})
                print(f"  {it['id']:9s} {'OK ' if ok else 'BAD'} got={got!r:30.30} want={it['answer']!r:10} "
                      f"{t.get('predicted_n')}tok {r['total_s']:.1f}s", flush=True)
        finally:
            srv.stop()
    json.dump(out, open(os.path.join(log_dir, "quality.json"), "w"), indent=1)
    cats = sorted({o["category"] for o in out})
    binfo = common.build_info(build_dir)
    row = {
        "ts": time.strftime("%FT%T"), "name": a.name, "lever": a.lever, "engine": a.engine, "build": a.build,
        "engine_commit": binfo.get("commit"), "model": os.path.basename(model), "model_sha256": common.sha256_of(model),
        "cmd": " ".join(shlex.quote(c) for c in srv.cmd), "request_extra": extra, "think": a.think, "max_tokens": a.max_tokens,
        "n_items": len(out), "n_ok": sum(o["ok"] for o in out), "accuracy": sum(o["ok"] for o in out) / max(1, len(out)),
        "by_category": {c: [sum(o["ok"] for o in out if o["category"] == c), sum(1 for o in out if o["category"] == c)] for c in cats},
        "tokens_total": sum(o["predicted_n"] or 0 for o in out), "wall_total_s": sum(o["total_s"] for o in out),
        "median_item_s": statistics.median(o["total_s"] for o in out),
        "n_truncated": sum(1 for o in out if o["finish"] == "length"),
        "failed_ids": [o["id"] for o in out if not o["ok"]],
        "raw_log": os.path.relpath(os.path.join(log_dir, "quality.json"), common.ROOT),
    }
    common.append_jsonl(a.out, row)
    print(f"{a.name}: {row['n_ok']}/{row['n_items']} correct  {row['by_category']}  tokens {row['tokens_total']}  "
          f"wall {row['wall_total_s']:.0f}s  truncated {row['n_truncated']}")


if __name__ == "__main__":
    main()
