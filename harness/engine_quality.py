#!/usr/bin/env python3
"""Task-accuracy guardrail (same 50 items and checker as quality.py) for the non-llama.cpp engines.

  engines/ov-venv/bin/python harness/engine_quality.py --engine ov  --name qual-ov-f32 --props '{"INFERENCE_PRECISION_HINT":"f32"}'
  python3 harness/engine_quality.py --engine mnn --name qual-mnn

Thinking off, greedy, max 512 new tokens, no system turn. Appends one row to results/quality.jsonl.
"""
import argparse, json, os, re, statistics, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from quality import check
from quality_items import items


def run_ov(a, its, emit):
    import openvino as ov, openvino_genai as og
    model = os.path.join(common.MODELS, "ov", "Qwen3.5-2B-int4-ov")
    pipe = og.VLMPipeline(model, "CPU", **json.loads(a.props))
    tok = pipe.get_tokenizer()
    for it in its:
        msgs = [{"role": "user", "content": it["prompt"]}]
        hist = og.ChatHistory(msgs) if hasattr(og, "ChatHistory") else msgs
        prompt = tok.apply_chat_template(hist, add_generation_prompt=True, extra_context={"enable_thinking": False})
        gc = og.GenerationConfig()
        gc.max_new_tokens, gc.do_sample, gc.apply_chat_template = a.max_tokens, False, False
        t = time.perf_counter()
        res = pipe.generate(prompt, generation_config=gc)
        emit(it, res.texts[0], time.perf_counter() - t, res.perf_metrics.get_num_generated_tokens())
    return f"openvino {ov.__version__}; genai {og.__version__}", "OpenVINO/Qwen3.5-2B-int4-ov"


def run_mnn(a, its, emit):
    mnn = os.path.join(common.ENGINES, "MNN")
    cfg = os.path.join(common.MODELS, "mnn", "Qwen3.5-2B-MNN", "config.json")
    d = os.path.join(common.LOGS, a.name)
    open(os.path.join(d, "system.txt"), "w").close()
    extra = json.dumps({"thread_num": 4, "precision": a.mnn_precision, "memory": "low"})
    for it in its:
        open(os.path.join(d, "user.txt"), "w").write(it["prompt"])
        out = subprocess.run([os.path.join(mnn, "build", "mnn_workload"), cfg, os.path.join(d, "system.txt"),
                              os.path.join(d, "user.txt"), str(a.max_tokens), "0", os.path.join(d, "out"), extra],
                             capture_output=True, text=True).stdout
        m = re.search(r"gen_n=(\d+).* total_s=([\d.]+)", out)
        emit(it, open(os.path.join(d, "out.run0.txt")).read(), float(m[2]), int(m[1]))
    commit = subprocess.check_output(["git", "-C", mnn, "rev-parse", "HEAD"], text=True).strip()
    return f"MNN {commit}", "taobao-mnn/Qwen3.5-2B-MNN (HQQ 4-bit)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True, choices=["ov", "mnn"])
    p.add_argument("--name", required=True)
    p.add_argument("--props", default="{}")
    p.add_argument("--mnn-precision", default="low")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    its = items()[: a.limit or None]
    log_dir = os.path.join(common.LOGS, a.name)
    os.makedirs(log_dir, exist_ok=True)
    out = []

    def emit(it, text, secs, ntok):
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        ok, got = check(it, text)
        out.append({"id": it["id"], "category": it["category"], "ok": ok, "got": got, "want": it["answer"],
                    "total_s": secs, "predicted_n": ntok, "content": text})
        print(f"  {it['id']:9s} {'OK ' if ok else 'BAD'} got={got!r:30.30} want={it['answer']!r:10} {ntok}tok {secs:.1f}s", flush=True)

    with common.bench_lock(a.name):
        common.wait_idle()
        ver, model = (run_ov if a.engine == "ov" else run_mnn)(a, its, emit)
    json.dump(out, open(os.path.join(log_dir, "quality.json"), "w"), indent=1)
    cats = sorted({o["category"] for o in out})
    row = {"ts": time.strftime("%FT%T"), "name": a.name, "lever": "quality", "engine": a.engine, "engine_commit": ver,
           "model": model, "cmd": " ".join(sys.argv), "think": "off", "max_tokens": a.max_tokens,
           "n_items": len(out), "n_ok": sum(o["ok"] for o in out), "accuracy": sum(o["ok"] for o in out) / max(1, len(out)),
           "by_category": {c: [sum(o["ok"] for o in out if o["category"] == c), sum(1 for o in out if o["category"] == c)] for c in cats},
           "tokens_total": sum(o["predicted_n"] or 0 for o in out), "wall_total_s": sum(o["total_s"] for o in out),
           "median_item_s": statistics.median(o["total_s"] for o in out),
           "n_truncated": sum(1 for o in out if (o["predicted_n"] or 0) >= a.max_tokens),
           "failed_ids": [o["id"] for o in out if not o["ok"]],
           "raw_log": os.path.relpath(os.path.join(log_dir, "quality.json"), common.ROOT)}
    common.append_jsonl(os.path.join(common.RESULTS, "quality.jsonl"), row)
    print(f"{a.name}: {row['n_ok']}/{row['n_items']} correct  {row['by_category']}  tokens {row['tokens_total']}  wall {row['wall_total_s']:.0f}s")


if __name__ == "__main__":
    main()
