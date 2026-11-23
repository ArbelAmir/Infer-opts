#!/usr/bin/env python3
"""Print tables from results/*.jsonl. Workload rows are shown as ratios against the `baseline` config
(or --baseline NAME). Later rows with the same (name, workload) replace earlier ones.

  harness/summarize.py                      # everything
  harness/summarize.py --lever threads      # one lever
  harness/summarize.py --lbench             # llama-bench sweeps
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WL = ["chat_short", "long_prompt", "long_gen", "code_edit", "extract"]


def load(name):
    p = os.path.join(ROOT, "results", name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", default="")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--lbench", action="store_true")
    ap.add_argument("--metric", default="total_s", choices=["total_s", "gen_tps", "pp_tps", "ttft_s"])
    a = ap.parse_args()
    if a.lbench:
        for r in load("llama_bench.jsonl"):
            if a.lever and a.lever not in r["lever"]:
                continue
            s = sorted(r["samples_ts"])
            print(f"{r['name']:30s} {r['model'][:32]:32s} t={r['n_threads']} rp={r['repack']} {r['kind']} "
                  f"{s[len(s)//2]:8.2f} t/s [{s[0]:.2f}..{s[-1]:.2f}]")
        return
    rows = {}
    for r in load("results.jsonl"):
        rows[(r["name"], r["workload"])] = r
    names = []
    for (n, w), r in rows.items():
        if n not in names and (not a.lever or a.lever in r["lever"] or n == a.baseline):
            names.append(n)
    base = {w: rows.get((a.baseline, w)) for w in WL}
    print(f"{'config':30s} " + " ".join(f"{w:>24s}" for w in WL) + "   rss_GB")
    print(f"{'':30s} " + " ".join(f"{'pp t/s | gen t/s | total':>24s}" for w in WL))
    for n in names:
        cells, rss = [], None
        for w in WL:
            r = rows.get((n, w))
            if not r:
                cells.append(f"{'-':>24s}")
                continue
            rss = r["peak_rss_kb"]
            b = base[w]
            sp = f"x{b['total_s'] / r['total_s']:.2f}" if b else ""
            cells.append(f"{r['pp_tps'] or 0:6.0f} |{r['gen_tps'] or 0:6.2f} |{r['total_s']:6.1f}s {sp:>5s}"[:24].rjust(24))
        print(f"{n:30s} " + " ".join(cells) + f"   {(rss or 0) / 1048576:5.2f}")


if __name__ == "__main__":
    main()
