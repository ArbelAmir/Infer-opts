#!/usr/bin/env python3
"""OpenVINO GenAI workload benchmark (run with engines/ov-venv/bin/python).

Holds the machine-wide bench lock, waits for idle, loads the pipeline once, then for each
workload runs 1 warm-up + N measured greedy repeats and appends one row per workload to
results/engines.jsonl.

  ov_bench.py --name ov-default
  ov_bench.py --name ov-bf16 --props '{"INFERENCE_PRECISION_HINT":"bf16"}'
  ov_bench.py --name ov-mtp --mtp
"""
import argparse, json, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from workloads import workloads

import openvino as ov
import openvino_genai as og

MODEL = os.path.join(common.MODELS, "ov", "Qwen3.5-2B-int4-ov")
WL = ["chat_short", "long_prompt", "long_gen"]


def vm_hwm_kb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1])


def med(v):
    return statistics.median(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--props", default="{}", help="JSON dict of OpenVINO properties")
    ap.add_argument("--mtp", action="store_true", help="MTP speculative decoding via draft_model")
    ap.add_argument("--num-assistant", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workloads", default=",".join(WL))
    a = ap.parse_args()
    props = json.loads(a.props)
    ver = og.__version__
    logdir = os.path.join(common.LOGS, "ov", a.name)
    os.makedirs(logdir, exist_ok=True)
    logpath = os.path.join(logdir, time.strftime("%Y%m%d-%H%M%S") + ".log")
    log = open(logpath, "w")

    def P(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    wls = workloads()
    with common.bench_lock(f"ov_bench {a.name}") as waited:
        P("lock waited s:", waited, "openvino", ov.__version__, "genai", ver)
        P("idle busy fraction:", common.wait_idle())
        kw = dict(props)
        sc = og.SchedulerConfig()
        sc.enable_prefix_caching = False
        t0 = time.perf_counter()
        if a.mtp:
            kw["draft_model"] = og.draft_model(MODEL, "CPU")
            kw["scheduler_config"] = sc
        pipe = og.VLMPipeline(MODEL, "CPU", **kw)
        load_s = time.perf_counter() - t0
        P("load_s", load_s)
        tok = pipe.get_tokenizer()
        cmd = " ".join(sys.argv)
        for wname in a.workloads.split(","):
            w = wls[wname]
            # apply the Qwen3.5 chat template ourselves with thinking disabled
            hist = og.ChatHistory(w["messages"]) if hasattr(og, "ChatHistory") else w["messages"]
            prompt = tok.apply_chat_template(hist, add_generation_prompt=True,
                                             extra_context={"enable_thinking": False})
            gc = og.GenerationConfig()
            gc.max_new_tokens = w["max_tokens"]
            gc.do_sample = False
            gc.apply_chat_template = False  # prompt is already templated
            if a.mtp:
                gc.num_assistant_tokens = a.num_assistant
            P(f"== {wname} prompt tail: {prompt[-120:]!r}")
            runs, text0 = [], None
            for i in range(1 + a.repeats):
                t = time.perf_counter()
                res = pipe.generate(prompt, generation_config=gc)
                wall = time.perf_counter() - t
                pm = res.perf_metrics
                text = res.texts[0] if hasattr(res, "texts") else str(res)
                r = dict(wall_s=wall, ttft_s=pm.get_ttft().mean / 1000.0,
                         tpot_ms=pm.get_tpot().mean, thr_tps=pm.get_throughput().mean,
                         prompt_n=pm.get_num_input_tokens(),
                         predicted_n=pm.get_num_generated_tokens())
                r["pp_tps"] = r["prompt_n"] / r["ttft_s"]
                r["gen_tps"] = 1000.0 / r["tpot_ms"] if r["tpot_ms"] else None
                P(("warmup" if i == 0 else f"rep{i}"), json.dumps(r))
                if i == 0:
                    text0 = text
                    P("TEXT:", text)
                else:
                    runs.append(r)
            row = dict(
                ts=time.strftime("%Y-%m-%dT%H:%M:%S"), name=a.name, engine="openvino-genai",
                engine_version=f"openvino {ov.__version__}; genai {ver}",
                model="OpenVINO/Qwen3.5-2B-int4-ov (INT4_ASYM g128)", workload=wname, cmd=cmd,
                config=dict(props, mtp=a.mtp, num_assistant_tokens=a.num_assistant if a.mtp else None,
                            thinking=False, greedy=True),
                prompt_n=runs[0]["prompt_n"], predicted_n=runs[0]["predicted_n"],
                pp_tps=med([r["pp_tps"] for r in runs]),
                gen_tps=med([r["gen_tps"] for r in runs]),
                gen_tps_source="1000/perf_metrics.get_tpot().mean (decode after first token)",
                ttft_s=med([r["ttft_s"] for r in runs]), total_s=med([r["wall_s"] for r in runs]),
                pp_tps_range=[min(r["pp_tps"] for r in runs), max(r["pp_tps"] for r in runs)],
                gen_tps_range=[min(r["gen_tps"] for r in runs), max(r["gen_tps"] for r in runs)],
                load_s=load_s, peak_rss_kb=vm_hwm_kb(), raw_log=logpath,
                output_head=text0[:200],
            )
            os.makedirs(common.RESULTS, exist_ok=True)
            with open(os.path.join(common.RESULTS, "engines.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")
            P("ROW", json.dumps(row))
        try:
            cm = pipe  # compiled-model properties are not exposed by VLMPipeline; see ov_probe
        except Exception:
            pass
    log.close()


if __name__ == "__main__":
    main()
