#!/usr/bin/env python3
"""Workload benchmark for llama.cpp-family servers (anything exposing llama-server's HTTP API).

Starts one server for one configuration, runs the fixed workloads (warm-up + N repeats each),
and appends one row per (config, workload) to results/results.jsonl with full provenance.
Holds the machine-wide benchmark lock for the whole run and waits for an idle machine first.

Example:
  harness/bench.py --name baseline --build default --model Qwen3.5-2B-Q4_K_M.gguf
  harness/bench.py --name t3 --build native --model X.gguf --server-args "-t 3 -tb 4 --mlock"
"""
import argparse, hashlib, http.client, json, os, shlex, signal, statistics, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from workloads import workloads


def http_json(port, path, timeout=5):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("GET", path)
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, body


def chat_stream(port, body, timeout=3600):
    """POST a streaming chat completion. Returns dict with client-side TTFT/total and server timings."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    payload = json.dumps(body)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {r.read()[:500]!r}")
    ttft = None
    content, reasoning, timings, usage, finish = [], [], None, None, None
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        ev = json.loads(data)
        if "timings" in ev:
            timings = ev["timings"]
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices", []):
            d = ch.get("delta", {})
            piece, rpiece = d.get("content"), d.get("reasoning_content")
            if (piece or rpiece) and ttft is None:
                ttft = time.perf_counter() - t0
            if piece:
                content.append(piece)
            if rpiece:
                reasoning.append(rpiece)
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    total = time.perf_counter() - t0
    c.close()
    return {"ttft_s": ttft, "total_s": total, "timings": timings or {}, "usage": usage,
            "content": "".join(content), "reasoning": "".join(reasoning), "finish": finish}


def vm_hwm_kb(pid):
    out = {}
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith(("VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem")):
                k, v = line.split(":")
                out[k] = int(v.split()[0])
    except FileNotFoundError:
        pass
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def spread(xs):
    xs = [x for x in xs if x is not None]
    return [min(xs), max(xs)] if xs else None


class Server:
    def __init__(self, binary, model, args, port, log_path, prefix="", env=None, model_flag="-m"):
        self.cmd = (shlex.split(prefix) if prefix else []) + [binary, model_flag, model, "--host", "127.0.0.1",
                                                              "--port", str(port)] + shlex.split(args)
        self.port, self.log_path = port, log_path
        self.env = dict(os.environ, **(env or {}))

    def start(self, timeout=1800):
        import socket
        for _ in range(120):  # a previous server's port can linger briefly; wait until it can be bound
            with socket.socket() as sk:
                try:
                    sk.bind(("127.0.0.1", self.port))
                    break
                except OSError:
                    time.sleep(1)
        self.log = open(self.log_path, "w")
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT, env=self.env)
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with {self.proc.returncode}; see {self.log_path}")
            try:
                st, _ = http_json(self.port, "/health", timeout=2)
                if st == 200:
                    break
            except OSError:
                pass
            if time.perf_counter() - self.t0 > timeout:
                self.stop()
                raise RuntimeError("server start timeout")
            time.sleep(0.05)
        self.load_s = time.perf_counter() - self.t0
        return self.load_s

    def stop(self):
        mem = vm_hwm_kb(self.proc.pid)
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.log.close()
        return mem


def run_config(a):
    engine_dir = os.path.join(common.ENGINES, a.engine)
    build_dir = os.path.join(engine_dir, f"build-{a.build}")
    binary = a.binary or os.path.join(build_dir, "bin", "llama-server")
    model = common.model_path(a.model)
    log_dir = os.path.join(common.LOGS, a.name)
    os.makedirs(log_dir, exist_ok=True)
    wl_all = workloads()
    names = list(wl_all) if a.workloads == "all" else a.workloads.split(",")
    extra = json.loads(a.body) if a.body else {}
    env = dict(kv.split("=", 1) for kv in a.env)
    rows = []
    def make_body(wn):
        w = wl_all[wn]
        body = {"messages": w["messages"], "max_tokens": w["max_tokens"], "stream": True,
                "temperature": 0, "seed": 42, "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": a.think == "on"}}
        body.update(extra)
        return body

    def show(wn, i, r):
        t = r["timings"]
        print(f"  {a.name:28s} {wn:12s} rep{i} pp {t.get('prompt_n')}tok {t.get('prompt_per_second') or 0:8.1f} t/s | "
              f"gen {t.get('predicted_n')}tok {t.get('predicted_per_second') or 0:7.2f} t/s | ttft {r['ttft_s'] or -1:6.2f}s "
              f"total {r['total_s']:6.2f}s{'  (warmup)' if r['warmup'] else ''}", flush=True)

    per_wl, props, mem, mem_after_load = {}, {}, {}, {}
    with common.bench_lock(a.name) as waited:
        busy = common.wait_idle()
        if a.drop_caches:
            subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True)
        srv = Server(binary, model, a.server_args, a.port, os.path.join(log_dir, "server.log"), a.prefix, env)
        if a.isolate:
            # Fresh server per measured request: required for drafting/caching features that keep state
            # across requests (n-gram pools, prompt cache), so a repeat can't feed on its own earlier output.
            tiny = {"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8, "stream": True,
                    "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
            for wn in names:
                per_wl[wn] = []
                for i in range(a.reps):
                    load_s = srv.start()
                    try:
                        chat_stream(a.port, tiny)
                        r = chat_stream(a.port, make_body(wn))
                    finally:
                        m = srv.stop()
                    mem = m if (m.get("VmHWM") or 0) > (mem.get("VmHWM") or 0) else mem
                    r["warmup"] = False
                    per_wl[wn].append(r)
                    show(wn, i, r)
        else:
            load_s = srv.start()
            try:
                _, props = http_json(a.port, "/props")
                props = json.loads(props)
            except Exception:
                props = {}
            mem_after_load = vm_hwm_kb(srv.proc.pid)
            try:
                for wn in names:
                    body = make_body(wn)
                    reps = []
                    for i in range(a.warmup + a.reps):
                        r = chat_stream(a.port, body)
                        r["warmup"] = i < a.warmup
                        reps.append(r)
                        show(wn, i, r)
                    per_wl[wn] = reps
            finally:
                mem = srv.stop()
    binfo = common.build_info(build_dir)
    for wn, reps in per_wl.items():
        meas = [r for r in reps if not r["warmup"]]
        g = lambda k: [r["timings"].get(k) for r in meas]
        raw_path = os.path.join(log_dir, f"{wn}.json")
        json.dump(reps, open(raw_path, "w"), indent=1)
        row = {
            "ts": time.strftime("%FT%T"), "name": a.name, "lever": a.lever, "workload": wn,
            "engine": a.engine, "build": a.build, "engine_commit": binfo.get("commit"),
            "build_cmake_args": binfo.get("cmake_args"), "cc": binfo.get("cc_version"),
            "model": os.path.basename(model), "model_sha256": common.sha256_of(model),
            "model_bytes": os.path.getsize(model),
            "cmd": " ".join(shlex.quote(c) for c in srv.cmd), "env": env, "request_extra": extra, "think": a.think,
            "reps": len(meas), "warmup": a.warmup, "isolate": a.isolate,
            "prompt_n": med(g("prompt_n")), "predicted_n": med(g("predicted_n")),
            "pp_tps": med(g("prompt_per_second")), "pp_tps_range": spread(g("prompt_per_second")),
            "gen_tps": med(g("predicted_per_second")), "gen_tps_range": spread(g("predicted_per_second")),
            "ttft_s": med([r["ttft_s"] for r in meas]), "ttft_range": spread([r["ttft_s"] for r in meas]),
            "total_s": med([r["total_s"] for r in meas]), "total_range": spread([r["total_s"] for r in meas]),
            "draft_n": med(g("draft_n")), "draft_n_accepted": med(g("draft_n_accepted")),
            "load_s": load_s, "peak_rss_kb": mem.get("VmHWM"), "rss_anon_kb": mem.get("RssAnon"),
            "rss_file_kb": mem.get("RssFile"), "rss_after_load_kb": mem_after_load.get("VmHWM"),
            "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx"),
            "output_sha1": hashlib.sha1(meas[-1]["content"].encode()).hexdigest()[:12] if meas else None,
            "finish": meas[-1]["finish"] if meas else None,
            "busy_before": round(busy, 3), "lock_wait_s": round(waited or 0, 1),
            "raw_log": os.path.relpath(raw_path, common.ROOT),
            "server_log": os.path.relpath(os.path.join(log_dir, "server.log"), common.ROOT),
            "host": common.host_info(),
        }
        common.append_jsonl(a.out, row)
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="unique config name (also the log dir)")
    p.add_argument("--lever", default="", help="which lever this config tests (free text tag)")
    p.add_argument("--engine", default="llama.cpp", help="source dir under engines/")
    p.add_argument("--build", default="default", help="build variant (engines/<engine>/build-<variant>)")
    p.add_argument("--binary", default="", help="explicit server binary (overrides engine/build)")
    p.add_argument("--model", required=True, help="model file under models/ (or absolute)")
    p.add_argument("--server-args", default="", help="extra llama-server args, one string")
    p.add_argument("--prefix", default="", help="command prefix, e.g. 'taskset -c 0-2'")
    p.add_argument("--env", action="append", default=[], help="KEY=VAL for the server process")
    p.add_argument("--body", default="", help="JSON merged into every request body")
    p.add_argument("--think", choices=["on", "off"], default="off")
    p.add_argument("--workloads", default="all")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--isolate", action="store_true", help="fresh server per measured request (stateful features)")
    p.add_argument("--drop-caches", action="store_true", help="drop page cache first (cold load time)")
    p.add_argument("--out", default=os.path.join(common.RESULTS, "results.jsonl"))
    a = p.parse_args()
    rows = run_config(a)
    for r in rows:
        print(f"{r['name']:28s} {r['workload']:12s} pp {r['pp_tps'] or 0:8.1f} t/s  gen {r['gen_tps'] or 0:7.2f} t/s  "
              f"ttft {r['ttft_s'] or -1:6.2f}s  total {r['total_s']:7.2f}s  rss {((r['peak_rss_kb'] or 0)/1048576):5.2f}G  load {r['load_s']:.1f}s")


if __name__ == "__main__":
    main()
