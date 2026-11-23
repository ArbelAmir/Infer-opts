# Benchmark harness

CPU inference benchmark harness for GGUF models on llama.cpp-family engines (plus adapters for
OpenVINO GenAI and MNN). Python 3 standard library only.

## One command

```bash
./run_all.sh            # from the repo root: build engines, fetch models, run the core matrix
QUICK=1 ./run_all.sh    # baseline + recommended configuration only
```

`run_all.sh` needs outbound access to `github.com` (engine source) and `huggingface.co` plus its
download CDN (`us.aws.cdn.hf.co` at the time of writing) for weights.

## Layout

| Path | What it is |
|---|---|
| `harness/common.py` | machine-wide lock, idle check, provenance helpers |
| `harness/locked.sh` | run any CPU-heavy command under the same lock |
| `harness/build_llamacpp.sh` | build an engine variant, write `BUILD_INFO.json` (commit, cmake cache, compiler) |
| `harness/fetch_model.sh` | resumable Hugging Face download + sha256 sidecar |
| `harness/workloads.py` | the five fixed workloads |
| `harness/bench.py` | workload benchmark against `llama-server` → `results/results.jsonl` |
| `harness/lbench.py` | `llama-bench` wrapper for wide synthetic sweeps → `results/llama_bench.jsonl` |
| `harness/quality.py`, `quality_items.py` | 50 checkable task items → `results/quality.jsonl` |
| `harness/kld.sh` | perplexity / KL divergence vs. a reference model → `results/kld.jsonl` |
| `harness/cache_bench.py` | shared-prefix prompt caching on/off → `results/cache.jsonl` |
| `harness/prompt_prune.py` | prompt pruning (BM25 / head truncation) with accuracy → `results/prompt_prune.jsonl` |
| `harness/throughput.py` | concurrent requests on parallel slots → `results/throughput.jsonl` |
| `harness/prune_layers.py` | write a GGUF with transformer blocks removed (depth pruning) |
| `harness/ov_*.py`, `harness/mnn*` | OpenVINO GenAI and MNN adapters → `results/engines.jsonl` |
| `harness/engine_quality.py` | the same 50 task items on OpenVINO GenAI / MNN |
| `harness/summarize.py` | turn the JSONL files into ratio-vs-baseline tables |
| `experiments/*.sh` | the experiment matrices that produced the report, in order |
| `results/` | machine-readable results (JSONL, one row per measurement) |
| `logs/` | raw logs behind every row (`raw_log` / `server_log` fields point here) |

`models/`, `engines/` and `data/` are git-ignored; they are re-created by `run_all.sh`.

## Measurement rules the harness enforces

- **One measurement at a time.** `bench.py`, `lbench.py`, `quality.py`, builds and quantization all take an
  exclusive `flock` on `/tmp/infer-opts-bench.lock`. Run anything else CPU-heavy through `harness/locked.sh`.
- **Idle machine.** Before each run the harness samples `/proc/stat` and waits (up to 120 s) for <5% busy;
  the observed value is stored as `busy_before`.
- **Warm-up + repeats.** Each workload gets 1 warm-up and N measured repeats (default 3; baseline 5).
  Rows store the median and the min..max range.
- **Fixed inputs.** Fixed prompts, `temperature 0`, `seed 42`, `cache_prompt false`, fixed output caps.
- **Provenance.** Every row carries the engine commit, cmake args, compiler, model file + sha256,
  exact command line, and the path of its raw log.

## Metrics

`pp_tps` prompt-processing tokens/s · `gen_tps` generation tokens/s · `ttft_s` client-side time to first
streamed token · `total_s` wall time of the request · `peak_rss_kb` server VmHWM · `load_s` process start
to `/health` OK. For drafting configs `draft_n` / `draft_n_accepted` are recorded as well.

## Adding things

**A model.** `harness/fetch_model.sh <hf_repo> <file_in_repo> [local_name]`, then pass `--model <local_name>`.

**A configuration.** One `bench.py` call is one configuration:

```bash
harness/bench.py --name my-config --lever threads --build default \
    --model Qwen3.5-2B-Q4_K_M.unsloth.gguf --server-args "-c 8192 -np 1 -t 3"
```

`--body '{"..."}'` merges JSON into each request, `--think on` enables Qwen thinking, `--prefix` wraps the
server command (`taskset`, `numactl`), `--env K=V` sets environment variables.

**A build variant.** `harness/build_llamacpp.sh llama.cpp myvariant -DGGML_LTO=ON` creates
`engines/llama.cpp/build-myvariant`; select it with `--build myvariant`.

**An engine.** Anything that ships a llama-server-compatible binary: clone it under `engines/<name>`, build
with `harness/build_llamacpp.sh <name> <variant> ...` and pass `--engine <name>`. Engines with a different
API get a small adapter script that writes the same row schema (see `harness/ov_bench.py`, `harness/mnn_bench.sh`).

## Reading results

```bash
harness/summarize.py                # tables with ratios against the `baseline` rows
```

`harness/summarize.py --baseline NAME` picks another reference row (use a baseline measured in the same
session: this host drifted by up to 20% between sessions). ik_llama.cpp needs `--jinja` in `--server-args`,
otherwise it ignores the request's `enable_thinking` flag and the model thinks.
