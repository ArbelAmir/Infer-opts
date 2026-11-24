# Fast local Qwen inference on a 4-core CPU: measured results

Machine: 4 vCPU Intel Sapphire Rapids (AVX-512, VNNI, AMX), 15.7 GiB RAM, no GPU, Linux.
Measured 2026-11-23/24. All numbers are from this machine; every row in `results/*.jsonl` carries the
engine commit, build flags, model checksum, exact command and raw log path.

## Recommended configuration

**Run the same Q4_K_M file on ik_llama.cpp with run-time repack, one slot, and an explicit context size.**

```bash
git clone https://github.com/ikawrakow/ik_llama.cpp && cd ik_llama.cpp
git checkout a28f79a5ac4b8db54eddf3907dc4e607ccca460c
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON && cmake --build build -j --target llama-server
# model: unsloth/Qwen3.5-2B-GGUF, file Qwen3.5-2B-Q4_K_M.gguf (the baseline file, unchanged)
build/bin/llama-server -m Qwen3.5-2B-Q4_K_M.gguf -c 8192 -np 1 --jinja -rtr
# add for edit/rewrite-style work (output repeats the input):   --spec-type ngram-simple
# in requests: "chat_template_kwargs": {"enable_thinking": false}, "cache_prompt": true
```

Or, from this repo: `QUICK=1 ./run_all.sh` (builds both engines, fetches the model, runs baseline and
recommended configuration, prints the ratio table). `run_all.sh` was written at the end and has **not**
been executed end to end; its individual commands are the ones used throughout.

Baseline = stock llama.cpp (`2392423`), default build, same file, default runtime flags. Both measured in the
same session (`baseline-r7`/`-r8` bracketing `ikj-q4km-rtr-r2`):

| | Baseline | Recommended | Ratio |
|---|---|---|---|
| Generation speed | 14.5 tok/s | 21.0 tok/s | 1.45× |
| Prompt processing, 3.5k-token prompt | 185 tok/s | 336 tok/s | 1.8× |
| Time to first token, 3.5k-token prompt | 19.1 s | 10.5 s | 1.8× |
| Short chat, total | 5.9 s | 3.8 s | 1.55× |
| Long prompt, short answer, total | 21.9 s | 12.2 s | 1.8× |
| Long generation (400 tokens), total | 26.7 s | 19.8 s | 1.35× |
| Code edit, total | 33.7 s | 24.3 s | 1.4× (2.2× with n-gram drafting) |
| Structured extraction, total | 31.2 s | 22.2 s | 1.4× |
| Peak memory | 6.06 GB | 1.84 GB | 0.30× |

**Quality cost: none expected and none measured.** The model file is identical; on the 50-item task set the
recommended configuration scored 44/50 against 45/50 for the baseline (one item, inside noise).

Caveat on the generation ratio: in one other session the same pair gave only 1.16× (17.4 vs 15.0 tok/s),
in two sessions about 1.45×. The prompt-processing gain (1.75–1.9×) reproduced every time. Treat generation
as "1.2–1.45×, usually the upper end".

## What to do, in order

Ratios are against the baseline in the same session unless noted. "Quality" is KL divergence from the BF16
model on wikitext-2 (lower is better; the baseline file is 0.023) plus the 50-item task score (baseline 45/50;
differences of ±3 items are noise).

| # | Lever | Prompt speed | Generation speed | Memory | Quality | Effort | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Don't let a small model think unbounded: thinking off, or capped (`--reasoning-budget 256`) | – | answers arrive 9× sooner than unlimited thinking | – | off 45/50; cap-256 47/50; unlimited 18/50 | one flag | high for the 2B model |
| 2 | Cache shared prompt prefixes (`cache_prompt: true`) | TTFT 17.6 s → 3.1 s on a repeated 3.5k prefix | – | – | none | request field | high |
| 3 | Prune long prompts by relevance (BM25 to 25%) | total 18.2 s → 4.8 s | – | – | 12/12 answers kept (head truncation: 5/12) | app-side code | medium (one document set) |
| 4 | ik_llama.cpp with `-rtr` | 1.8× | 1.2–1.45× | – | none (same file) | build another engine | high |
| 5 | Set `-c 8192 -np 1` instead of server defaults | 1.0× | 1.0× | 6.06 → 3.07 GB (stock) | none | two flags | high |
| 6 | n-gram drafting (`--spec-type ngram-simple`) | – | code edit 1.8×; others 0.9–1.05× | none | none (verified sampling) | one flag | high |
| 7 | Q4_0 instead of Q4_K_M (on ik `-rtr`: 23.8 tok/s; on stock: ~21) | ik: 1.5–1.9×; stock: 1.0× | 1.6× (ik), 1.45× (stock) | −0.1 GB | KLD 0.078; 44/50 | re-quantize | high on speed, medium on quality |
| 8 | Q8_0 on stock llama.cpp (AMX path) | 1.0–1.1× | 1.35× | 4.63 GB at 8k ctx | KLD 0.001; 45/50 | re-quantize | high |
| 9 | Throughput mode: 8 parallel slots | – | 43.6 tok/s aggregate (3.0×), 5.7 tok/s per request | 3.56 GB | none | flags | high |
| 10 | clang instead of gcc (stock, Q4_K_M) | 1.15–1.2× | 1.0–1.1× | – | none | rebuild | medium; no gain on Q4_0/Q8_0 |

Items 1–3 are about not doing work; they are larger than every engine-level change and stack with all of them.
Items 7 and 8 are alternatives to each other and to the recommended file; I would only take Q4_0 where
generation speed matters more than the last bit of fidelity.

### Supporting detail

**Thinking** (2B, stock, 50 items, cap 2048 tokens when on). Off: 45/50, 8.4k tokens, 593 s. Budget 256:
47/50 (reasoning items 14/14 vs 12/14), 19.1k tokens, 1488 s. Unlimited: 18/50, 77.7k tokens, 5601 s — 32 of
50 items ran into the 2048-token cap without producing an answer. Capped thinking costs 2.5× the time for a
gain that is inside noise overall and suggestive only on the reasoning items.

**Build.** A default llama.cpp build already uses `-march=native`; AVX-512 and the AMX repack path were active
without any option. Forcing a generic build (`GGML_NATIVE=OFF`) halves prompt speed (187 → 102 tok/s) and leaves
generation unchanged. LTO: no gain. Disabling OpenMP: −10% generation. AMX repack does nothing for Q4_K_M
(it only covers Q8_0/Q4_0/Q4_1-style formats), which is why those formats are fast on stock llama.cpp and
slow with `--no-repack` (Q4_0: 18.0 → 12.2 tok/s).

**Quantization** (llama-bench, stock, 4 threads; prompt / generation tok/s; KLD vs BF16):

| Format | Prompt | Generation | KLD | Size |
|---|---|---|---|---|
| BF16 | 103 | 7.7 | 0 | 3.78 GB |
| Q8_0 | 140 | 18.4 | 0.0010 | 2.01 GB |
| Q6_K | 109 | 9.0 | 0.0071 | 1.56 GB |
| Q5_K_M | 169 | 10.8 | 0.0148 | – |
| Q4_K_M, unsloth imatrix (baseline) | 187 | 14.1 | 0.0226 | 1.28 GB |
| Q4_K_M, static | 185 | 12.8 | 0.0445 | 1.27 GB |
| IQ4_XS | 195 | 12.8 | 0.0463 | 1.20 GB |
| Q4_1 | 143 | 18.4 | 0.0624 | 1.29 GB |
| Q4_0 | 133 | 18.0 | 0.0776 | 1.20 GB |
| Q3_K_M | 173 | 14.3 | 0.164 | – |
| Q2_K | 146 | 15.8 | 1.99 (unusable) | – |

An importance-matrix quantization halves the KLD at the same size and speed (0.045 → 0.023): prefer
published imatrix files over self-made static ones.

**Threads and memory.** 1/2/3/4 threads: 4.0/7.1/9.6/12.3 tok/s generation; 6 or 8 threads on 4 cores:
6.9–7.2 (worse than 2). Strict pinning, polling level, process priority, mmap/mlock, KV-cache quantization,
transparent huge pages (ik `-thp`): all within noise. Micro-batch 64: −10% prompt speed; ≥128: flat.
Generation slows as the context fills (12.3 tok/s empty → 10.2 at 8k tokens).

**Decoding speedups** (total time vs no drafting, same engine):

| Method | Chat | Long generation | Code edit | Extraction |
|---|---|---|---|---|
| Draft model 0.8B, 4 tokens (stock) | 0.48× | 0.56× | 0.97× | 1.00× |
| Draft model, 16 tokens, p-min 0.75 | 0.54× | 0.55× | 1.11× | 1.20× |
| n-gram simple (stock) | 0.94× | 0.98× | 1.78× | 1.05× |
| n-gram simple (ik `-rtr`) | 0.86× | 0.97× | 1.83× | 1.09× |
| MTP heads, 3 tokens (stock) | 0.78× | 0.75× | 1.23× | 1.30× |
| MTP heads, 2 tokens (ik `-rtr`) | 0.68× | 0.98× | 1.26× | 1.38× |

A separate draft model competes for the same four cores and loses. MTP needs the MTP variant of the file and
4 GB, and only pays on structured output.

**Other engines** (their own 4-bit conversions, thinking off, my rerun in the baseline-r7/r8 session):

| Engine | Generation | Long-prompt speed | Short-prompt TTFT (455 tok) | Memory | Task score |
|---|---|---|---|---|---|
| Stock llama.cpp baseline | 14.5 | 185 | 2.2 s | 6.06 GB | 45/50 |
| ik_llama.cpp `-rtr` | 21.0 | 336 | 1.4 s | 1.84 GB | 44/50 |
| MNN 3.6.2, HQQ 4-bit | 25.5 (1.75×) | 173 | 2.5 s | 1.9–2.0 GB | 40/50 |
| OpenVINO GenAI 2026.4, int4, f32 hint | 19.0 (1.3×) | 499 (2.7×) | 5.8 s | 3.7 GB | 41/50 |
| OpenVINO GenAI, default (bf16) | 12.6 | 571 (3.1×) | 5.3 s | 3.3 GB | not run |

MNN has the fastest generation measured and OpenVINO by far the fastest long-prompt processing, but both
scored 4–5 items lower than the GGUF files. With 50 items that is at the edge of noise, not proof of a
quality loss — it is the reason neither is the recommendation. OpenVINO is also slow on mid-size prompts
(86 tok/s on a 455-token prompt), and its MTP mode was slower than plain decoding.

**Depth pruning** (drop aligned groups of 4 of the 24 blocks, no fine-tuning, static Q4_K_M):

| Variant | Prompt | Generation | PPL (ref 13.5) | KLD |
|---|---|---|---|---|
| Full 2B Q4_K_M | 198 | 14.7 | 14.1 | 0.045 |
| Drop blocks 12–15 | 228 | 14.1 | 30.7 | 0.84 |
| Drop blocks 16–19 | 228 | 14.1 | 39.9 | 1.02 |
| Drop blocks 4–7 | 227 | 13.9 | 38.7 | 1.00 |
| Drop blocks 20–23 | 228 | 13.5 | 2606 | 5.31 |
| Drop 8–11 and 16–19 | 253 | 16.3 | 49.5 | 1.27 |
| Qwen3.5-0.8B Q4_K_M | 431 | 29.6 | 19.8 | 0.43 |
| 2B Q3_K_M | 173 | 14.3 | 15.6 | 0.16 |

Removing a sixth of the blocks gives +15% prompt speed, no generation gain, and worse fidelity than the
next smaller model, which is twice as fast. Depth pruning without retraining is dominated on every axis.
Neither server exposes an early-exit or layer-skip option (checked `--help` of both).

**Throughput mode** (stock, 200-token essays): 1/2/4/8 concurrent requests → 14.7/17.4/28.6/43.6 tok/s
aggregate, per-request latency 13.6/23.0/27.9/36.7 s. Good for batch jobs, bad for interactive use.

## Larger models in 15 GB

| Model | Generation | Prompt speed | Peak memory | Task score |
|---|---|---|---|---|
| Qwen3.5-2B Q4_K_M (recommended config) | 21 tok/s | 336 | 1.8 GB | 44–45/50 |
| Qwen3.5-9B Q4_K_M (5.7 GB), ik `-rtr` | 4.7 tok/s | 83 | 6.8 GB | 46/50 |
| Qwen3.6-35B-A3B UD-Q2_K_XL (12.3 GB), ik, mmap | 7.1 tok/s | 83 | 13.0 GB | 48/50 |
| same, stock llama.cpp | 6.8 tok/s | 50 | 13.0 GB | not run |

The 35B mixture-of-experts model is the most capable Qwen that genuinely ran here: it generates faster than
the dense 9B because only ~3B parameters are active per token, and scored highest. Limits: it only fits
memory-mapped — with ik `-rtr` (12.07 tok/s in llama-bench) the server was killed for lack of memory
mid-run — a 3.5k-token prompt takes 43 s to first token, and at ~2.7 bits per weight its fidelity versus the
full-precision model was not measured (no reference fits). Qwen3.6-27B and the Qwen3.8 line were not run:
a dense 27B at a size that fits would be slower than the 35B MoE; Qwen3.8-Flash-Next quantizations exceed 100 GB.

## What did not help

- Draft-model speculative decoding (0.5× on free-form text).
- More threads than cores; pinning, priority, polling, mlock, huge pages, KV quantization.
- LTO; disabling OpenMP (worse).
- Unlimited thinking on the 2B model (18/50, 9× the time).
- Depth pruning.
- OpenVINO's MTP mode; MNN `memory=normal` (6 tok/s, 9.7 GB).
- IQ4_XS and Q6_K on ik `-rtr` for generation (17 and 14 tok/s vs 21 for Q4_K_M).

## Failed, skipped or not tested

- **ONNX Runtime GenAI**: not evaluated — no GenAI-format Qwen3.5 export was found and builder support was unverified.
- **BLAS builds**: not tested; llama.cpp does not route quantized matmuls through BLAS (my understanding, not measured).
- **Intel compilers, explicit hugetlbfs pages**: not available / not tried.
- **AMX inside OpenVINO and MNN**: not verified; only their end-to-end speed was measured.
- **35B with `-rtr`**: out of memory in the server. **35B IQ3_XXS** (13.2 GB): llama-bench only (11.0 tok/s with `-rtr`), no accuracy run.
- **ik first workload run** had thinking enabled (ik needs `--jinja` to honour the flag); those `ik-*` rows are kept but superseded by `ikj-*`.
- **Depth-pruning first attempt** failed in the GGUF writer; fixed and rerun (`12b`).
- **Two container restarts** interrupted the queue; affected runs were repeated. Q4_1 and IQ4_XS task scores were dropped for time.
- **Three stock rows of experiment 09** failed to start (port still bound); the harness now waits for the port and they were rerun.
- **Host drift**: the baseline moved between 11.8 and 15.9 tok/s across sessions (`baseline` … `baseline-r10`). Ratios are only quoted within a session.
- The 50-item task set resolves differences of about ±3 items; KLD covers GGUF files only.

## What to test next

1. A larger task set (300+ items) on MNN and OpenVINO: if their scores hold up, MNN is the generation-speed winner and OpenVINO the long-prompt winner.
2. Capped thinking at 64/128 tokens and on the 9B/35B models, where thinking may pay differently.
3. ik `-rtr` for the 35B with a smaller quantization or 24 GB RAM (llama-bench suggests 12 tok/s).
4. An imatrix Q4_0, which could keep the Q4_0 speed with less fidelity loss.
5. n-gram drafting combined with prompt caching on a real edit loop.

## What should generalize (my judgment, not measured elsewhere)

- Likely general: thinking control, prompt caching and prompt pruning dominate engine tuning; threads = physical cores; draft models lose on CPU-bound machines; one slot and an explicit context save memory.
- Likely specific to AVX-512/AMX Xeons: the Q8_0/Q4_0 advantage on stock llama.cpp (it comes from the AMX path), and the size of the ik gain.
- Likely specific to small models: unlimited thinking collapsing, and generation barely changing when blocks are removed.
- Unknown: whether MNN's lead holds on CPUs without AVX-512.

## Reproducing

`./run_all.sh` (see `harness/README.md` for adding a model, engine or configuration);
`experiments/*.sh` are the exact matrices in the order they were run; `harness/summarize.py --baseline NAME`
prints ratio tables from `results/results.jsonl`.
