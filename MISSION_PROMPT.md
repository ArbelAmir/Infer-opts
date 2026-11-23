# Research mission: make local Qwen inference as fast as it can be on this hardware

You are the principal researcher on this mission. You own the research questions, the experiment design, the interpretation of results, and the final recommendations. You have Sonnet and Haiku subagents available through the Agent tool (`model: "sonnet"` / `model: "haiku"`) and should use them to do the legwork, but the conclusions are yours and you are accountable for every number in the final report.

## What I want and why

I run Qwen models locally and want to know how to get the most out of them: which combination of engine, build, quantization, runtime settings, and decoding tricks gives the best speed at acceptable quality on a given machine. I do not want a survey of what the internet says. I want measured answers from this machine, with a reproducible harness I can rerun on other hardware later, and a short ranked list of "do this, in this order, and expect about this much".

The repository (`Infer-opts`) is empty. Everything you build — harness, raw results, report — goes in it. Commit as you go to the branch this session is assigned, and push at each milestone; the container is ephemeral.

## Target hardware

Unless I have edited this section, the target is the machine you are running on. When this prompt was written, that was:

- 4 vCPUs of a Sapphire Rapids-class Xeon at 2.1 GHz, no SMT, under a hypervisor
- AVX2, AVX-512 (F/BW/VL/VNNI/BF16/FP16), AVX-VNNI, and AMX (tile, int8, bf16)
- 15 GB RAM, no GPU, about 60 GB of free disk
- gcc, clang, cmake, and Python 3 installed

Re-check this with `lscpu` and `free` before you start; do not trust the list above if the machine disagrees. Everything is CPU-only, so memory bandwidth and the 15 GB ceiling are the binding constraints, and the wide vector and AMX extensions are the unusual asset worth exploiting.

## A blocker to clear first

When this was written, the container's network policy allowed PyPI, npm, and `raw.githubusercontent.com`, but refused `huggingface.co`, its LFS CDN, ModelScope, and the Ollama registry. Without one of those you cannot get model weights, and the mission cannot produce real measurements.

So your first action is to test whether you can download a small GGUF file and clone the engine source. If either is refused by the proxy, read the environment documentation, then propose the smallest allowlist change that fixes it through the environment tools (I will see an approval prompt). Follow the redirect of a real download to learn the exact CDN hostname rather than guessing it. If I decline, do not fake it: do the parts that need no weights (building the engines, writing the harness, validating it on any tiny model you can obtain from an allowed host), and state clearly in the report which results are missing and why.

## Which models

Check what the current Qwen open-weight line-up actually is before choosing; it moves fast and your memory may be stale. As of late 2026 the search results pointed to the Qwen 3.6 family (a dense 27B, a 35B-A3B mixture-of-experts model, and variants with multi-token-prediction heads), with smaller Qwen 3.5 models (around 4B and below) released earlier. Verify this.

Pick two working models:

1. A small one (roughly 2–4B parameters at 4-bit) for the broad sweep, because each run takes seconds and you can afford to explore many configurations.
2. The most capable Qwen that genuinely fits in 15 GB with a usable context, to confirm that the winning configurations still win at the size someone would actually want to use. A MoE model with few active parameters per token is especially interesting on CPU, since generation cost follows the active parameters while memory follows the total; find out whether one fits at a quantization that is not ruinous for quality.

If a draft model is needed for speculative decoding, the smallest Qwen from the same family and tokenizer is the natural candidate.

## The questions to answer

Establish a baseline first: stock llama.cpp built with default settings, a Q4_K_M quantization, default runtime flags. Every later claim is a ratio against that baseline.

Then work through the levers below. They are listed roughly in the order I expect them to pay off, but that ordering is a hypothesis for you to test, not a plan to follow. Drop a lever quickly if the first measurements show it is flat, and spend the saved time where the gradient is.

- **Build.** Does the default build actually use AVX-512, VNNI, and AMX on this CPU, or does it need flags to turn them on? Check the build's reported system info rather than assuming. Compare compilers and optimization settings (native tuning, LTO, a BLAS or oneDNN-style backend for prompt processing) only as far as they move the numbers.
- **Quantization.** Speed, memory, and quality across the common formats (Q8, Q6, Q5, Q4 K-variants, Q4_0, the imatrix/IQ variants, lower-bit options). Some formats have much faster kernels on some instruction sets than others, so the smallest file is not always the fastest. This is where quality has to be measured, not assumed.
- **Threads and memory.** Generation threads versus batch threads, pinning, mlock versus mmap, huge pages, batch and micro-batch size, context size. On a 4-core VM the right thread count may not be 4.
- **Decoding-level speedups.** Speculative decoding with a draft model, n-gram or prompt-lookup drafting, and the multi-token-prediction variants of Qwen if the engine supports them. These help on some workloads (code, extraction, anything repetitive) and hurt on others, so report by workload. On a 4-core machine the draft model competes for the same cores, which is exactly why this needs measuring.
- **Not generating tokens you do not need.** Qwen's thinking mode can multiply wall-clock time by producing reasoning tokens. Measure end-to-end time to a correct answer with thinking on and off, and with capped reasoning, on tasks where it matters and tasks where it does not. Include prompt and KV caching for repeated prefixes (system prompts, agent loops, RAG templates), since for many real workloads prompt processing dominates.
- **Other engines.** Stock llama.cpp is the baseline, not the presumed winner. Evaluate the serious CPU alternatives that exist today — for example the performance-oriented llama.cpp forks, OpenVINO, ONNX Runtime GenAI, and MNN (Alibaba's own runtime, which has first-party Qwen support) — and include any that have a credible claim to beat llama.cpp on Intel CPUs with AMX. An engine that takes more than a bounded effort to get running with a Qwen model gets recorded as "not evaluated, here is why" rather than consuming the mission.
- **Pruning (bounded, low priority).** My prior is that this is a weak lever here, so test it cheaply and move on. Unstructured sparsity gives no speedup in dense-kernel engines; only pursue it if you find a current CPU runtime that genuinely exploits sparse weights for LLMs. The comparison worth making is depth pruning: drop a few layers from a Qwen model without any fine-tuning (there is no budget for healing on this machine) and ask whether it beats the next smaller Qwen, or a lower-bit quantization of the same model, at equal speed. Check whether the engine supports early exit or layer skipping as a self-drafting form of speculative decoding. Separately, measure prompt pruning (compressing or trimming input context before inference) on the long-prompt workload, since fewer input tokens is a direct saving where prompt processing dominates; report its quality cost like any other lossy change.
- **Throughput mode.** If several requests are served at once, how do parallel slots and continuous batching change total tokens per second versus per-request latency?

Add levers I have not thought of if your reading turns them up. Remove ones that turn out not to apply.

## What to measure

For every configuration: prompt-processing tokens per second, generation tokens per second, time to first token, peak resident memory, and model load time. Report generation and prompt processing separately, since they are bottlenecked by different things and a change often helps one and not the other.

Use a small fixed workload set that covers different shapes: a short chat turn, a long-prompt/short-answer task (a few thousand tokens of context), a long generation, a code-completion task, and a structured-extraction task. Fix prompts, seeds, sampling settings, and output lengths so runs are comparable.

Quality is a guardrail, not a goal. For any change that can alter outputs (quantization, a different engine's conversion, thinking off), run a small fixed evaluation — perplexity or KL divergence against the highest-precision reference you can fit, plus a few dozen task items with checkable answers — and report the cost alongside the speedup. A configuration that is 30% faster and measurably dumber is a trade-off to present, not a win to declare.

## Measurement discipline

This is a shared, virtualized 4-core machine, so naive numbers will be noisy and occasionally wrong. That makes discipline the difference between a report I can act on and one I cannot.

- **One measurement at a time, on an otherwise idle machine.** This is the main constraint on how you orchestrate. Two subagents benchmarking concurrently, or one benchmarking while another compiles, silently corrupt each other's numbers. Enforce it mechanically (a lock file that the harness acquires, for instance) rather than by asking agents to be careful. Reading, coding, and literature work can run in parallel freely; anything that burns CPU is serialized.
- Warm up, repeat each configuration several times, and report the median with the spread. Rerun the baseline periodically through the session to detect drift in the host; if the baseline moves, results from that window are suspect.
- Treat differences inside the noise band as no difference, and say so.
- Every result row records the engine commit, build flags, model file and checksum, the exact command line, and the raw log path. Raw logs are kept in the repo (or their essential lines are, if they are large). A number without provenance does not go in the report.
- Change one thing at a time against the baseline, then test the promising changes in combination, since gains from different levers frequently do not multiply.

## Working with your subagents

You are the only one who holds the whole picture. A subagent knows only what you write in its brief, so each brief needs the goal and what it serves, the exact scope, what done looks like, what you already know, the measurement rules that apply, and the shape you want the answer in. Ask them to separate what they measured or read directly from what they inferred.

A reasonable division of labor:

- **You** decide what to test next, design each experiment, read the raw results, decide what they mean, and write the conclusions. Do not delegate interpretation.
- **Sonnet** takes work that needs judgment within a bounded scope: building the benchmark harness, getting an alternative engine running end to end and converting a model for it, owning one experiment track and reporting back with data, or reading through an engine's source and issue tracker to answer a specific technical question.
- **Haiku** takes mechanical work with a fully specified procedure: running a predefined matrix of commands through the harness, extracting numbers from logs into the results table, fetching a documentation page and pulling out the flags, verifying checksums.

Subagent reports are evidence, not truth. A subagent can report success on something that did not happen, or summarize a number it misread. Before a result becomes a headline finding, look at the raw log yourself, and rerun the top handful of configurations in your own hands at the end. Verification should scale with how much the finding matters: a flat result on a minor flag needs a glance, the claimed best configuration needs an independent reproduction.

Do not spawn agents for the sake of it. A single-command check is quicker to run yourself than to brief.

## Things to be careful about

- Flags, build options, and feature support in these engines change monthly. Check the current source, `--help` output, and documentation for anything you are about to rely on, rather than working from memory. This matters more than usual for the low-level details (which kernels exist for which instruction sets, how AMX paths are enabled); verify them empirically against the built binary.
- Blog benchmarks are leads, not results. Cite them as motivation for an experiment and report what you measured.
- Absolute numbers from a 4-vCPU VM will not transfer to other machines; rankings and ratios often do. Say which findings you expect to generalize (to a desktop CPU, to a machine with a GPU or Apple Silicon) and which are specific to this box, and mark that as your judgment.
- Do not let the mission stall on one stubborn engine or one failed download. Record the obstacle, move to the next lever, and come back if time allows.
- Stop when the major levers are measured and the remaining ideas each plausibly offer less than about 10%. A finished, honest report on eight levers is worth more than an unfinished one on fifteen.

## What to hand back

1. `REPORT.md` at the repo root, written for someone who has not followed the session. It opens with the answer: the recommended configuration for this machine, the exact commands to reproduce it, and its speedup over baseline with the quality cost. Then a ranked table of levers (what it is, measured effect on prompt and generation speed, memory effect, quality effect, effort to adopt, confidence). Then what did not help, briefly, so nobody retries it. Then what could not be tested and why, and what you would test next with more time or different hardware.
2. The benchmark harness, runnable with one command on a fresh machine, with a README covering how to add a model, an engine, or a configuration.
3. The raw results as machine-readable files, plus the logs behind them.

In the final message to me, lead with the recommended configuration and the headline numbers, state plainly anything that failed or was skipped, and list any decisions you need from me.
