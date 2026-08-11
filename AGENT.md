# Scout

## What this is

Scout is a from-scratch LLM inference engine written in Rust. It's a portfolio project built to demonstrate deep ML systems / inference optimization knowledge for internship recruiting (primary target: Baseten Model Performance Engineer, Summer 2027 cycle; also relevant to RBC Borealis ML Software Engineer work). The whole point is depth over breadth: rather than wrapping vLLM or an existing serving framework, every core piece (scheduler, KV cache allocator, batching logic) is implemented from the ground up.

Status: Steps 1-2 (naive baseline, static batching) prototyped in Python against Qwen2.5-0.5B-Instruct for fast iteration — throwaway prototypes per the tech stack decision below, not the real engine. Rust engine work not yet started. This doc is the context/architecture reference to build from.

## Goals

- Build genuine understanding of how production inference engines (vLLM, SGLang, TensorRT-LLM) work internally, not just surface-level familiarity
- Have a concrete, benchmarked system to talk through in interviews — acceptance rates, throughput numbers, design tradeoffs
- Serve as the natural "went all the way down" project alongside Parsec (LLM orchestration, one layer up) and Ascent (distributed parameter server, adjacent systems work)

## Tech stack

- **Core engine**: Rust
- **Python bindings**: PyO3, exposing a Python API (`engine.generate()`, `generate_batch()`, streaming)
- **Serving interface**: gRPC
- **GPU kernels**: CUDA, accessed via FFI from Rust. Scope decision — integrate FlashAttention-2 rather than rewrite attention kernels from scratch; hand-write kernels only where it's the point of a specific milestone (e.g. a basic matmul/attention kernel for the "naive" baseline), not to reinvent well-optimized ops.
- **Model format**: load weights from safetensors (GGUF as a stretch goal)
- **Target model**: Qwen2.5-Coder-7B-Instruct (HF: `Qwen/Qwen2.5-Coder-7B-Instruct`) (`Qwen2ForCausalLM` architecture — 28 layers, hidden_size 3584, GQA with 28 query heads / 4 KV heads, 32768 max context). Prototyping/Step 1-2 baselines used Qwen2.5-0.5B-Instruct for fast iteration; those numbers don't transfer to the 7B target and were re-baselined once the model size changed.

## Architecture components

1. **Model loader** — reads safetensors weights, builds the model graph
2. **CUDA kernel layer** — matrix ops and attention, called via FFI from Rust; Rust owns orchestration, CUDA owns the actual GPU math
3. **KV cache allocator** — the core systems problem. Handles variable-length sequences, block-based (paged) allocation, eviction under memory pressure. Conceptually similar to OS virtual memory management.
4. **Scheduler** — continuous batching logic: treats the batch as a rolling window, inserting new requests as soon as a slot frees up, instead of naive batching that waits for the whole batch to finish
5. **Speculative decoding coordinator** — n-gram/prompt-lookup speculative decoding: cache n-grams seen so far (in generated output and/or prompt context), and when the current token sequence matches a previously-seen n-gram prefix, propose the cached continuation as draft tokens; target model verifies all K in one forward pass, coordinator rewinds on rejection. Chosen over a trained draft model (two-model spec decoding) or EAGLE (trained hidden-state draft head) specifically because it requires no separate training pipeline — it's an algorithm implementable directly in the Rust scheduler, consistent with this project's "integrate/orchestrate, don't retrain" philosophy. Works best on repetitive text, which code (imports, boilerplate, repeated identifiers) has a lot of — a good match for the coding-model target.
6. **gRPC serving layer** — external interface
7. **PyO3 bindings** — Python-facing API on top of the Rust engine

## Build roadmap (staged, in order)

1. Naive inference loop (single sequence, no batching, no cache reuse tricks beyond basic KV caching)
2. Static batching
3. Paged KV cache (block-based allocator)
4. Continuous batching scheduler
5. Fused attention kernels (integrate FlashAttention-2 via FFI)
6. Speculative decoding (draft + verify loop)
7. Quantization: weight-only 4-bit + fused dequant-GEMM (integrate an existing kernel — AWQ/GPTQ/Marlin — via FFI first; hand-rolling the fused kernel is a further stretch goal on top, once the integrated version works)

Known gap vs. what Baseten's stack actually needs (per their posted role): **chunked prefill** is not on this roadmap yet and should be added as a stretch milestone after continuous batching — it's the mechanism for interleaving long prefill computations with ongoing decode steps so one big new request doesn't stall everyone else's generation.

## Paged KV cache: fragmentation benchmark (step 3)

`engine/examples/fragmentation_report.rs` (`cargo run --example fragmentation_report -p engine`) measures the actual memory-efficiency win paging gives over naive reservation — pure Rust bookkeeping against the real `BlockAllocator`, no GPU or model needed, so it runs anywhere including outside the pod.

- **Method**: sample 64 sequence lengths from a seeded `LogNormal(mu=ln(150), sigma=1.2)` distribution (median ~150 tokens, long tail — deliberately skewed like real chat/code workloads, mirroring why `bench_sharegpt.py` uses real ShareGPT length variance instead of a uniform spread). Compare two strategies against that same workload: **naive** (`num_sequences * ceil(max_seq_len / block_size)` — every sequence reserves the full 32768-token max context upfront, Qwen2.5-Coder-7B-Instruct's actual context length) vs. **paged** (drives the real `BlockAllocator::allocate_block_for` per sequence, growing each sequence's block list lazily to only `ceil(true_length / block_size)` blocks, block_size = 16 tokens).
- **Result**: naive requires 131,072 blocks; paged uses only 955 — **99.27% of naive's worst-case reservation goes unused**.
- **Why this matters**: this is the internal-fragmentation result the vLLM paper's paging design is built to solve — reserving worst-case space per request wastes almost all of it in practice, since most real requests are far shorter than the max context window. It's the concrete "why paging" story for step 3, parallel to how the INT8 finding below is the concrete "why not bitsandbytes" story for step 7.

## Quantization: step 7 on the roadmap (see also step-7 bullet above)

First attempt (`bitsandbytes` int8) was tried and dropped for being slower, not faster — kept below as a record of what was learned. Revisited as step 7: weight-only 4-bit quantization with a kernel that fuses dequantization into the GEMM (AWQ, GPTQ, or Marlin specifically, since Marlin targets exactly this fused-dequant-int4-GEMM case and is what vLLM itself uses), integrated via FFI the same way FlashAttention-2 is planned to be — link against existing kernel source, don't reimplement it, per the "don't reinvent optimized GPU kernels" principle. Hand-rolling the fused kernel is a further stretch goal after that integration works.

- Python prototype (`server_static_batch.py`, `QUANTIZE=int8` env var) integrated `bitsandbytes` `LLM.int8()` against Qwen2.5-Coder-7B-Instruct on the A40 dev pod.
- **HumanEval pass@1**: fp16 baseline 0.805 (164/164 problems, `bench/results/humaneval_fp16_full.jsonl` — not directly comparable to Qwen's published 88.4, different eval harness/prompt conventions, but a valid same-harness reference point). INT8 run was not completed to full accuracy comparison once the speed finding below made the direction moot.
- **Key finding — INT8 was slower, not faster**: single-request generation latency went from 4.23s (fp16) to 11.25s (int8) for the same prompt/max_tokens, a ~2.7x *slowdown*. Root cause: `bitsandbytes`' `LLM.int8()` is a mixed-precision decomposition scheme (outlier activation columns routed through fp16 matmuls, non-outliers through int8) designed to reduce memory footprint so large models fit on smaller GPUs — it is not optimized for inference speed. On a GPU with ample headroom for the model (A40, 7B model), the decomposition overhead outweighs any memory-bandwidth savings.
- **Conclusion**: `bitsandbytes` was dropped, since it works against the project's core goal (fast inference / throughput) on hardware with no memory pressure to justify its tradeoff — but the direction wasn't abandoned, just the tool. See step 7 above for the revisited plan.

## Design principles for this project

- Prioritize the scheduler and KV cache allocator — that's where the real systems engineering is, and where interview questions will dig deepest
- Don't reinvent optimized GPU kernels wholesale; the value is in the orchestration/systems layer, not out-competing FlashAttention
- Every stage should be independently benchmarkable (tokens/sec, acceptance rate once spec decoding lands) so there's a clear before/after story at each milestone
- One repo, multiple folders (engine, kernels, bindings, serving) — not split across repos, since the binding between CUDA kernel and Rust engine is part of the story

## Repo / naming

- Repo name: `scout`
- Resume-facing description should lead with "LLM Inference Engine" (descriptive), with Scout as the repo's internal name

## Possible future direction: tab-autocomplete serving mode

Not in current scope — a narrative/stretch idea, not a roadmap commitment. Scout's infrastructure (model loader, CUDA FFI layer, KV cache allocator, and especially n-gram speculative decoding — code's high local repetition, e.g. imports and boilerplate, is exactly what n-gram matching exploits) transfers well to a tab-autocomplete / code-completion serving backend, which is a real, recognizable product shape (Copilot-style).

The core roadmap as scoped optimizes for throughput under concurrency (batching, paging, continuous batching). A real autocomplete pipeline would instead center on:
- **Single-request TTFT as the primary metric**, not aggregate tok/s — users expect a suggestion in well under 100ms per keystroke, not after a batch fills
- **Fill-in-the-middle (FIM) prompting** — needs prefix *and* suffix context around the cursor, not a left-to-right chat prompt; Qwen2.5-Coder supports FIM tokens, but the serving layer's prompt construction would need to change
- **Short, bursty completions** — a few tokens to a line, not 128-256 tokens, which likely means much tighter batch-formation windows than the current design
- **Cancellation/preemption** — if the user keeps typing past where a completion started, the in-flight generation should be cancelled rather than wasting GPU cycles; not currently a scheduler feature on this roadmap

Worth keeping as a talking point ("here's how the same core engine would need to adapt for a different serving workload") rather than pulling it into current scope, since it pulls toward a different center of gravity (single-stream latency + preemption) than "prioritize the scheduler and KV cache allocator for throughput," which is the current design principle.

## Possible future direction: hybrid n-gram + lightweight draft head speculative decoding

Not in current scope — a stretch idea layered on top of step 6, not a roadmap commitment. The intuition: n-gram/prompt-lookup drafting (the chosen step-6 approach) can only propose a continuation that has appeared *verbatim* before in context — excellent for code's highly repetitive patterns (imports, boilerplate, repeated identifiers), but blind to anything genuinely novel. A small trained draft head (EAGLE-style — predicting from the target model's own hidden states, not an independent model) generalizes past exact repeats, so it can still propose plausible tokens for the "in-between" content n-gram can't match (new variable names, natural-language comments, first-time logic). The two failure modes barely overlap, so running both and using whichever is currently confident/available should push acceptance rate higher than either alone — this is a real direction in the spec-decoding literature (hybrid/ensemble drafting), not a novel-but-unproven idea.

Why it's a stretch goal and not step 6 itself: it reintroduces exactly what EAGLE was rejected for originally (see the speculative decoding rationale in "Architecture components" above) — a training pipeline. N-gram needs zero training, matching this project's "integrate/orchestrate, don't retrain" philosophy; a draft head needs training data (aligned target-model hidden states → next-token pairs), an actual training loop, and export/load infrastructure for its weights, none of which exists anywhere in Scout and is a genuinely different category of work (ML training) than everything else on the roadmap (systems/orchestration). It would also add real scheduler complexity — deciding per-step which drafter to trust, or running both and reconciling disagreements.

Worth keeping as a talking point / extension idea for interviews ("here's how I'd push acceptance rate further") once n-gram spec decoding itself is built and benchmarked, not before.

## Related projects (for context, not part of Scout itself)

- **Parsec** — LLM orchestration library (Python), the layer above where Scout sits
- **Ascent** — distributed logistic regression parameter server in C over TCP sockets, adjacent systems work