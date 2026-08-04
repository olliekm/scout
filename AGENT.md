# Scout

## What this is

Scout is a from-scratch LLM inference engine written in Rust. It's a portfolio project built to demonstrate deep ML systems / inference optimization knowledge for internship recruiting (primary target: Baseten Model Performance Engineer, Summer 2027 cycle; also relevant to RBC Borealis ML Software Engineer work). The whole point is depth over breadth: rather than wrapping vLLM or an existing serving framework, every core piece (scheduler, KV cache allocator, batching logic) is implemented from the ground up.

Status: not started yet. This doc is the context/architecture reference to build from.

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

## Architecture components

1. **Model loader** — reads safetensors weights, builds the model graph
2. **CUDA kernel layer** — matrix ops and attention, called via FFI from Rust; Rust owns orchestration, CUDA owns the actual GPU math
3. **KV cache allocator** — the core systems problem. Handles variable-length sequences, block-based (paged) allocation, eviction under memory pressure. Conceptually similar to OS virtual memory management.
4. **Scheduler** — continuous batching logic: treats the batch as a rolling window, inserting new requests as soon as a slot frees up, instead of naive batching that waits for the whole batch to finish
5. **Speculative decoding coordinator** — draft model generates K candidate tokens, target model verifies all K in one forward pass; needs to keep two models' KV caches in sync and correctly "rewind" state on rejected tokens
6. **gRPC serving layer** — external interface
7. **PyO3 bindings** — Python-facing API on top of the Rust engine

## Build roadmap (staged, in order)

1. Naive inference loop (single sequence, no batching, no cache reuse tricks beyond basic KV caching)
2. Static batching
3. Paged KV cache (block-based allocator)
4. Continuous batching scheduler
5. Fused attention kernels (integrate FlashAttention-2 via FFI)
6. Speculative decoding (draft + verify loop)

Known gap vs. what Baseten's stack actually needs (per their posted role): **chunked prefill** is not on this roadmap yet and should be added as a stretch milestone after continuous batching — it's the mechanism for interleaving long prefill computations with ongoing decode steps so one big new request doesn't stall everyone else's generation.

## Design principles for this project

- Prioritize the scheduler and KV cache allocator — that's where the real systems engineering is, and where interview questions will dig deepest
- Don't reinvent optimized GPU kernels wholesale; the value is in the orchestration/systems layer, not out-competing FlashAttention
- Every stage should be independently benchmarkable (tokens/sec, acceptance rate once spec decoding lands) so there's a clear before/after story at each milestone
- One repo, multiple folders (engine, kernels, bindings, serving) — not split across repos, since the binding between CUDA kernel and Rust engine is part of the story

## Repo / naming

- Repo name: `scout`
- Resume-facing description should lead with "LLM Inference Engine" (descriptive), with Scout as the repo's internal name

## Related projects (for context, not part of Scout itself)

- **Parsec** — LLM orchestration library (Python), the layer above where Scout sits
- **Ascent** — distributed logistic regression parameter server in C over TCP sockets, adjacent systems work