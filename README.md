# Scout

A from-scratch LLM inference engine, written in Rust from the ground up —
scheduler, paged KV cache allocator, and batching logic included, rather
than wrapped around an existing serving framework.

Target model: [`Qwen2.5-Coder-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
(28 layers, GQA, 32k context).

## Why

Most "inference engine" projects glue together vLLM or TGI and call it a
day. Scout goes the other direction: every core piece of the serving stack
— admission control, block-based KV cache management, continuous batching —
is implemented by hand, in Rust, down to the CUDA FFI boundary. The point
is depth, not a shipped product.

## Architecture

```
                     ┌─────────────────────┐
   gRPC request  ──▶ │   Scheduler          │  admission control,
                     │   (scheduler.rs)      │  continuous batching
                     └──────────┬───────────┘
                                │ allocates/frees blocks
                     ┌──────────▼───────────┐
                     │   Block Allocator     │  paged KV cache,
                     │  (block_allocator.rs) │  block bookkeeping
                     └──────────┬───────────┘
                                │ backed by
                     ┌──────────▼───────────┐
                     │   CUDA kernels        │  paged GPU buffer,
                     │  (kernels/, FFI)       │  attention, matmul
                     └───────────────────────┘
```

- **`engine/`** — the Rust core: block allocator, scheduler, and the safe
  wrapper around the GPU allocator.
- **`kernels/`** — CUDA, compiled via `nvcc` and linked through FFI. Pod-only;
  gated behind the `gpu` feature since it can't build without an NVIDIA
  toolchain.
- **Python prototypes** (`server.py`, `server_static_batch.py`) — throwaway
  reference implementations of the naive and static-batching baselines,
  used to get real tok/s numbers before committing to the Rust design.

## Roadmap

- [x] Naive inference loop *(Python prototype)*
- [x] Static batching *(Python prototype)*
- [x] Paged KV cache — block-based allocator *(Rust, CPU + GPU-backed)*
- [ ] Continuous batching scheduler — admission control done, per-iteration
      dispatch next
- [ ] Fused attention kernels (FlashAttention-2 via FFI)
- [ ] Speculative decoding (n-gram / prompt-lookup draft + verify)

## Benchmarks

Measured against `Qwen/Qwen2.5-Coder-7B-Instruct` on an A40, fp16, from the
Python prototypes (`bench/results/`) — the reference numbers the Rust engine
is being built to beat.

| Stage | Workload | Result |
|---|---|---|
| Naive loop | single sequence, mixed prompt lengths | 34 tok/s mean · 70ms TTFT (p50 58ms) |
| Static batching | synthetic burst of 8, batch size 4 | 72 tok/s |
| Static batching | ShareGPT burst, 16 prompts, batch size 4 | 108 tok/s |
| Static batching | ShareGPT burst, 32 prompts, batch size 4 | 108 tok/s |
| Correctness | HumanEval pass@1, 164 problems | 0.805 |

INT8 quantization (`bitsandbytes`) was tried and dropped — it made single-request
latency ~2.7x *slower* (4.2s → 11.3s), since `LLM.int8()` trades speed for
memory footprint and this model already fits comfortably on an A40. Details
in `AGENT.md`.

## Building

```sh
cargo build                      # engine, CPU-only paths
cargo test                       # engine + block allocator + scheduler tests
cargo build --features gpu       # requires nvcc; pod only
```

## Status

Early — the Rust engine is at the paged KV cache / admission control stage.
See `AGENT.md` for the full design rationale and staged build plan.
