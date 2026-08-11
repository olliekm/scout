# Vendored: FlashAttention-2

Source: https://github.com/Dao-AILab/flash-attention (`csrc/flash_attn/src/`),
commit tracked in `AGENT.md`'s attention integration notes. License:
BSD-3-Clause (see upstream repo).

These files are copied unmodified except where noted inline. They are the
CUDA kernel implementation only -- the PyTorch-facing wrapper
(`csrc/flash_attn/flash_api.cpp`, which takes `at::Tensor` and requires
libtorch) is deliberately NOT vendored. Scout's own `kernels/src/attention.cu`
calls directly into `run_mha_fwd`/`run_mha_fwd_splitkv_dispatch`
(declared in `flash.h`, explicitly instantiated for this model's exact
config in `flash_fwd_hdim128_bf16_causal_sm80.cu` /
`flash_fwd_split_hdim128_bf16_causal_sm80.cu`) using `Flash_fwd_params`
directly -- a plain C-style struct with raw device pointers, no PyTorch
types involved.

Only the head_dim=128, bf16, causal kernel variant is vendored (matching
Qwen2.5-Coder-7B-Instruct's config) -- not the full matrix of
dtype/head_dim/architecture combinations upstream ships (fp16, other head
dims, non-causal, backward/training kernels are all irrelevant here and
excluded).

One real dependency substitution: `flash_fwd_launch_template.h` includes
`<c10/cuda/CUDAException.h>` (a PyTorch/libtorch header) for two trivial
error-check macros. `kernels/vendor/stubs/c10/cuda/CUDAException.h`
replaces it with a torch-free equivalent -- see that file's own comment
for why. `FLASHATTENTION_DISABLE_DROPOUT` is defined at compile time
(dropout is training-only, irrelevant for inference) to avoid needing
the ATen/Philox RNG headers dropout support would otherwise pull in.

CUTLASS (the template library these kernels are built on) is NOT vendored
here -- see `kernels/build.rs` for why (27MB/805 files, fetched at build
time instead of committed).
