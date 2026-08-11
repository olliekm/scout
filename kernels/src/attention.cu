// Real FlashAttention-2, via the vendored kernels in vendor/flash_attn/ --
// see vendor/flash_attn/README.md for what's vendored and why. This file
// is OUR OWN wrapper: it builds a Flash_fwd_params (a plain C struct, no
// PyTorch types -- see flash.h) directly from raw device pointers, then
// calls into the same run_mha_fwd dispatcher flash_api.cpp itself uses,
// bypassing their PyTorch tensor-unwrapping layer entirely.
//
// Scope of this first version: contiguous (non-paged) KV cache, causal
// masking, GQA (h != h_k), bf16 data, no dropout/alibi/rotary (RoPE is
// already applied separately by rope.cu before this runs -- see
// engine's forward-pass orchestration once that exists). Paged KV cache
// (block_table) is a real follow-up: Flash_fwd_params already has the
// fields for it, this version just doesn't populate them yet.
//
// Units, confirmed directly from flash-attention's own set_params_fprop
// (not guessed): ALL strides are in ELEMENTS, not bytes. For a
// contiguous tensor shaped [batch, seqlen, heads, head_dim]:
//   row_stride  (between consecutive tokens)  = heads * head_dim
//   head_stride (between consecutive heads)   = head_dim
//   batch_stride (between consecutive batches) = seqlen * heads * head_dim

#include <cuda_bf16.h>
#include <cmath>

#include <cutlass/numeric_types.h> // for cutlass::bfloat16_t, used below
#include "flash.h"

using namespace FLASH_NAMESPACE;

// Declared in flash.h, explicitly instantiated for (bf16, head_dim=128,
// causal=true) in the two vendored flash_fwd_*hdim128_bf16_causal_sm80.cu
// files -- this dispatcher itself (copied from flash_api.cpp, which is
// otherwise off-limits since the REST of that file needs libtorch) picks
// between them based on params.num_splits.
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel = false) {
    if (params.num_splits <= 1 && !force_split_kernel) {
        run_mha_fwd_<cutlass::bfloat16_t, 128, true>(params, stream);
    } else {
        run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, true>(params, stream);
    }
}

extern "C" {

// q: [batch, seqlen_q, num_heads,   head_dim]  (bf16, device pointer)
// k: [batch, seqlen_k, num_heads_k, head_dim]  (bf16, device pointer)
// v: [batch, seqlen_k, num_heads_k, head_dim]  (bf16, device pointer)
// out:            [batch, seqlen_q, num_heads, head_dim]  (bf16, device pointer, caller-allocated)
// softmax_lse_out: [batch, num_heads, seqlen_q] (f32, device pointer, caller-allocated scratch --
//                   FlashAttention needs this internally regardless of whether the caller
//                   cares about the actual log-sum-exp values)
// All tensors contiguous in the layout above -- no custom strides yet.
//
// YOUR CODE HERE, in four stages:
//   1. `Flash_fwd_params params = {};` (zero-init -- this default-disables
//      everything this version doesn't use: dropout, alibi, rotary,
//      varlen cu_seqlens, paged block_table, all stay null/0/false)
//   2. Set the fields set_params_fprop always sets:
//        params.is_bf16 = true;
//        params.q_ptr = (void*)q;  params.k_ptr = (void*)k;  params.v_ptr = (void*)v;
//        params.o_ptr = (void*)out;
//        params.q_row_stride = num_heads * head_dim;      (and same pattern for k/v/o --
//        params.q_head_stride = head_dim;                  k/v use num_heads_k, not num_heads)
//        params.q_batch_stride = seqlen_q * num_heads * head_dim;   (same pattern for k/v/o)
//        params.softmax_lse_ptr = (void*)softmax_lse_out;
//   3. Set the dimensions and GQA ratio:
//        params.b = batch;  params.h = num_heads;  params.h_k = num_heads_k;
//        params.h_h_k_ratio = num_heads / num_heads_k;
//        params.d = head_dim;  params.d_rounded = head_dim;  (128 is already a multiple of 32,
//                                                              so rounding is a no-op here)
//        params.seqlen_q = seqlen_q;  params.seqlen_k = seqlen_k;
//        params.seqlen_q_rounded = ((seqlen_q + 127) / 128) * 128;  (round UP to a multiple of 128,
//        params.seqlen_k_rounded = ((seqlen_k + 127) / 128) * 128;   the pattern flash_api.cpp uses)
//   4. Set the softmax scale, dropout (disabled), and causal flag, then launch:
//        float scale = 1.0f / sqrtf((float)head_dim);
//        params.scale_softmax = scale;
//        params.scale_softmax_log2 = scale * (float)M_LOG2E;
//        params.p_dropout = 1.0f;       // 1.0 = keep everything, i.e. no dropout
//        params.is_causal = true;
//        run_mha_fwd(params, /*stream=*/0);
//        return cudaGetLastError() == cudaSuccess;
bool flash_attention_bf16(
    const void* q, const void* k, const void* v, void* out, void* softmax_lse_out,
    int batch, int num_heads, int num_heads_k, int seqlen_q, int seqlen_k, int head_dim
) {
    Flash_fwd_params params = {};

    params.is_bf16 = true;
    params.q_ptr = (void*)q;
    params.k_ptr = (void*)k;
    params.v_ptr = (void*)v;
    params.o_ptr = out;

    params.q_row_stride = num_heads * head_dim;
    params.k_row_stride = num_heads_k * head_dim;
    params.v_row_stride = num_heads_k * head_dim;
    params.o_row_stride = num_heads * head_dim;

    params.q_head_stride = head_dim;
    params.k_head_stride = head_dim;
    params.v_head_stride = head_dim;
    params.o_head_stride = head_dim;

    params.q_batch_stride = seqlen_q * num_heads * head_dim;
    params.k_batch_stride = seqlen_k * num_heads_k * head_dim;
    params.v_batch_stride = seqlen_k * num_heads_k * head_dim;
    params.o_batch_stride = seqlen_q * num_heads * head_dim;

    params.softmax_lse_ptr = softmax_lse_out;

    params.b = batch;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.d = head_dim;
    params.d_rounded = head_dim;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = ((seqlen_q + 127) / 128) * 128;
    params.seqlen_k_rounded = ((seqlen_k + 127) / 128) * 128;

    float scale = 1.0f / sqrtf((float)head_dim);
    params.scale_softmax = scale;
    params.scale_softmax_log2 = scale * (float)M_LOG2E;
    params.p_dropout = 1.0f;
    params.is_causal = true;

    run_mha_fwd(params, /*stream=*/0);
    return cudaGetLastError() == cudaSuccess;
}

} // extern "C"
