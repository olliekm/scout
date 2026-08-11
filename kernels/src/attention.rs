//! Real FlashAttention-2 via the vendored kernels in vendor/flash_attn/ --
//! see attention.cu's header for the full picture (units, GQA, scope of
//! this first version). Data is bf16, not f32 -- unlike matmul/rmsnorm/rope,
//! this kernel operates on the same dtype the model's weights actually
//! ship in. Represented on the Rust side as `*const u16`/`*mut u16`: bf16
//! is a 2-byte type (same bit width as u16), and nothing on the Rust side
//! ever needs to interpret those bits as a number -- only the CUDA/CUTLASS
//! code does -- so there's no reason to pull in a dedicated bf16 crate
//! just to pass an opaque pointer through.

unsafe extern "C" {
    fn flash_attention_bf16(
        q: *const u16,
        k: *const u16,
        v: *const u16,
        out: *mut u16,
        softmax_lse_out: *mut f32,
        batch: i32,
        num_heads: i32,
        num_heads_k: i32,
        seqlen_q: i32,
        seqlen_k: i32,
        head_dim: i32,
    ) -> bool;
}

/// Run FlashAttention-2 forward. `q`/`k`/`v`/`out` are bf16 device buffers
/// (contiguous, `[batch, seqlen, heads, head_dim]` layout -- `heads` is
/// `num_heads` for q/out, `num_heads_k` for k/v, since Qwen2's GQA has
/// fewer KV heads than query heads). `softmax_lse_out` is a caller-
/// allocated f32 scratch buffer of `batch * num_heads * seqlen_q` elements
/// -- required by the kernel internally regardless of whether the caller
/// uses the values.
///
/// Steps:
///   1. Cast every `usize` dimension to `i32` (same reasoning as every
///      other kernel wrapper's dimension casts).
///   2. unsafe { flash_attention_bf16(q, k, v, out, softmax_lse_out, batch_i32, num_heads_i32, num_heads_k_i32, seqlen_q_i32, seqlen_k_i32, head_dim_i32) }
///      SAFETY: caller guarantees every buffer is sized per the layout
///      described above and already resident on the GPU.
///   3. Return the bool flash_attention_bf16 gives back directly.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_bf16_safe(
    q: *const u16,
    k: *const u16,
    v: *const u16,
    out: *mut u16,
    softmax_lse_out: *mut f32,
    batch: usize,
    num_heads: usize,
    num_heads_k: usize,
    seqlen_q: usize,
    seqlen_k: usize,
    head_dim: usize,
) -> bool {
    todo!()
}
