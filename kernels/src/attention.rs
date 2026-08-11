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
    let batch_i32 = batch as i32;
    let num_heads_i32 = num_heads as i32;
    let num_heads_k_i32 = num_heads_k as i32;
    let seqlen_q_i32 = seqlen_q as i32;
    let seqlen_k_i32 = seqlen_k as i32;
    let head_dim_i32 = head_dim as i32;

    unsafe {
        flash_attention_bf16(q, k, v, out, softmax_lse_out, batch_i32, num_heads_i32, num_heads_k_i32, seqlen_q_i32, seqlen_k_i32, head_dim_i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuBuffer;

    // bf16 is the top 16 bits of an f32's bit pattern (same sign/exponent,
    // truncated mantissa) -- no `half` crate needed just for test data,
    // since the values below (small integers/simple fractions) are exact
    // in bf16 regardless of truncation vs. round-to-nearest.
    fn f32_to_bf16_bits(f: f32) -> u16 {
        (f.to_bits() >> 16) as u16
    }

    fn bf16_bits_to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    fn bf16_to_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| f32_to_bf16_bits(v).to_le_bytes())
            .collect()
    }

    fn bytes_to_bf16(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    }

    #[test]
    fn attention_with_one_kv_position_returns_v_unchanged() {
        // softmax over a single score is always [1.0], regardless of what
        // that score actually is -- so with seqlen_k=1, attention output
        // is exactly 1.0 * V = V, no matter what Q/K contain. head_dim
        // must be 128: that's the only kernel variant compiled
        // (flash_fwd_hdim128_bf16_causal_sm80.cu).
        let (batch, num_heads, num_heads_k, seqlen_q, seqlen_k, head_dim) =
            (1usize, 1usize, 1usize, 1usize, 1usize, 128usize);

        let q_data: Vec<f32> = vec![0.0; head_dim]; // irrelevant to the result, see above
        let k_data: Vec<f32> = vec![0.0; head_dim]; // irrelevant to the result, see above
        let v_data: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1 - 6.0).collect();

        let q_bytes = bf16_to_bytes(&q_data);
        let k_bytes = bf16_to_bytes(&k_data);
        let v_bytes = bf16_to_bytes(&v_data);

        let mut q_buf = GpuBuffer::new(q_bytes.len()).unwrap();
        let mut k_buf = GpuBuffer::new(k_bytes.len()).unwrap();
        let mut v_buf = GpuBuffer::new(v_bytes.len()).unwrap();
        let out_buf = GpuBuffer::new(v_bytes.len()).unwrap();
        let lse_buf = GpuBuffer::new(batch * num_heads * seqlen_q * 4).unwrap();
        assert!(q_buf.copy_from_host(&q_bytes));
        assert!(k_buf.copy_from_host(&k_bytes));
        assert!(v_buf.copy_from_host(&v_bytes));

        let ok = flash_attention_bf16_safe(
            q_buf.ptr() as *const u16,
            k_buf.ptr() as *const u16,
            v_buf.ptr() as *const u16,
            out_buf.ptr() as *mut u16,
            lse_buf.ptr() as *mut f32,
            batch,
            num_heads,
            num_heads_k,
            seqlen_q,
            seqlen_k,
            head_dim,
        );
        assert!(ok);

        let mut out_bytes = vec![0u8; v_bytes.len()];
        assert!(out_buf.copy_to_host(&mut out_bytes));
        let out = bytes_to_bf16(&out_bytes);

        for (i, (&got, &v)) in out.iter().zip(v_data.iter()).enumerate() {
            // Compare against v ROUND-TRIPPED through bf16, not the raw f32
            // -- v was already truncated to bf16 before upload (v_bytes),
            // so that's the only value the kernel could possibly have
            // passed through unchanged, not the original f32.
            let expected = bf16_bits_to_f32(f32_to_bf16_bits(v));
            assert!(
                (got - expected).abs() < 1e-2, // bf16 has ~2-3 decimal digits of precision
                "index {i}: expected {expected}, got {got}"
            );
        }
    }
}
