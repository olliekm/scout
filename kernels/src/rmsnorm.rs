//! RMSNorm via a hand-written CUDA kernel -- see rmsnorm.cu's header for
//! the full formula/parallel-reduction rationale, and why this one is
//! hand-rolled rather than integrated (unlike matmul/attention, RMSNorm
//! isn't a solved problem worth pulling in a library for).

unsafe extern "C" {
    fn rmsnorm_f32(
        x: *const f32,
        weight: *const f32,
        out: *mut f32,
        num_tokens: i32,
        hidden_size: i32,
        eps: f32,
    ) -> bool;
}

/// Qwen2.5-Coder-7B-Instruct's `rms_norm_eps`, per its HF config.json --
/// hardcoded alongside `NUM_LAYERS` in `engine/src/model.rs`, same
/// single-target-model reasoning: no other checkpoint loads through this
/// path, so there's nothing to make this configurable for yet.
pub const RMS_NORM_EPS: f32 = 1e-6;

/// Named `_safe` (not just `rmsnorm_f32`) because a plain free function
/// can't share a name with the `extern "C"` declaration above -- unlike
/// `CublasHandle::matmul_f32`, which could reuse its FFI function's name
/// since a method call (`self.matmul_f32(...)`) and a bare function call
/// (`matmul_f32(...)`) are different namespace paths, this is a bare
/// function too, so it needs a genuinely different name.
///
/// Apply RMSNorm to `num_tokens` rows of `hidden_size` f32 elements each
/// -- `x`, `weight`, `out` are all device pointers (already uploaded via
/// `GpuBuffer::copy_from_host` or similar). No aliasing check is done on
/// the Rust or CUDA side, so pass distinct buffers for `x` and `out`
/// rather than relying on in-place behavior.
///
/// Steps:
///   1. Cast num_tokens/hidden_size from usize to i32 (same reasoning as
///      `CublasHandle::matmul_f32`'s m/n/k cast -- these dimensions are
///      nowhere near i32::MAX).
///   2. unsafe { rmsnorm_f32(x, weight, out, num_tokens_i32, hidden_size_i32, eps) }
///      SAFETY: caller guarantees x/weight/out point at valid device
///      memory of the right size (x and out: num_tokens * hidden_size
///      floats each; weight: hidden_size floats).
///   3. Return the bool rmsnorm_f32 gives back directly.
pub fn rmsnorm_f32_safe(
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    num_tokens: usize,
    hidden_size: usize,
    eps: f32,
) -> bool {
    let num_tokens_i32 = num_tokens as i32;
    let hidden_size_i32 = hidden_size as i32;
    unsafe { rmsnorm_f32(x, weight, out, num_tokens_i32, hidden_size_i32, eps) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuBuffer;

    fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-3, "expected {expected:?}, got {actual:?}");
        }
    }

    #[test]
    fn rmsnorm_computes_a_correct_result_per_token() {
        // hidden_size = 4 is deliberately smaller than THREADS_PER_BLOCK
        // (256 in rmsnorm.cu) -- exercises the case where most threads in
        // a block have an EMPTY grid-stride loop (only 4 of 256 threads
        // do any real work), not just the "many elements per thread"
        // case a larger hidden_size would test.
        //
        // Two tokens with different sums of squares (25 and 100) are
        // chosen so their RMS values come out exact (2.5 and 5.0), making
        // the expected output computable by hand instead of needing a
        // reference implementation to compare against.
        let (num_tokens, hidden_size) = (2usize, 4usize);
        let x_data = f32_to_bytes(&[
            0.0, 0.0, 3.0, 4.0, // token 0: sum of squares = 25, rms = 2.5
            6.0, 8.0, 0.0, 0.0, // token 1: sum of squares = 100, rms = 5.0
        ]);
        let weight_data = f32_to_bytes(&[2.0, 1.0, 1.0, 0.5]);
        let expected = vec![
            0.0, 0.0, 1.2, 0.8, // [0/2.5*2, 0/2.5*1, 3/2.5*1, 4/2.5*0.5]
            2.4, 1.6, 0.0, 0.0, // [6/5*2, 8/5*1, 0/5*1, 0/5*0.5]
        ];

        let mut x_buf = GpuBuffer::new(x_data.len()).unwrap();
        let mut weight_buf = GpuBuffer::new(weight_data.len()).unwrap();
        let out_buf = GpuBuffer::new(x_data.len()).unwrap();
        assert!(x_buf.copy_from_host(&x_data));
        assert!(weight_buf.copy_from_host(&weight_data));

        let ok = rmsnorm_f32_safe(
            x_buf.ptr() as *const f32,
            weight_buf.ptr() as *const f32,
            out_buf.ptr() as *mut f32,
            num_tokens,
            hidden_size,
            1e-6,
        );
        assert!(ok);

        let mut out_bytes = vec![0u8; x_data.len()];
        assert!(out_buf.copy_to_host(&mut out_bytes));
        assert_close(&bytes_to_f32(&out_bytes), &expected);
    }
}
