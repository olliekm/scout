//! RoPE (rotary position embeddings) via a hand-written CUDA kernel --
//! see rope.cu's header for the full formula and why `positions` has to
//! be passed explicitly rather than inferred from row index.

unsafe extern "C" {
    fn rope_f32(
        x: *mut f32,
        positions: *const i32,
        num_tokens: i32,
        num_heads: i32,
        head_dim: i32,
        theta: f32,
    ) -> bool;
}

/// Apply RoPE IN-PLACE to `x` -- a device buffer of `num_tokens` rows,
/// each `num_heads * head_dim` f32 elements (Q or K's projection output;
/// never V, which RoPE doesn't touch). `positions` is a device buffer of
/// `num_tokens` i32s, one real sequence position per row -- required
/// because continuous batching means row index and position aren't the
/// same thing (see rope.cu's header comment for why).
///
/// `theta` is passed as a parameter rather than hardcoded here (unlike
/// `NUM_LAYERS`/`RMS_NORM_EPS` elsewhere in this codebase) -- its correct
/// value for Qwen2.5-Coder-7B-Instruct hasn't been confirmed against a
/// real config.json yet. Check that file once you have it on the pod,
/// then consider promoting it to a hardcoded constant here, matching the
/// other two, once verified.
///
/// Named `_safe` for the same reason as `rmsnorm_f32_safe`: a bare
/// function can't share a name with its own `extern "C"` declaration.
///
/// Steps:
///   1. Cast num_tokens/num_heads/head_dim from usize to i32.
///   2. unsafe { rope_f32(x, positions, num_tokens_i32, num_heads_i32, head_dim_i32, theta) }
///      SAFETY: caller guarantees x points at num_tokens * num_heads *
///      head_dim valid device floats, positions at num_tokens valid
///      device i32s.
///   3. Return the bool rope_f32 gives back directly.
pub fn rope_f32_safe(
    x: *mut f32,
    positions: *const i32,
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    theta: f32,
) -> bool {
    let num_tokens_i32 = num_tokens as i32;
    let num_heads_i32 = num_heads as i32;
    let head_dim_i32 = head_dim as i32;
    unsafe {
        rope_f32(x, positions, num_tokens_i32, num_heads_i32, head_dim_i32, theta)
    }
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
    fn rope_at_position_zero_is_identity() {
        // Every rotation angle is pos * freq -- at pos = 0, that's 0
        // regardless of freq, so cos(0)=1/sin(0)=0 makes every rotation a
        // no-op. A free, exact test before needing any hand-computed trig.
        let (num_tokens, num_heads, head_dim) = (1usize, 1usize, 4usize);
        let x_data = f32_to_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let positions_data: Vec<u8> = 0i32.to_le_bytes().to_vec();

        let mut x_buf = GpuBuffer::new(x_data.len()).unwrap();
        let mut pos_buf = GpuBuffer::new(positions_data.len()).unwrap();
        assert!(x_buf.copy_from_host(&x_data));
        assert!(pos_buf.copy_from_host(&positions_data));

        let ok = rope_f32_safe(
            x_buf.ptr() as *mut f32,
            pos_buf.ptr() as *const i32,
            num_tokens,
            num_heads,
            head_dim,
            10000.0,
        );
        assert!(ok);

        let mut out_bytes = vec![0u8; x_data.len()];
        assert!(x_buf.copy_to_host(&mut out_bytes));
        assert_close(&bytes_to_f32(&out_bytes), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_rotates_correctly_at_a_nonzero_position() {
        // head_dim = 4 -> two pairs: (x0, x2) with i=0, (x1, x3) with i=1.
        // theta = 10000, pos = 1:
        //   i=0: freq = theta^0 = 1        -> angle = 1
        //   i=1: freq = theta^-0.5 = 0.01  -> angle = 0.01
        // Expected values are cos(1)/sin(1)/cos(0.01)/sin(0.01) applied by
        // hand to x = [1, 0, 0, 1] (hand-computable trig, not a mystery
        // reference implementation).
        let (num_tokens, num_heads, head_dim) = (1usize, 1usize, 4usize);
        let x_data = f32_to_bytes(&[1.0, 0.0, 0.0, 1.0]);
        let positions_data: Vec<u8> = 1i32.to_le_bytes().to_vec();
        let expected = vec![
            0.5403023, // x0*cos(1) - x2*sin(1) = cos(1)
            -0.0099998, // x1*cos(0.01) - x3*sin(0.01) = -sin(0.01)
            0.8414710, // x0*sin(1) + x2*cos(1) = sin(1)
            0.9999500, // x1*sin(0.01) + x3*cos(0.01) = cos(0.01)
        ];

        let mut x_buf = GpuBuffer::new(x_data.len()).unwrap();
        let mut pos_buf = GpuBuffer::new(positions_data.len()).unwrap();
        assert!(x_buf.copy_from_host(&x_data));
        assert!(pos_buf.copy_from_host(&positions_data));

        let ok = rope_f32_safe(
            x_buf.ptr() as *mut f32,
            pos_buf.ptr() as *const i32,
            num_tokens,
            num_heads,
            head_dim,
            10000.0,
        );
        assert!(ok);

        let mut out_bytes = vec![0u8; x_data.len()];
        assert!(x_buf.copy_to_host(&mut out_bytes));
        assert_close(&bytes_to_f32(&out_bytes), &expected);
    }
}
