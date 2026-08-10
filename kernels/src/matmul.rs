//! Basic GEMM (matrix multiply) via cuBLAS -- see matmul.cu's header
//! comment for the full row-major/column-major rationale and the
//! "integrate, don't reinvent" call this mirrors from FlashAttention-2.
//! This is the FFI declaration + safe-wrapper side; the actual cuBLAS
//! calls live in the .cu file.

use std::ffi::c_void;

// cublasHandle_t is an opaque pointer type on the C side -- Rust doesn't
// need to know cuBLAS's internal struct layout, only that it's a handle
// value to hand back unchanged to every cuBLAS call, same idea as
// GpuBuffer's raw pointer in lib.rs.
unsafe extern "C" {
    fn cublas_create_handle() -> *mut c_void;
    fn cublas_destroy_handle(handle: *mut c_void);
    fn matmul_f32(
        handle: *mut c_void,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> bool;
}

/// Owns a cuBLAS handle -- created once, reused across every matmul
/// call, destroyed when dropped. Same ownership shape as `GpuBuffer` in
/// lib.rs: a raw FFI resource wrapped in a struct so `Drop` guarantees
/// cleanup instead of relying on every call site to remember it.
pub struct CublasHandle {
    handle: *mut c_void,
}

impl CublasHandle {
    /// Create a new cuBLAS handle. Returns `None` if cuBLAS failed to
    /// initialize -- mirrors `GpuBuffer::new`'s null-pointer-means-
    /// failure convention.
    ///
    /// Steps:
    ///   1. `let handle = unsafe { cublas_create_handle() };`
    ///      SAFETY: cublas_create_handle is a pure initialization call,
    ///      same reasoning as gpu_alloc_buffer in GpuBuffer::new.
    ///   2. If handle.is_null(), return None.
    ///   3. Otherwise Some(CublasHandle { handle }).
    pub fn new() -> Option<Self> {
        let handle = unsafe { cublas_create_handle() };
        if handle.is_null() { return None; }
        Some(CublasHandle { handle })
    }

    /// Compute C = A * B for row-major f32 matrices already resident on
    /// the GPU: A is m x k, B is k x n, C is m x n. All three pointers
    /// must already point at valid device memory of the right size --
    /// this function does no bounds checking of its own (same
    /// "caller's responsibility" contract as `block_ptr_for` callers
    /// being responsible for a valid seq_id/position).
    ///
    /// Steps:
    ///   1. Cast m/n/k from usize to i32 (cuBLAS's dimension arguments
    ///      are i32, not usize -- `as i32` is fine here, matrix
    ///      dimensions are nowhere near i32::MAX).
    ///   2. unsafe { matmul_f32(self.handle, a, b, c, m, n, k) }
    ///      SAFETY: caller guarantees a/b/c point at valid, appropriately
    ///      sized device memory (document this at the call site).
    ///   3. Return the bool matmul_f32 gives back directly.
    pub fn matmul_f32(&self, a: *const f32, b: *const f32, c: *mut f32, m: usize, n: usize, k: usize) -> bool {
        let m_i32 = m as i32;
        let n_i32 = n as i32;
        let k_i32 = k as i32;
        return unsafe { matmul_f32(self.handle, a, b, c, m_i32, n_i32, k_i32) }
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        // SAFETY: self.handle was returned by cublas_create_handle in
        // `new` and has not been destroyed yet (Drop runs at most once
        // per value, same guarantee GpuBuffer's Drop relies on).
        unsafe { cublas_destroy_handle(self.handle) }
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

    #[test]
    fn creates_and_drops_a_handle() {
        let handle = CublasHandle::new();
        assert!(handle.is_some());
        // handle drops here -- passing is the signal cublas_destroy_handle
        // didn't crash, same "implicit assertion via Drop" pattern as
        // GpuBuffer's buffer_is_freed_when_dropped test.
    }

    #[test]
    fn matmul_f32_computes_a_correct_rectangular_product() {
        // A (2x3, row-major) * B (3x2, row-major) = C (2x2, row-major).
        // Deliberately rectangular, not square -- a row-major/column-major
        // mixup in the cuBLAS argument swap (matmul.cu) would very likely
        // produce a dimension mismatch or an obviously wrong result here,
        // where a square case could accidentally look right (e.g. by
        // silently computing C^T instead of C).
        let (m, n, k) = (2usize, 2usize, 3usize);
        let a_data = f32_to_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // 2x3
        let b_data = f32_to_bytes(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]); // 3x2
        // Expected, hand-computed: C = A * B (2x2)
        let expected = vec![58.0, 64.0, 139.0, 154.0];

        let mut a_buf = GpuBuffer::new(a_data.len()).unwrap();
        let mut b_buf = GpuBuffer::new(b_data.len()).unwrap();
        let c_buf = GpuBuffer::new(m * n * 4).unwrap();
        assert!(a_buf.copy_from_host(&a_data));
        assert!(b_buf.copy_from_host(&b_data));

        let handle = CublasHandle::new().unwrap();
        let ok = handle.matmul_f32(
            a_buf.ptr() as *const f32,
            b_buf.ptr() as *const f32,
            c_buf.ptr() as *mut f32,
            m,
            n,
            k,
        );
        assert!(ok);

        let mut c_bytes = vec![0u8; m * n * 4];
        assert!(c_buf.copy_to_host(&mut c_bytes));
        assert_eq!(bytes_to_f32(&c_bytes), expected);
    }
}
