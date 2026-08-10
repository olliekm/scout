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
