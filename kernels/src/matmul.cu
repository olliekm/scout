// Basic GEMM (matrix multiply) via cuBLAS -- AGENT.md's "integrate,
// don't reinvent well-optimized ops" call, same decision as
// FlashAttention-2 for attention. cuBLAS is NVIDIA's own dense linear
// algebra library; every linear layer in the model (q/k/v/o proj, MLP,
// lm_head) is ultimately a GEMM, so this is the one kernel that makes
// most of the model's weights actually usable.
//
// IMPORTANT gotcha, worth knowing before writing the GEMM call below --
// this produces silently WRONG NUMBERS rather than a compile error if
// missed: cuBLAS assumes COLUMN-MAJOR matrix layout (its Fortran
// heritage), but every tensor loaded via safetensors/PyTorch is
// ROW-MAJOR. The standard trick to compute a row-major C = A * B using a
// column-major-only library, with zero extra data movement: ask cuBLAS
// to compute C^T = B^T * A^T instead -- i.e. swap which pointer goes in
// the "A" argument slot vs the "B" slot, and swap m/n accordingly.
// Column-major storage of C^T is BIT-FOR-BIT IDENTICAL to row-major
// storage of C, so the same bytes, interpreted the "wrong" way on
// purpose, land in the right place.

#include <cuda_runtime.h>
#include <cublas_v2.h>

extern "C" {

// Create a cuBLAS handle. cuBLAS requires one of these to be created
// once and reused across every call -- creating a fresh handle per
// matmul would add real, avoidable setup overhead to every single linear
// layer in the model. Returns the handle on success, or nullptr on
// failure (cublasHandle_t is itself already a pointer type, same shape
// as gpu_alloc_buffer returning void* directly).
//
// YOUR CODE HERE: cublasHandle_t handle; check cublasCreate(&handle)
// against CUBLAS_STATUS_SUCCESS; return handle on success, nullptr
// otherwise.
cublasHandle_t cublas_create_handle() {
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return nullptr;
    }
    return handle;
}

// Destroy a handle created by cublas_create_handle.
//
// YOUR CODE HERE: cublasDestroy(handle).
void cublas_destroy_handle(cublasHandle_t handle) {
    cublasDestroy(handle);
}

// Compute C = A * B for row-major fp32 matrices already resident on the
// GPU: A is m x k, B is k x n, C is m x n (all device pointers). Returns
// true on success, false on any cuBLAS error.
//
// YOUR CODE HERE: call cublasSgemm, applying the row-major/column-major
// trick from the file header comment above -- you are computing
// C^T = B^T * A^T from cuBLAS's point of view, which means the pointer
// and dimension arguments you pass as cuBLAS's "A"/"m"/"lda" are THIS
// function's B/n/n (not a's), and vice versa. alpha=1.0f, beta=0.0f
// (C = 1*A*B + 0*C -- no accumulation into existing C contents). Use
// CUBLAS_OP_N for both transpose arguments (neither operand needs
// transposing beyond the swap already described).
bool matmul_f32(
    cublasHandle_t handle,
    const float* a, const float* b, float* c,
    int m, int n, int k
) {
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasStatus_t status = cublasSgemm(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      b,
      n,
      a,
      k,
      &beta,
      c,
      n
      );
  return status == CUBLAS_STATUS_SUCCESS;
}   


} // extern "C"
