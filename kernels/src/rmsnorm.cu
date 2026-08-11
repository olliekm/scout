// RMSNorm (root-mean-square layer normalization) -- Qwen2 uses this
// instead of LayerNorm for its input_layernorm/post_attention_layernorm
// weights. Unlike matmul/attention, this is NOT a "well-optimized op to
// integrate rather than reinvent" -- the whole computation is: mean of
// squares, normalize, scale by a learned per-channel weight. Simple
// enough to be a genuine milestone to hand-write, per AGENT.md's
// "hand-write kernels only where it's the point of a specific milestone."
//
// Formula, per token (row) of hidden_size elements:
//   rms = sqrt(mean(x_i^2) + eps)
//   out_i = (x_i / rms) * weight_i
//
// Parallelization: one CUDA block per token, THREADS_PER_BLOCK threads
// within the block cooperate to compute that token's sum of squares -- a
// REDUCTION, the classic CUDA pattern this kernel exists to teach.
// hidden_size (3584 for Qwen2.5-Coder-7B) is larger than THREADS_PER_BLOCK,
// so each thread first sums several elements on its own (a grid-stride
// loop over its slice of the row: indices threadIdx.x, threadIdx.x +
// blockDim.x, threadIdx.x + 2*blockDim.x, ...), then the partial sums
// across all threads in the block get combined via a shared-memory tree
// reduction.

#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

extern "C" {

// One block per token (blockIdx.x = token index / row). blockDim.x
// (== THREADS_PER_BLOCK) threads cooperate on that token's hidden_size
// elements.
//
// YOUR CODE HERE, in three stages:
//   1. int row = blockIdx.x; const float* row_x = x + row * hidden_size;
//      float sum = 0.0f;
//      for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
//          sum += row_x[i] * row_x[i];
//      }
//      (each thread now holds a PARTIAL sum of squares over its slice)
//   2. Combine every thread's partial sum into one total via shared
//      memory:
//      __shared__ float shared[THREADS_PER_BLOCK];
//      shared[threadIdx.x] = sum;
//      __syncthreads();
//      for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
//          if (threadIdx.x < stride) {
//              shared[threadIdx.x] += shared[threadIdx.x + stride];
//          }
//          __syncthreads();
//      }
//      -- shared[0] now holds the full sum of squares for this token.
//      (Why __syncthreads() every iteration: every thread must finish
//      WRITING its half of the current round before any thread reads
//      from it in the next round -- skipping this creates a race.)
//   3. float* row_out = out + row * hidden_size;
//      float rms = sqrtf(shared[0] / hidden_size + eps);
//      for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
//          row_out[i] = (row_x[i] / rms) * weight[i];
//      }
__global__ void rmsnorm_kernel(
    const float* x, const float* weight, float* out,
    int hidden_size, float eps
) {
    int row = blockIdx.x;
    const float* row_x = x + row * hidden_size;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
      sum += row_x[i] * row_x[i];
    }

    __shared__ float shared[THREADS_PER_BLOCK];
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride) {
        shared[threadIdx.x] += shared[threadIdx.x + stride];
      }
      __syncthreads();
    }

    float* row_out = out + row * hidden_size;
    float rms = sqrtf(shared[0] / hidden_size + eps);
    for (int i = threadIdx.x; i < hidden_size; i+= blockDim.x) {
      row_out[i] = (row_x[i] / rms) * weight[i];
    }
}

// Host-side launcher: one block per token (num_tokens blocks total), a
// fixed THREADS_PER_BLOCK threads per block.
//
// YOUR CODE HERE:
//   1. rmsnorm_kernel<<<num_tokens, THREADS_PER_BLOCK>>>(x, weight, out, hidden_size, eps);
//   2. Check for launch errors: cudaGetLastError() -- kernel launches are
//      asynchronous and don't return a status directly the way
//      cudaMemcpy/cudaMalloc do (a bad launch config, e.g. too many
//      threads per block, fails asynchronously); this separate call is
//      how you find out.
//   3. Return true if cudaGetLastError() == cudaSuccess, false otherwise.
bool rmsnorm_f32(
    const float* x, const float* weight, float* out,
    int num_tokens, int hidden_size, float eps
) {
  rmsnorm_kernel<<<num_tokens, THREADS_PER_BLOCK>>>(x, weight, out, hidden_size, eps);
  if (cudaGetLastError() == cudaSuccess) {
    return true;
  }
  return false;
}

} // extern "C"
