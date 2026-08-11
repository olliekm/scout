// RoPE (Rotary Position Embeddings) -- applied to Q and K (never V) right
// after their projections, before attention runs. Like RMSNorm, this is
// hand-written rather than integrated: it's a simple elementwise
// trig/rotation computation, not a solved-and-optimized problem the way
// matmul/attention are.
//
// The math (HF's "rotate half" convention, what Qwen2/Llama actually use
// -- NOT the interleaved-pairs convention from the original RoPE paper):
// each head's head_dim-sized vector is split into two HALVES, and
// element i (0 <= i < head_dim/2) is paired with element i + head_dim/2.
// For a token at position `pos`, pair i is rotated by an angle that
// depends on both `pos` and `i`:
//   freq_i = theta ^ (-2*i / head_dim)
//   angle  = pos * freq_i
//   new_x_i             =  x_i * cos(angle) - x_{i+head_dim/2} * sin(angle)
//   new_x_{i+head_dim/2} =  x_i * sin(angle) + x_{i+head_dim/2} * cos(angle)
//
// Every pair's rotation is fully independent -- unlike RMSNorm, there's
// NO reduction here, so no shared memory and no __syncthreads() needed.
// Each thread reads its own pair, computes its own angle, writes its own
// result, and never needs to know what any other thread is doing.
//
// Why `positions` is a separate array, not just "thread's row index":
// with continuous batching, one forward pass's batch of tokens can span
// MULTIPLE sequences sitting at DIFFERENT positions simultaneously (e.g.
// row 0 might be sequence A's 40th token, row 1 might be sequence B's
// 3rd) -- the row index alone tells you nothing about the actual
// position to rotate by, so the caller has to supply it explicitly.

#include <cuda_runtime.h>

extern "C" {

// Grid layout: blockIdx.x = token index, blockIdx.y = head index,
// threadIdx.x = pair index (0..head_dim/2) -- a 2D grid + 1D block, so
// each thread's three coordinates map directly onto (token, head, pair)
// with no manual div/mod arithmetic needed to recover them.
//
// YOUR CODE HERE:
//   1. int token = blockIdx.x, head = blockIdx.y, i = threadIdx.x;
//   2. float* base = x + token * num_heads * head_dim + head * head_dim;
//      (this thread's head's slice of this token's row)
//   3. float x1 = base[i], x2 = base[i + head_dim / 2];
//   4. int pos = positions[token];
//      float freq = powf(theta, -2.0f * i / head_dim);
//      float angle = pos * freq;
//      float cos_a = cosf(angle), sin_a = sinf(angle);
//   5. base[i] = x1 * cos_a - x2 * sin_a;
//      base[i + head_dim / 2] = x1 * sin_a + x2 * cos_a;
//      (in-place is safe here -- x1/x2 were already read into registers
//      in step 3 before either write happens, and no other thread ever
//      touches this thread's (token, head, i) slice)
__global__ void rope_kernel(
    float* x, const int* positions,
    int num_heads, int head_dim, float theta
) {
  int token = blockIdx.x;
  int head = blockIdx.y;
  int i = threadIdx.x;

  float* base = x + token * num_heads * head_dim + head * head_dim;
  float x1 = base[i];
  float x2 = base[i + head_dim / 2];
  
  int pos = positions[token];
  float freq = powf(theta, -2.0f * i / head_dim);
  float angle = pos * freq;
  float cos_a = cosf(angle);
  float sin_a = sinf(angle);

  base[i] = x1 * cos_a - x2 * sin_a;
  base[i + head_dim / 2] = x1 * sin_a + x2 * cos_a;
}

// Host-side launcher. Grid: (num_tokens, num_heads) blocks. Block:
// head_dim/2 threads -- exactly enough to cover every pair once, no
// grid-stride loop needed (head_dim/2 is small, e.g. 64 for this
// model's head_dim=128, well under the 1024 threads-per-block limit).
//
// YOUR CODE HERE:
//   1. dim3 grid(num_tokens, num_heads);
//      rope_kernel<<<grid, head_dim / 2>>>(x, positions, num_heads, head_dim, theta);
//   2. Check cudaGetLastError() against cudaSuccess, same pattern as
//      rmsnorm_f32/matmul_f32.
bool rope_f32(
    float* x, const int* positions,
    int num_tokens, int num_heads, int head_dim, float theta
) {
  dim3 grid(num_tokens, num_heads);
  rope_kernel<<<grid, head_dim / 2>>>(x, positions, num_heads, head_dim, theta);
  return cudaGetLastError() == cudaSuccess;
}

} // extern "C"
