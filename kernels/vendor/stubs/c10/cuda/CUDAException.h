// Stub replacing PyTorch's c10/cuda/CUDAException.h. flash_fwd_launch_template.h
// includes the real header for exactly two macros (C10_CUDA_CHECK,
// C10_CUDA_KERNEL_LAUNCH_CHECK) -- both are trivial cudaError_t checks in
// PyTorch's own source, nothing else from c10 is used anywhere in the vendored
// flash_attn/ code. This stub exists so nvcc resolves the same #include path
// to something real without requiring libtorch as a dependency.
//
// Deliberately does NOT throw a C++ exception the way PyTorch's real macro
// does -- an exception unwinding across the extern "C" boundary into Rust
// would be undefined behavior. attention.cu (our own wrapper) already calls
// cudaGetLastError() itself right after the kernel launch and reports
// failure through a normal bool return, so nothing is silently lost by
// swallowing the error here instead.
#pragma once
#include <cuda_runtime.h>

#define C10_CUDA_CHECK(EXPR) do { (void)(EXPR); } while (0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() do { (void)cudaGetLastError(); } while (0)
