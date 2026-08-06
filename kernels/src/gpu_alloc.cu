// Smallest possible proof that Rust can ask the GPU for real memory and get
// a real pointer back. No kernel logic, no attention math -- this is the
// FFI boundary check, not the paged-attention kernel itself. That kernel is
// a later milestone (AGENT.md Step 5, integrated via FFI same as this).
//
// extern "C" disables C++ name mangling -- without it, the compiled symbol
// name would be something like `_Z14gpu_alloc_bufm` (encoding argument
// types into the name, standard C++ behavior), which Rust's FFI declarations
// (which assume plain C linkage) would have no way to find. `extern "C"`
// forces a plain, unmangled symbol name that Rust can link against directly.

#include <cuda_runtime.h>

extern "C" {

// Allocate `size_bytes` of GPU memory. Returns the device pointer on
// success, or nullptr on failure (out of memory, no GPU, etc.) -- the Rust
// side treats a null pointer as this function's error signal, since CUDA
// error codes themselves aren't exposed across this minimal boundary yet.
void* gpu_alloc_buffer(size_t size_bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size_bytes); 
    if (err != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

// Free a buffer previously returned by gpu_alloc_buffer. Freeing a null
// pointer is a defined no-op in CUDA (matches cudaFree's own documented
// behavior), so the Rust side doesn't need to guard against calling this
// with a null pointer -- it's already safe.
void gpu_free_buffer(void* ptr) {
    cudaFree(ptr);
}

} // extern "C"
