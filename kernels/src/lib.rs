//! Thinnest possible proof that Rust can ask the GPU for real memory and
//! get a real pointer back, via FFI into gpu_alloc.cu. No kernel logic here
//! -- see that file's header comment for full scope/rationale.
//!
//! `unsafe` in Rust: everywhere else in this project (engine/), the
//! compiler statically proves memory safety -- no null derefs, no
//! use-after-free, no data races, all checked at compile time. Calling into
//! C/CUDA breaks that: the compiler has no way to verify what a foreign
//! function does with a pointer, or whether it's still valid by the time
//! you use it. `unsafe { ... }` is Rust's way of saying "the compiler can't
//! check this block, but I've verified it's safe by hand." The pattern used
//! here -- and the one worth keeping throughout this whole project -- is to
//! keep `unsafe` blocks as small and as deep as possible, and wrap every
//! raw FFI call in a SAFE function immediately, so nothing outside this
//! file ever needs to write `unsafe` itself.

use std::ffi::c_void;

pub mod matmul;
pub mod rmsnorm;
pub mod rope;
pub use matmul::CublasHandle;

// This block declares the C functions gpu_alloc.cu defines, so Rust knows
// their signatures and can link against them (the build.rs script is what
// makes the actual compiled code available to link against; this block is
// just the type-level declaration side). `unsafe extern "C"` because
// calling ANY foreign function is unsafe by definition -- Rust can't verify
// what happens on the other side of this boundary.
unsafe extern "C" {
    fn gpu_alloc_buffer(size_bytes: usize) -> *mut c_void;
    fn gpu_free_buffer(ptr: *mut c_void);
    fn gpu_copy_to_device(dst: *mut c_void, src: *const c_void, size_bytes: usize) -> bool;
    fn gpu_copy_to_host(dst: *mut c_void, src: *const c_void, size_bytes: usize) -> bool;
}

/// A GPU memory buffer, owned by this struct. This is where BlockId will
/// eventually stop being an abstract number and become a real offset into
/// a buffer allocated by this type -- not wired up yet, this milestone is
/// just proving the alloc/free round-trip itself works.
pub struct GpuBuffer {
    ptr: *mut c_void,
    size_bytes: usize,
}

impl GpuBuffer {
    /// Allocate `size_bytes` of GPU memory. Returns `None` if the
    /// allocation failed (out of GPU memory, no GPU present, etc.) --
    /// gpu_alloc_buffer signals failure with a null pointer, which this
    /// safe wrapper translates into Rust's normal "might fail" idiom
    /// instead of leaking a raw null pointer out to safe code.
    pub fn new(size_bytes: usize) -> Option<Self> {
        // SAFETY: gpu_alloc_buffer is a pure allocation call with no
        // aliasing/lifetime preconditions on Rust's side to violate --
        // it either returns a valid freshly-allocated pointer or null.
        let ptr = unsafe { gpu_alloc_buffer(size_bytes) };

        if ptr.is_null() {
            return None;
        }

        Some(GpuBuffer { ptr, size_bytes })
    }

    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Copy `data` from host memory into this buffer's device memory.
    /// Returns false (without attempting the copy) if `data` is larger
    /// than this buffer -- copying more than the allocation holds would
    /// write past its end, so this is checked on the Rust side rather
    /// than trusting the FFI call to catch it.
    ///
    /// Steps:
    ///   1. If data.len() > self.size_bytes, return false.
    ///   2. unsafe { gpu_copy_to_device(self.ptr, data.as_ptr() as *const c_void, data.len()) }
    ///      -- SAFETY: self.ptr is a valid device allocation of at least
    ///      data.len() bytes (checked above), data.as_ptr() is valid for
    ///      data.len() bytes per Rust's own slice guarantees.
    pub fn copy_from_host(&mut self, data: &[u8]) -> bool {
        if data.len() > self.size_bytes {
            return false;
        }
        unsafe {
            gpu_copy_to_device(self.ptr, data.as_ptr() as *const c_void, data.len())
        }
        
    }

    /// Copy this buffer's device memory into `data` (host memory) -- the
    /// reverse of copy_from_host, same bounds check and SAFETY reasoning.
    pub fn copy_to_host(&self, data: &mut [u8]) -> bool {
        if data.len() > self.size_bytes {
            return false;
        }
        unsafe {
            gpu_copy_to_host(data.as_mut_ptr() as *mut c_void, self.ptr, data.len())
        }
    }
}

// Drop is Rust's destructor trait -- this method runs automatically when a
// GpuBuffer goes out of scope, no matter how (normal return, early return,
// panic unwinding). This is what makes GpuBuffer's GPU memory impossible to
// leak by forgetting to free it -- ownership + Drop together give you
// deterministic cleanup without a garbage collector, which is exactly the
// property you want managing GPU memory (a leaked cudaMalloc doesn't get
// cleaned up by anything else).
impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // SAFETY: self.ptr was returned by gpu_alloc_buffer in `new` and
        // has not been freed yet (Drop runs at most once per value, Rust's
        // ownership rules guarantee that) -- this is exactly the pointer
        // gpu_free_buffer expects.
        unsafe {
            gpu_free_buffer(self.ptr);
        }
    }
}

/// A GPU memory region carved into `num_blocks` fixed-size blocks, each
/// `block_size_bytes` bytes -- the real-memory counterpart to
/// BlockAllocator's abstract BlockIds (engine/src/block_allocator.rs). One
/// cudaMalloc for the whole region up front; after that, resolving a
/// BlockId to a device pointer is pure offset arithmetic, no further CUDA
/// allocation calls. This is what keeps block alloc/free off the hot path
/// of talking to the GPU allocator at all.
pub struct PagedGpuBuffer {
    // YOUR FIELD HERE.
    //
    // GpuBuffer.ptr is private, so PagedGpuBuffer can't reach in and copy
    // out a raw pointer -- and even if it could, that would leave two
    // owners of the same pointer with no rule for who calls cudaFree
    // (whichever one drops first frees memory the other still points at).
    // Instead, PagedGpuBuffer should OWN a GpuBuffer outright:
    //   buffer: GpuBuffer,
    // Storing the whole GpuBuffer (not just its pointer) means Drop is
    // inherited for free -- when a PagedGpuBuffer is dropped, its `buffer`
    // field is dropped too, which is what actually calls cudaFree. You get
    // correct cleanup without writing a Drop impl yourself.
    block_size_bytes: usize,
    num_blocks: usize,
    buffer: GpuBuffer,
}

impl PagedGpuBuffer {
    /// Allocate one GPU region big enough for `num_blocks` blocks of
    /// `block_size_bytes` each. Returns `None` on allocation failure, same
    /// convention as GpuBuffer::new.
    ///
    /// Steps:
    ///   1. total_bytes = num_blocks * block_size_bytes
    ///   2. Call GpuBuffer::new(total_bytes) -- it returns Option<GpuBuffer>
    ///   3. match on that: Some(buffer) => wrap Self { buffer,
    ///      block_size_bytes, num_blocks } in Some(..); None => propagate
    ///      None straight through (don't panic/assert -- a real alloc
    ///      failure, e.g. GPU OOM, is an expected, recoverable case the
    ///      caller needs to see, not a bug).
    pub fn new(num_blocks: usize, block_size_bytes: usize) -> Option<Self> {
        let total_bytes = num_blocks * block_size_bytes;
        match GpuBuffer::new(total_bytes) {
            Some(buffer) => {
                Some(Self {block_size_bytes, num_blocks, buffer})
            }
            None => {
                None
            }
        }
    }

    /// Resolve a block index (0..num_blocks) to its device pointer within
    /// the region -- base pointer + block_index * block_size_bytes. No CUDA
    /// call here, just arithmetic on a pointer Rust already owns.
    ///
    /// You'll need a way to read the raw pointer out of `self.buffer` --
    /// GpuBuffer has a `size_bytes()` getter already; it needs an analogous
    /// one for the pointer (add `pub fn ptr(&self) -> *mut c_void` next to
    /// `size_bytes()` on GpuBuffer, returning `self.ptr`).
    ///
    /// Once you have the base pointer, `*mut c_void` doesn't support `+`
    /// directly -- look at `<*mut T>::add`, keeping in mind it steps in
    /// units of the pointee type's size (`c_void` is a zero-sized/opaque
    /// type here), so you'll likely want to cast to `*mut u8` first, do
    /// byte-wise arithmetic, then cast back.
    ///
    /// What should happen if block_index >= num_blocks?
    pub fn block_ptr(&self, block_index: usize) -> Option<*mut c_void> {
        if block_index >= self.num_blocks {
            return None;
        }
        let base_ptr: *mut u8 = self.buffer.ptr() as *mut u8;
        let offset: usize = block_index * self.block_size_bytes;
        let block_ptr: *mut u8 = unsafe { base_ptr.add(offset) };
        Some(block_ptr as *mut c_void)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_a_real_gpu_buffer() {
        let buf = GpuBuffer::new(1024);
        assert!(buf.is_some());
        assert_eq!(buf.unwrap().size_bytes(), 1024);
    }

    #[test]
    fn buffer_is_freed_when_dropped() {
        // This test's real assertion is implicit: if gpu_free_buffer were
        // never called, or were called with a bad pointer, a large enough
        // loop of alloc-then-drop would eventually exhaust GPU memory or
        // crash. Passing is the signal that Drop is correctly wired up.
        for _ in 0..1000 {
            let buf = GpuBuffer::new(1024 * 1024); // 1 MiB each
            assert!(buf.is_some());
            // buf drops here at the end of each loop iteration
        }
    }

    #[test]
    fn allocates_a_paged_gpu_buffer() {
        let buf = PagedGpuBuffer::new(4, 1024);
        assert!(buf.is_some());
    }

    #[test]
    fn block_ptr_is_none_out_of_bounds() {
        let buf = PagedGpuBuffer::new(4, 1024).unwrap();
        assert!(buf.block_ptr(4).is_none()); // valid range is 0..4
        assert!(buf.block_ptr(100).is_none());
    }

    #[test]
    fn block_ptr_is_some_in_bounds() {
        let buf = PagedGpuBuffer::new(4, 1024).unwrap();
        for i in 0..4 {
            assert!(buf.block_ptr(i).is_some());
        }
    }

    #[test]
    fn block_ptr_advances_by_block_size_bytes() {
        // Each block's pointer should sit exactly block_size_bytes past the
        // previous one -- this is the actual paging guarantee: block_ptr(i)
        // must land on the start of a disjoint, non-overlapping region.
        let block_size_bytes = 1024;
        let buf = PagedGpuBuffer::new(4, block_size_bytes).unwrap();

        let p0 = buf.block_ptr(0).unwrap() as usize;
        let p1 = buf.block_ptr(1).unwrap() as usize;
        let p3 = buf.block_ptr(3).unwrap() as usize;

        assert_eq!(p1 - p0, block_size_bytes);
        assert_eq!(p3 - p0, 3 * block_size_bytes);
    }
}
