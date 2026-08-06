//! Connects the abstract bookkeeping in `block_allocator.rs` to real GPU
//! memory in the `kernels` crate. `BlockAllocator` deliberately knows
//! nothing about the GPU (see its module doc: "pure bookkeeping, no
//! GPU/model involved") -- this type is where that separation gets bridged,
//! by composing a `BlockAllocator` with a `PagedGpuBuffer` rather than
//! modifying either one.
//!
//! Only compiled with `--features gpu`, since `kernels` requires `nvcc` and
//! can only build on the pod (see kernels/build.rs).

use crate::block_allocator::BlockAllocator;
use kernels::PagedGpuBuffer;
use std::ffi::c_void;

pub struct GpuBlockAllocator {
    // YOUR FIELD HERE.
    //
    // Same shape as PagedGpuBuffer holding a GpuBuffer: this type should
    // OWN both an allocator and a GPU buffer, not borrow references to
    // them. Two fields:
    //   allocator: BlockAllocator,
    //   gpu: PagedGpuBuffer,
}

impl GpuBlockAllocator {
    /// Construct a GpuBlockAllocator wrapping `num_blocks` blocks of
    /// `block_size` tokens each (bookkeeping side) backed by a real GPU
    /// region of `num_blocks` blocks of `block_size_bytes` bytes each
    /// (memory side). These are two different units for "block size" --
    /// BlockAllocator counts tokens per block, PagedGpuBuffer counts bytes
    /// per block -- so both numbers are needed here, not just one.
    ///
    /// Returns `None` if the GPU allocation fails (mirrors
    /// PagedGpuBuffer::new's contract) -- BlockAllocator::new itself can't
    /// fail, so the only failure path to propagate is the GPU side.
    pub fn new(num_blocks: usize, block_size: usize, block_size_bytes: usize) -> Option<Self> {
        todo!()
    }

    /// Resolve a sequence's logical token position directly to a GPU device
    /// pointer, composing the two lookups this type exists to connect:
    ///   1. self.allocator.locate(seq_id, position) -> Option<(BlockId, usize)>
    ///      (the usize here is the in-block token offset -- NOT used for
    ///      the pointer lookup itself, block_ptr resolves to the START of
    ///      the block; a caller doing byte-level addressing within the
    ///      block would need it separately)
    ///   2. self.gpu.block_ptr(block_id) -> Option<*mut c_void>
    ///
    /// Both steps can fail (unknown seq_id/position, or an out-of-range
    /// block_id) -- `?` on an Option propagates a None from either step
    /// straight through as this function's None, without needing an
    /// explicit match.
    pub fn block_ptr_for(&self, seq_id: u64, position: usize) -> Option<*mut c_void> {
        todo!()
    }
}
