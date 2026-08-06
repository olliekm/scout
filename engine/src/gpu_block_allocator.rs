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
    allocator: BlockAllocator,
    gpu: PagedGpuBuffer,
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
        let allocator = BlockAllocator::new(num_blocks, block_size);
        match PagedGpuBuffer::new(num_blocks, block_size_bytes) {
            Some(gpu) => {
                Some(Self { allocator, gpu })
            }
            None => {
                None
            }
        }
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
        let (block_id, _) = self.allocator.locate(seq_id, position)?;
        self.gpu.block_ptr(block_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_successfully() {
        let alloc = GpuBlockAllocator::new(4, 4, 1024);
        assert!(alloc.is_some());
    }

    #[test]
    fn block_ptr_for_unknown_seq_is_none() {
        let alloc = GpuBlockAllocator::new(4, 4, 1024).unwrap();
        assert!(alloc.block_ptr_for(99, 0).is_none());
    }

    #[test]
    fn block_ptr_for_position_beyond_allocated_is_none() {
        let mut alloc = GpuBlockAllocator::new(4, 4, 1024).unwrap();
        alloc.allocator.allocate_block_for(1); // one block: holds positions 0..4
        assert!(alloc.block_ptr_for(1, 4).is_none()); // position 4 needs a second block
    }

    #[test]
    fn block_ptr_for_resolves_to_matching_gpu_block_ptr() {
        let mut alloc = GpuBlockAllocator::new(4, 4, 1024).unwrap();
        let block_id = alloc.allocator.allocate_block_for(1).unwrap();
        let expected = alloc.gpu.block_ptr(block_id).unwrap();
        assert_eq!(alloc.block_ptr_for(1, 0).unwrap(), expected);
    }

    #[test]
    fn block_ptr_for_crosses_block_boundary() {
        // block_size = 4 tokens: positions 0..4 land in the first allocated
        // block, position 4 needs a second block to have been allocated.
        let mut alloc = GpuBlockAllocator::new(4, 4, 1024).unwrap();
        let b0 = alloc.allocator.allocate_block_for(1).unwrap();
        let b1 = alloc.allocator.allocate_block_for(1).unwrap();
        assert_ne!(b0, b1);

        let ptr_in_first = alloc.block_ptr_for(1, 3).unwrap();
        let ptr_in_second = alloc.block_ptr_for(1, 4).unwrap();

        assert_eq!(ptr_in_first, alloc.gpu.block_ptr(b0).unwrap());
        assert_eq!(ptr_in_second, alloc.gpu.block_ptr(b1).unwrap());
    }

    #[test]
    fn block_ptr_for_ignores_within_block_offset() {
        // block_ptr resolves to the START of the block regardless of the
        // in-block offset locate() computes -- positions 0..3 within the
        // same block should all resolve to the same pointer.
        let mut alloc = GpuBlockAllocator::new(4, 4, 1024).unwrap();
        let block_id = alloc.allocator.allocate_block_for(1).unwrap();
        let expected = alloc.gpu.block_ptr(block_id).unwrap();

        for position in 0..4 {
            assert_eq!(alloc.block_ptr_for(1, position).unwrap(), expected);
        }
    }
}









