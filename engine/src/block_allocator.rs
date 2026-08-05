//! Paged KV cache block allocator -- the "core systems problem" per AGENT.md.
//!
//! Scope for this first milestone: pure bookkeeping, no GPU/model involved.
//! We're tracking WHICH block IDs are free/in-use, not actually touching any
//! memory. Think of this as reserving seats in a stadium by seat number --
//! we're not building the seats, just tracking who's sitting where.
//!
//! Rust concepts you'll need, if new to the language:
//!
//! - `struct` is like a Python class with only data (fields), no methods
//!   defined inline -- methods go in a separate `impl` block below the struct.
//! - `Vec<T>` is Rust's growable array type, roughly Python's `list`, but
//!   every element must be the same type `T` (Rust has no dynamic typing).
//! - `Option<T>` replaces `None`-as-a-possible-value: instead of a function
//!   silently returning `None` when Python would, Rust makes you explicit --
//!   a function returning `Option<BlockId>` either returns `Some(the_value)`
//!   or `None`, and the CALLER is forced (by the compiler) to handle both
//!   cases before using the value. This is Rust's main tool for "this might
//!   fail, handle it" instead of exceptions.
//! - `&mut self` in a method signature means "this method needs to mutate
//!   the struct it's called on" -- roughly like Python's `self` but Rust
//!   requires you to be explicit about whether a method reads (`&self`) or
//!   writes (`&mut self`) the struct's data. This is the ownership system
//!   starting to show up: only one piece of code can hold a `&mut` reference
//!   to something at a time, which is exactly the property you want for an
//!   allocator (two things can't both think they own the same block).

/// A block's identity is just its index into the pool. `usize` is Rust's
/// "unsigned integer sized for indexing" type -- what you'd use for a list
/// index. A type alias (`type BlockId = usize`) doesn't create a new type,
/// it's just a readable name -- but it's still worth using instead of raw
/// `usize` everywhere, so function signatures self-document what the number
/// MEANS (a block id, not a token count or anything else).
pub type BlockId = usize;

pub struct BlockAllocator {
    /// Total number of blocks this allocator manages. Fixed at construction --
    /// this stands in for "GPU memory reserved for KV cache, divided into
    /// num_blocks fixed-size chunks."
    num_blocks: usize,

    // YOUR FIELD(S) HERE.
    //
    // You need some way to track which block IDs are currently free. A
    // natural choice: `Vec<BlockId>` used as a stack (free list) -- when a
    // block is freed, push its id; when allocating, pop an id off.
    //
    // Add the field(s) you need here, e.g.:
    //   free_blocks: Vec<BlockId>,
}

impl BlockAllocator {
    /// Construct a new allocator managing `num_blocks` blocks, ALL initially
    /// free. This is the Rust equivalent of a constructor / `__init__` --
    /// by convention it's a static function named `new` (not a keyword,
    /// just the idiom), returning `Self` (shorthand for `BlockAllocator`).
    pub fn new(num_blocks: usize) -> Self {
        todo!("construct a BlockAllocator with all num_blocks ids initially free")
    }

    /// Try to allocate ONE free block. Returns `Some(block_id)` if a block
    /// was available, `None` if the allocator is out of blocks (this is the
    /// Rust way of expressing "might fail, no exception" -- the caller has
    /// to handle the None case, the compiler won't let them forget).
    pub fn allocate(&mut self) -> Option<BlockId> {
        todo!("pop a block id off the free list, if any are available")
    }

    /// Return a block to the free pool. Takes ownership of nothing special
    /// here -- `block_id` is a plain `usize`, Rust's `Copy` types (small,
    /// stack-only values like integers) don't have the "ownership moves"
    /// behavior that e.g. a `String` or `Vec` would -- so you don't need to
    /// worry about ownership subtleties for this one, just push it back.
    pub fn free(&mut self, block_id: BlockId) {
        todo!("return block_id to the free list")
    }

    /// How many blocks are currently free. Useful for tests and for the
    /// scheduler later (needs to know if there's room before admitting a
    /// new request).
    pub fn num_free(&self) -> usize {
        todo!("return the count of free blocks")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_with_all_blocks_free() {
        let allocator = BlockAllocator::new(4);
        assert_eq!(allocator.num_free(), 4);
    }

    #[test]
    fn allocate_reduces_free_count() {
        let mut allocator = BlockAllocator::new(4);
        let block = allocator.allocate();
        assert!(block.is_some());
        assert_eq!(allocator.num_free(), 3);
    }

    #[test]
    fn allocate_until_exhausted_returns_none() {
        let mut allocator = BlockAllocator::new(2);
        assert!(allocator.allocate().is_some());
        assert!(allocator.allocate().is_some());
        assert!(allocator.allocate().is_none());
        assert_eq!(allocator.num_free(), 0);
    }

    #[test]
    fn free_returns_block_to_pool() {
        let mut allocator = BlockAllocator::new(1);
        let block = allocator.allocate().unwrap();
        assert_eq!(allocator.num_free(), 0);
        allocator.free(block);
        assert_eq!(allocator.num_free(), 1);
    }

    #[test]
    fn freed_block_can_be_reallocated() {
        let mut allocator = BlockAllocator::new(1);
        let block = allocator.allocate().unwrap();
        allocator.free(block);
        let reallocated = allocator.allocate();
        assert!(reallocated.is_some());
    }
}
