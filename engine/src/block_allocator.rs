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

use std::collections::HashMap;


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
    free_blocks: Vec<BlockId>,

    // YOUR FIELD HERE.
    //
    // This is what turns a generic block pool into a KV-cache allocator:
    // a map from sequence id to the list of blocks that sequence currently
    // owns, so that when a sequence finishes you know exactly which blocks
    // to return to the free pool, and so a growing sequence can find its
    // own block list to append to.
    //
    // `HashMap<K, V>` is Rust's dictionary type (Python's `dict`). Add:
    //   seq_blocks: HashMap<u64, Vec<BlockId>>,
    // (and add `use std::collections::HashMap;` near the top of the file --
    // unlike Vec/Option, HashMap isn't in Rust's "prelude" of auto-imported
    // types, so it needs an explicit `use`.)
}

impl BlockAllocator {
    /// Construct a new allocator managing `num_blocks` blocks, ALL initially
    /// free. This is the Rust equivalent of a constructor / `__init__` --
    /// by convention it's a static function named `new` (not a keyword,
    /// just the idiom), returning `Self` (shorthand for `BlockAllocator`).
    pub fn new(num_blocks: usize) -> Self {
        let free_blocks: Vec<BlockId> = (0..num_blocks).collect();

        let allo: BlockAllocator = BlockAllocator {
            num_blocks: num_blocks,
            free_blocks: free_blocks,
        };

        return allo
    }

    /// Try to allocate ONE free block. Returns `Some(block_id)` if a block
    /// was available, `None` if the allocator is out of blocks (this is the
    /// Rust way of expressing "might fail, no exception" -- the caller has
    /// to handle the None case, the compiler won't let them forget).
    pub fn allocate(&mut self) -> Option<BlockId> {
        self.free_blocks.pop()
    }

    /// Return a block to the free pool. Takes ownership of nothing special
    /// here -- `block_id` is a plain `usize`, Rust's `Copy` types (small,
    /// stack-only values like integers) don't have the "ownership moves"
    /// behavior that e.g. a `String` or `Vec` would -- so you don't need to
    /// worry about ownership subtleties for this one, just push it back.
    pub fn free(&mut self, block_id: BlockId) {
        self.free_blocks.push(block_id);
    }

    /// How many blocks are currently free. Useful for tests and for the
    /// scheduler later (needs to know if there's room before admitting a
    /// new request).
    pub fn num_free(&self) -> usize {
        return self.free_blocks.len();
    }

    /// Allocate one more block and record it as owned by `seq_id`. Returns
    /// the new block's id, or `None` if the pool is exhausted (this is the
    /// realistic "ran out of GPU memory for KV cache" case -- for this
    /// milestone, just fail cleanly and leave everything else untouched;
    /// eviction/preemption is the scheduler's problem later, not this type's).
    ///
    /// Two things to get right:
    ///   1. Reuse your existing `allocate()` to get a free block id -- don't
    ///      duplicate the pop-from-free-list logic here.
    ///   2. Record that this seq_id now owns that block, appending to
    ///      whatever `Vec<BlockId>` is already associated with `seq_id` in
    ///      the map (or starting a new one, if this is the sequence's first
    ///      block). `HashMap::entry(key).or_insert_with(Vec::new)` is the
    ///      idiomatic way to say "get the Vec for this key, creating an
    ///      empty one first if it doesn't exist yet" in one expression --
    ///      look up its docs if unfamiliar, or write it as an explicit
    ///      match/if-let on `self.seq_blocks.get_mut(&seq_id)` first if
    ///      that's clearer while learning.
    pub fn allocate_block_for(&mut self, seq_id: u64) -> Option<BlockId> {
        todo!("allocate a block, record it under seq_id in seq_blocks, return its id")
    }

    /// Free ALL blocks owned by `seq_id` (e.g. the sequence finished
    /// generating, or was evicted) and remove it from tracking entirely.
    ///
    /// `HashMap::remove(&key)` removes an entry and returns `Option<V>` (the
    /// value that was there, if any) -- exactly what you need here: get the
    /// Vec<BlockId> that seq_id owned, then free() each one back to the pool.
    pub fn free_sequence(&mut self, seq_id: u64) {
        todo!("remove seq_id's entry from seq_blocks, free() every block it owned")
    }

    /// How many blocks does `seq_id` currently own? Returns 0 if the
    /// sequence isn't tracked at all (never allocated, or already freed) --
    /// NOT an error case, so this returns a plain `usize`, not `Option`.
    pub fn blocks_owned_by(&self, seq_id: u64) -> usize {
        todo!("look up seq_id in seq_blocks, return how many blocks it owns (0 if untracked)")
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

    #[test]
    fn untracked_sequence_owns_zero_blocks() {
        let allocator = BlockAllocator::new(4);
        assert_eq!(allocator.blocks_owned_by(99), 0);
    }

    #[test]
    fn allocate_block_for_tracks_ownership() {
        let mut allocator = BlockAllocator::new(4);
        let b1 = allocator.allocate_block_for(1);
        assert!(b1.is_some());
        assert_eq!(allocator.blocks_owned_by(1), 1);

        let b2 = allocator.allocate_block_for(1);
        assert!(b2.is_some());
        assert_eq!(allocator.blocks_owned_by(1), 2);
        // the two blocks handed to the same sequence must be different blocks
        assert_ne!(b1, b2);

        assert_eq!(allocator.num_free(), 2);
    }

    #[test]
    fn allocate_block_for_two_sequences_are_independent() {
        let mut allocator = BlockAllocator::new(4);
        allocator.allocate_block_for(1);
        allocator.allocate_block_for(2);
        allocator.allocate_block_for(2);

        assert_eq!(allocator.blocks_owned_by(1), 1);
        assert_eq!(allocator.blocks_owned_by(2), 2);
    }

    #[test]
    fn allocate_block_for_returns_none_when_exhausted() {
        let mut allocator = BlockAllocator::new(1);
        assert!(allocator.allocate_block_for(1).is_some());
        assert!(allocator.allocate_block_for(1).is_none());
        // the failed attempt must not have corrupted seq 1's existing state
        assert_eq!(allocator.blocks_owned_by(1), 1);
    }

    #[test]
    fn free_sequence_returns_all_its_blocks() {
        let mut allocator = BlockAllocator::new(4);
        allocator.allocate_block_for(1);
        allocator.allocate_block_for(1);
        assert_eq!(allocator.num_free(), 2);

        allocator.free_sequence(1);
        assert_eq!(allocator.num_free(), 4);
        assert_eq!(allocator.blocks_owned_by(1), 0);
    }

    #[test]
    fn free_sequence_does_not_affect_other_sequences() {
        let mut allocator = BlockAllocator::new(4);
        allocator.allocate_block_for(1);
        allocator.allocate_block_for(2);

        allocator.free_sequence(1);

        assert_eq!(allocator.blocks_owned_by(1), 0);
        assert_eq!(allocator.blocks_owned_by(2), 1);
        assert_eq!(allocator.num_free(), 3);
    }

    #[test]
    fn free_sequence_on_untracked_id_is_a_harmless_no_op() {
        let mut allocator = BlockAllocator::new(4);
        allocator.free_sequence(404); // must not panic
        assert_eq!(allocator.num_free(), 4);
    }
}
