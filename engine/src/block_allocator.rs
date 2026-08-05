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
use std::collections::HashSet;


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
    seq_blocks: HashMap<u64, Vec<BlockId>>,

    /// Fixed number of tokens each block holds -- the KV-cache analogue of
    /// OS page size. Set once at construction; every block is this size, no
    /// exceptions (that uniformity is what makes the logical-position ->
    /// (block, offset) math below a simple divide/modulo instead of needing
    /// to search).
    block_size: usize,
    evictable: HashSet<u64>
    // YOUR FIELD HERE.
    //
    // The allocator itself has no idea which sequences are "important" (a
    // request that's already streamed output to a user vs. one just
    // admitted) -- that's scheduling policy, which lives above this type.
    // What the allocator DOES need: a place for the caller to mark which
    // tracked sequences are ALLOWED to be evicted if the pool runs out,
    // so allocate_block_for's failure path has somewhere to look for a
    // candidate instead of just failing outright.
    //
    // `HashSet<T>` is Rust's set type (Python's `set`) -- membership only,
    // no values. Add:
    //   evictable: HashSet<u64>,
    // and add `use std::collections::HashSet;` near the other `use` line.
}

impl BlockAllocator {
    /// Construct a new allocator managing `num_blocks` blocks of
    /// `block_size` tokens each, ALL blocks initially free. This is the
    /// Rust equivalent of a constructor / `__init__` -- by convention it's
    /// a static function named `new` (not a keyword, just the idiom),
    /// returning `Self` (shorthand for `BlockAllocator`).
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let free_blocks: Vec<BlockId> = (0..num_blocks).collect();
        let seq_blocks: HashMap<u64, Vec<BlockId>> = HashMap::new();
        let evictable: HashSet<u64> = HashSet::new();

        let allo: BlockAllocator = BlockAllocator {
            num_blocks: num_blocks,
            free_blocks: free_blocks,
            seq_blocks: seq_blocks,
            block_size: block_size,
            evictable: evictable,
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
    /// the new block's id, or `None` only if the pool is exhausted AND no
    /// evictable sequence exists to make room (a genuine "no more memory,
    /// nothing this allocator can do about it" case -- what the caller does
    /// then, e.g. queue the request, IS scheduler policy and out of scope
    /// here).
    ///
    /// On exhaustion, before giving up: check `any_evictable()`. If some
    /// sequence is a candidate, evict it (whole sequence -- see
    /// free_sequence, which already does exactly "free every block a
    /// sequence owns and stop tracking it") and retry the allocation once.
    /// One retry is enough: freeing an entire sequence's blocks always
    /// frees at least one, so the retry cannot fail for the same reason
    /// (pool exhaustion) it just succeeded in fixing.
    ///
    /// IMPORTANT: evicting seq_id ITSELF would be nonsensical (freeing the
    /// very sequence that's asking for more space, then immediately handing
    /// it a block back is at best pointless, at worst confusing) -- if the
    /// only evictable candidate happens to be seq_id itself, treat that the
    /// same as "no candidate" and return None rather than evict it.
    pub fn allocate_block_for(&mut self, seq_id: u64) -> Option<BlockId> {
        match self.allocate() {
            Some(block_id) => {
                self.seq_blocks.entry(seq_id).or_default().push(block_id);
                return Some(block_id);
            }
            None => {
                let victim: Option<u64> = self.evictable.iter().copied().find(|&id| id != seq_id);
                let victim_id: u64 = victim?;
                self.free_sequence(victim_id);
                match self.allocate() {
                    Some(block_id) => {
                        self.seq_blocks.entry(seq_id).or_default().push(block_id);
                        return Some(block_id);
                    }
                    None => {
                        None
                    }
                }
            }
        }
    }

    /// Free ALL blocks owned by `seq_id` (e.g. the sequence finished
    /// generating, or was evicted) and remove it from tracking entirely.
    ///
    /// `HashMap::remove(&key)` removes an entry and returns `Option<V>` (the
    /// value that was there, if any) -- exactly what you need here: get the
    /// Vec<BlockId> that seq_id owned, then free() each one back to the pool.
    pub fn free_sequence(&mut self, seq_id: u64) {
        match self.seq_blocks.remove(&seq_id) {
            Some(block_ids) => {
                for block_id in block_ids {
                    self.free(block_id);
                }
            }
            None => {}
        }
    }

    /// How many blocks does `seq_id` currently own? Returns 0 if the
    /// sequence isn't tracked at all (never allocated, or already freed) --
    /// NOT an error case, so this returns a plain `usize`, not `Option`.
    pub fn blocks_owned_by(&self, seq_id: u64) -> usize {
        match self.seq_blocks.get(&seq_id) {
            Some(block_list) => {
                return block_list.len()
            }
            None => {
                return 0
            }
        }
    }

    /// The core paging lookup: given a logical token `position` within
    /// `seq_id`'s sequence (0-indexed, in generation order), find WHICH
    /// physical block holds it and the offset within that block. This is
    /// exactly OS virtual-memory page translation: block_index = position
    /// / block_size (integer division), offset = position % block_size.
    ///
    /// Returns `None` if `seq_id` isn't tracked, OR if `position` is beyond
    /// how many tokens this sequence has actually allocated space for --
    /// both are "there's no valid answer" cases, not something to panic on
    /// (a scheduler bug asking about a position that doesn't exist yet
    /// should get a clean None to handle, not crash the allocator).
    ///
    /// Steps:
    ///   1. block_index = position / block_size ; offset = position % block_size
    ///      (integer division/modulo on `usize` -- Rust's `/` and `%` on
    ///      unsigned integers work exactly like this, no float conversion
    ///      needed)
    ///   2. Look up seq_id's Vec<BlockId> (self.seq_blocks.get(&seq_id))
    ///   3. Index into that Vec at block_index to get the actual BlockId --
    ///      but a plain `vec[index]` PANICS if index is out of bounds, which
    ///      is exactly what happens if position is further than this
    ///      sequence has allocated. Use `.get(block_index)` instead, which
    ///      returns `Option<&BlockId>` -- safe, no panic, matches the "no
    ///      valid answer -> None" contract this function needs.
    ///   4. If you got a real BlockId, return Some((the_block_id, offset));
    ///      otherwise None at any failed step.
    pub fn locate(&self, seq_id: u64, position: usize) -> Option<(BlockId, usize)> {
        let block_index: usize = position / self.block_size;
        let offset: usize = position % self.block_size;
        match self.seq_blocks.get(&seq_id) {
            Some(block_ids) => {
                match block_ids.get(block_index) {
                    Some(&block_id) => {
                        Some((block_id, offset))
                    }
                    None => {
                        None
                    }
                }
            }
            None => {
                None
            }
        }
    }

    /// Mark `seq_id` as a valid eviction candidate. Idempotent -- marking
    /// something that's already marked is a harmless no-op (that's just how
    /// set insertion works: inserting a value that's already a member
    /// changes nothing).
    pub fn mark_evictable(&mut self, seq_id: u64) {
        self.evictable.insert(seq_id);
    }

    /// Undo mark_evictable -- e.g. the sequence started actively decoding
    /// again and must not be evicted out from under it. Also idempotent:
    /// removing something not in the set is a harmless no-op.
    pub fn unmark_evictable(&mut self, seq_id: u64) {
        self.evictable.remove(&seq_id);
    }

    /// Is `seq_id` currently marked evictable?
    pub fn is_evictable(&self, seq_id: u64) -> bool {
        self.evictable.contains(&seq_id)
    }

    /// If the pool is exhausted, the caller (future scheduler) needs SOME
    /// evictable sequence to consider evicting. This does not pick "the
    /// best" victim by any policy (LRU, priority, etc.) -- picking a good
    /// victim is scheduling policy, which this type deliberately doesn't
    /// own. It just answers "is there ANYONE evictable at all," returning
    /// an arbitrary one's id if so.
    ///
    /// `HashSet` iteration order is unspecified (unlike Vec, which preserves
    /// insertion order) -- `.iter().next()` gets you "some" element, with no
    /// guarantee about which. That's fine here: this method's contract is
    /// explicitly "any candidate," not "the right candidate."
    pub fn any_evictable(&self) -> Option<u64> {
        match self.evictable.iter().next() {
            Some(seq_id) => {
                Some(*seq_id)
            }
            None => {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_with_all_blocks_free() {
        let allocator = BlockAllocator::new(4, 4);
        assert_eq!(allocator.num_free(), 4);
    }

    #[test]
    fn allocate_reduces_free_count() {
        let mut allocator = BlockAllocator::new(4, 4);
        let block = allocator.allocate();
        assert!(block.is_some());
        assert_eq!(allocator.num_free(), 3);
    }

    #[test]
    fn allocate_until_exhausted_returns_none() {
        let mut allocator = BlockAllocator::new(2, 4);
        assert!(allocator.allocate().is_some());
        assert!(allocator.allocate().is_some());
        assert!(allocator.allocate().is_none());
        assert_eq!(allocator.num_free(), 0);
    }

    #[test]
    fn free_returns_block_to_pool() {
        let mut allocator = BlockAllocator::new(1, 4);
        let block = allocator.allocate().unwrap();
        assert_eq!(allocator.num_free(), 0);
        allocator.free(block);
        assert_eq!(allocator.num_free(), 1);
    }

    #[test]
    fn freed_block_can_be_reallocated() {
        let mut allocator = BlockAllocator::new(1, 4);
        let block = allocator.allocate().unwrap();
        allocator.free(block);
        let reallocated = allocator.allocate();
        assert!(reallocated.is_some());
    }

    #[test]
    fn untracked_sequence_owns_zero_blocks() {
        let allocator = BlockAllocator::new(4, 4);
        assert_eq!(allocator.blocks_owned_by(99), 0);
    }

    #[test]
    fn allocate_block_for_tracks_ownership() {
        let mut allocator = BlockAllocator::new(4, 4);
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
        let mut allocator = BlockAllocator::new(4, 4);
        allocator.allocate_block_for(1);
        allocator.allocate_block_for(2);
        allocator.allocate_block_for(2);

        assert_eq!(allocator.blocks_owned_by(1), 1);
        assert_eq!(allocator.blocks_owned_by(2), 2);
    }

    #[test]
    fn allocate_block_for_returns_none_when_exhausted() {
        let mut allocator = BlockAllocator::new(1, 4);
        assert!(allocator.allocate_block_for(1).is_some());
        assert!(allocator.allocate_block_for(1).is_none());
        // the failed attempt must not have corrupted seq 1's existing state
        assert_eq!(allocator.blocks_owned_by(1), 1);
    }

    #[test]
    fn free_sequence_returns_all_its_blocks() {
        let mut allocator = BlockAllocator::new(4, 4);
        allocator.allocate_block_for(1);
        allocator.allocate_block_for(1);
        assert_eq!(allocator.num_free(), 2);

        allocator.free_sequence(1);
        assert_eq!(allocator.num_free(), 4);
        assert_eq!(allocator.blocks_owned_by(1), 0);
    }

    #[test]
    fn free_sequence_does_not_affect_other_sequences() {
        let mut allocator = BlockAllocator::new(4, 4);
        allocator.allocate_block_for(1);
        allocator.allocate_block_for(2);

        allocator.free_sequence(1);

        assert_eq!(allocator.blocks_owned_by(1), 0);
        assert_eq!(allocator.blocks_owned_by(2), 1);
        assert_eq!(allocator.num_free(), 3);
    }

    #[test]
    fn free_sequence_on_untracked_id_is_a_harmless_no_op() {
        let mut allocator = BlockAllocator::new(4, 4);
        allocator.free_sequence(404); // must not panic
        assert_eq!(allocator.num_free(), 4);
    }

    #[test]
    fn locate_untracked_sequence_returns_none() {
        let allocator = BlockAllocator::new(4, 4);
        assert!(allocator.locate(1, 0).is_none());
    }

    #[test]
    fn locate_first_block_first_offset() {
        let mut allocator = BlockAllocator::new(4, 4); // block_size = 4
        let b0 = allocator.allocate_block_for(1).unwrap();
        // position 0 -> block_index 0, offset 0 -> the very first block, first slot
        assert_eq!(allocator.locate(1, 0), Some((b0, 0)));
    }

    #[test]
    fn locate_within_first_block() {
        let mut allocator = BlockAllocator::new(4, 4);
        let b0 = allocator.allocate_block_for(1).unwrap();
        // position 3 is still within block_size=4 -> same block, offset 3
        assert_eq!(allocator.locate(1, 3), Some((b0, 3)));
    }

    #[test]
    fn locate_rolls_over_into_second_block() {
        let mut allocator = BlockAllocator::new(4, 4);
        let _b0 = allocator.allocate_block_for(1).unwrap();
        let b1 = allocator.allocate_block_for(1).unwrap();
        // position 5: block_index = 5/4 = 1, offset = 5%4 = 1 -> second
        // allocated block (b1), offset 1 -- matches the worked example.
        assert_eq!(allocator.locate(1, 5), Some((b1, 1)));
    }

    #[test]
    fn locate_beyond_allocated_range_returns_none() {
        let mut allocator = BlockAllocator::new(4, 4);
        allocator.allocate_block_for(1); // only 1 block = positions 0..4 valid
        // position 4 needs a second block that was never allocated
        assert!(allocator.locate(1, 4).is_none());
    }

    #[test]
    fn not_evictable_by_default() {
        let mut allocator = BlockAllocator::new(4, 4);
        allocator.allocate_block_for(1);
        assert!(!allocator.is_evictable(1));
    }

    #[test]
    fn mark_evictable_makes_it_evictable() {
        let mut allocator = BlockAllocator::new(4, 4);
        allocator.allocate_block_for(1);
        allocator.mark_evictable(1);
        assert!(allocator.is_evictable(1));
    }

    #[test]
    fn mark_evictable_is_idempotent() {
        let mut allocator = BlockAllocator::new(4, 4);
        allocator.mark_evictable(1);
        allocator.mark_evictable(1); // marking twice must not panic or error
        assert!(allocator.is_evictable(1));
    }

    #[test]
    fn unmark_evictable_reverses_mark() {
        let mut allocator = BlockAllocator::new(4, 4);
        allocator.mark_evictable(1);
        allocator.unmark_evictable(1);
        assert!(!allocator.is_evictable(1));
    }

    #[test]
    fn unmark_evictable_on_unmarked_id_is_a_harmless_no_op() {
        let mut allocator = BlockAllocator::new(4, 4);
        allocator.unmark_evictable(404); // must not panic
        assert!(!allocator.is_evictable(404));
    }

    #[test]
    fn any_evictable_returns_none_when_nothing_marked() {
        let mut allocator = BlockAllocator::new(4, 4);
        allocator.allocate_block_for(1);
        assert!(allocator.any_evictable().is_none());
    }

    #[test]
    fn any_evictable_returns_a_marked_seq_id() {
        let mut allocator = BlockAllocator::new(4, 4);
        allocator.mark_evictable(1);
        assert_eq!(allocator.any_evictable(), Some(1));
    }

    #[test]
    fn any_evictable_after_unmarking_only_marked_is_none() {
        let mut allocator = BlockAllocator::new(4, 4);
        allocator.mark_evictable(1);
        allocator.unmark_evictable(1);
        assert!(allocator.any_evictable().is_none());
    }

    #[test]
    fn allocate_block_for_still_fails_when_exhausted_with_no_evictable_victim() {
        let mut allocator = BlockAllocator::new(1, 4);
        allocator.allocate_block_for(1); // pool now exhausted, seq 1 not evictable
        assert!(allocator.allocate_block_for(2).is_none());
        // failed attempt must not have disturbed seq 1's state
        assert_eq!(allocator.blocks_owned_by(1), 1);
    }

    #[test]
    fn allocate_block_for_evicts_a_marked_victim_to_make_room() {
        let mut allocator = BlockAllocator::new(1, 4);
        allocator.allocate_block_for(1);
        allocator.mark_evictable(1);

        let result = allocator.allocate_block_for(2);

        assert!(result.is_some());
        assert_eq!(allocator.blocks_owned_by(1), 0); // victim fully evicted
        assert_eq!(allocator.blocks_owned_by(2), 1); // requester got its block
    }

    #[test]
    fn allocate_block_for_does_not_evict_itself() {
        // seq 1 is the ONLY tracked sequence, marked evictable, and is also
        // the one requesting more space. It must not be allowed to evict
        // itself to satisfy its own request.
        let mut allocator = BlockAllocator::new(1, 4);
        allocator.allocate_block_for(1);
        allocator.mark_evictable(1);

        assert!(allocator.allocate_block_for(1).is_none());
        // seq 1 must be untouched -- not evicted, still owns its 1 block
        assert_eq!(allocator.blocks_owned_by(1), 1);
    }

    #[test]
    fn allocate_block_for_prefers_a_different_evictable_victim_over_self() {
        let mut allocator = BlockAllocator::new(2, 4);
        allocator.allocate_block_for(1);
        allocator.allocate_block_for(2);
        allocator.mark_evictable(1);
        allocator.mark_evictable(2); // seq 2 itself is ALSO evictable

        // seq 2 requests more space -- pool is exhausted (2 blocks, both
        // owned). A valid victim exists that ISN'T seq 2 (namely seq 1), so
        // this must succeed by evicting seq 1, not by evicting seq 2 itself.
        let result = allocator.allocate_block_for(2);

        assert!(result.is_some());
        assert_eq!(allocator.blocks_owned_by(1), 0);
        assert_eq!(allocator.blocks_owned_by(2), 2);
    }
}
