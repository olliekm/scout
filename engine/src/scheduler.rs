//! Admission control -- the first, smallest slice of the continuous-batching
//! scheduler from AGENT.md. Standard decomposition (matches vLLM's own
//! split between admission and per-iteration dispatch, and the classic OS
//! long-term-scheduler / short-term-scheduler split): admission answers
//! "can this new request be accepted right now, given current resource
//! pressure," which is a distinct, independently-testable question from
//! "what should already-running requests do this round" (the decode loop,
//! built later on top of this).
//!
//! New Rust ideas versus block_allocator.rs:
//!
//! - This struct OWNS a `BlockAllocator` (a field of that type, not a
//!   reference to one) -- when a `Scheduler` is dropped, its `BlockAllocator`
//!   is dropped with it. This is Rust's default: types are composed by
//!   value, and ownership nests naturally, no manual memory management.
//! - `crate::block_allocator::BlockAllocator` is how you refer to a type
//!   defined in a SIBLING module -- `crate::` means "starting from this
//!   crate's root" (i.e. from lib.rs), then walk the module path. You'll
//!   need a `use crate::block_allocator::BlockAllocator;` line to avoid
//!   writing the full path every time, same idea as `use std::collections::HashMap`.

use crate::block_allocator::BlockAllocator;
use std::collections::HashSet;

pub struct Scheduler {
    allocator: BlockAllocator,

    // YOUR FIELD HERE.
    //
    // The scheduler needs to know WHICH sequence ids it currently considers
    // "admitted" / running -- distinct from what the allocator tracks
    // (allocator only knows about sequences that own at least one block;
    // the scheduler is the thing that will eventually decide admission,
    // eviction policy, run order, etc., using the allocator as its
    // mechanism, not its policy).
    //
    // A `HashSet<u64>` of admitted sequence ids is enough for this slice.
    // Add:
    //   admitted: HashSet<u64>,
}

impl Scheduler {
    /// Construct a Scheduler wrapping a fresh BlockAllocator with the given
    /// capacity. Mirrors BlockAllocator::new's shape -- this is where you
    /// practice constructing one struct that contains another.
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        todo!("build a BlockAllocator::new(num_blocks, block_size), wrap it and an empty admitted set in Self")
    }

    /// Try to admit a new sequence. Returns true if admission succeeded
    /// (the sequence now owns exactly one block and is tracked as
    /// "admitted"), false if it could not be admitted (allocator had no
    /// room and no valid eviction victim -- see BlockAllocator::
    /// allocate_block_for's own eviction logic, which this can just defer
    /// to rather than re-implementing).
    ///
    /// Every admitted sequence starts by claiming ONE block (its first
    /// chunk of prompt/KV-cache space) -- growing further as it generates
    /// is a later concern (the decode loop), not admission's job.
    ///
    /// Steps:
    ///   1. Call self.allocator.allocate_block_for(seq_id).
    ///   2. On Some(_): record seq_id in self.admitted, return true.
    ///   3. On None: return false (nothing to clean up -- allocate_block_for
    ///      guarantees it leaves everything untouched on failure, per its
    ///      own doc comment).
    pub fn try_admit(&mut self, seq_id: u64) -> bool {
        todo!("allocate a first block for seq_id via the allocator; on success, mark it admitted")
    }

    /// Is `seq_id` currently admitted (tracked as running by this scheduler)?
    pub fn is_admitted(&self, seq_id: u64) -> bool {
        todo!("check membership in self.admitted")
    }

    /// How many blocks are free in the underlying allocator right now.
    /// Thin pass-through -- useful for tests and for a future admission
    /// policy that wants to check capacity before even trying.
    pub fn num_free_blocks(&self) -> usize {
        todo!("delegate to self.allocator.num_free()")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admits_when_room_available() {
        let mut scheduler = Scheduler::new(4, 4);
        assert!(scheduler.try_admit(1));
        assert!(scheduler.is_admitted(1));
        assert_eq!(scheduler.num_free_blocks(), 3);
    }

    #[test]
    fn not_admitted_before_try_admit() {
        let scheduler = Scheduler::new(4, 4);
        assert!(!scheduler.is_admitted(1));
    }

    #[test]
    fn rejects_when_pool_exhausted_and_nothing_evictable() {
        let mut scheduler = Scheduler::new(1, 4);
        assert!(scheduler.try_admit(1)); // takes the only block
        assert!(!scheduler.try_admit(2)); // no room, seq 1 not evictable
        assert!(!scheduler.is_admitted(2));
        assert_eq!(scheduler.num_free_blocks(), 0);
    }

    #[test]
    fn admitting_same_seq_id_twice_still_only_uses_one_block_worth_of_admission_state() {
        // NOTE: this deliberately probes a real design question the doc
        // comments don't pin down -- what SHOULD happen if try_admit is
        // called again for an id that's already admitted? Two defensible
        // answers exist (allocate it ANOTHER block and grow it, vs. treat
        // re-admission as a no-op/error since admission should happen once
        // per sequence). Implement whichever you think is right, then we'll
        // check whether this test's expectation matches your choice --
        // don't force your implementation to fit this test blindly.
        let mut scheduler = Scheduler::new(4, 4);
        assert!(scheduler.try_admit(1));
        let free_after_first = scheduler.num_free_blocks();
        scheduler.try_admit(1);
        // if re-admission is a no-op, free block count shouldn't change again
        assert_eq!(scheduler.num_free_blocks(), free_after_first);
    }
}
