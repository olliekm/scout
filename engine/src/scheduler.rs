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
use std::collections::VecDeque;

pub struct Scheduler {
    allocator: BlockAllocator,
    admitted: HashSet<u64>,
    waiting: VecDeque<u64>,
    // YOUR FIELD HERE.
    //
    // Continuous batching's whole point is decoupling "a request exists"
    // from "a request is currently running" -- static batching (step 2)
    // could only admit new requests once the ENTIRE batch drained, because
    // it had no concept of a request waiting its turn. This field is that
    // concept: requests `submit`'d but not yet admitted sit here until
    // `dispatch` finds room for them.
    //
    // `VecDeque<u64>` (not `Vec<u64>`) specifically because dispatch needs
    // FIFO order -- first submitted, first admitted, for fairness -- and
    // `VecDeque::pop_front`/`push_front` are O(1), unlike `Vec::remove(0)`
    // which would shift every remaining element down on each call.
    //
    // Add:
    //   waiting: VecDeque<u64>,

}

impl Scheduler {
    /// Construct a Scheduler wrapping a fresh BlockAllocator with the given
    /// capacity. Mirrors BlockAllocator::new's shape -- this is where you
    /// practice constructing one struct that contains another.
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let allocator: BlockAllocator = BlockAllocator::new(num_blocks, block_size);
        let admitted: HashSet<u64> = HashSet::new();
        let waiting: VecDeque<u64> = VecDeque::new();
        Self { allocator, admitted, waiting }
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
        if self.is_admitted(seq_id) {
            return false
        }
        match self.allocator.allocate_block_for(seq_id) {
            Some(block_id) => {
                self.admitted.insert(seq_id);
                return true
            }
            None => {
                return false
            }
        }
    }

    /// Is `seq_id` currently admitted (tracked as running by this scheduler)?
    pub fn is_admitted(&self, seq_id: u64) -> bool {
        self.admitted.contains(&seq_id)
    }

    /// How many blocks are free in the underlying allocator right now.
    /// Thin pass-through -- useful for tests and for a future admission
    /// policy that wants to check capacity before even trying.
    pub fn num_free_blocks(&self) -> usize {
        self.allocator.num_free()
    }

    /// Record that `seq_id` wants to run, without trying to admit it yet.
    /// Unlike `try_admit`, this never fails and never touches the
    /// allocator -- it just queues the request in arrival order. Whether
    /// it actually gets a block this round, or has to wait behind others,
    /// is `dispatch`'s job, not this one's. Splitting "a request arrived"
    /// from "a request is running" into two separate steps is exactly what
    /// makes batching CONTINUOUS instead of static.
    pub fn submit(&mut self, seq_id: u64) {
        self.waiting.push_back(seq_id);
    }

    /// Admit as many waiting sequences as currently fit, in FIFO order.
    /// This is the "fill empty slots as they appear" half of continuous
    /// batching -- call it once per scheduling iteration, typically right
    /// after `complete` has freed up room from a finished sequence.
    ///
    /// Steps:
    ///   1. Loop: pop the front of `self.waiting` (`VecDeque::pop_front`).
    ///      If the queue is empty, stop -- nothing left to admit.
    ///   2. Call `self.try_admit(seq_id)` on the popped id.
    ///   3. On success: record it (you'll return these), then loop back to
    ///      step 1 -- there might still be room for the next one.
    ///   4. On failure: the pool didn't have room for this one. Push it
    ///      back onto the FRONT of `self.waiting` (`VecDeque::push_front`)
    ///      so it's not lost and stays first in line for the NEXT dispatch
    ///      call, then stop the loop -- if the pool couldn't fit this
    ///      (longest-waiting) request, that's as far as this round goes.
    ///   5. Return the ids that got admitted this round, in the order they
    ///      were admitted.
    pub fn dispatch(&mut self) -> Vec<u64> {
        let mut dispatched: Vec<u64> = Vec::new();
        while let Some(seq_id) = self.waiting.pop_front() {
            if self.try_admit(seq_id) {
                dispatched.push(seq_id);
            } else {
                self.waiting.push_front(seq_id);
                break;
            }
        }
        dispatched
    }

    /// Mark a running sequence as finished (hit EOS, hit max tokens,
    /// whatever the caller's stopping condition is) -- frees every block
    /// it owns and stops tracking it as admitted. This is the OTHER half
    /// of the rolling window: `dispatch` fills empty slots, `complete` is
    /// what CREATES one mid-batch, instead of waiting for every sequence
    /// in the batch to finish before anyone new gets in (step 2's
    /// behavior). `complete` only frees -- call `dispatch` again
    /// afterward to actually backfill the room it opened up.
    ///
    /// Steps:
    ///   1. `self.allocator.free_sequence(seq_id)` -- frees every block it
    ///      owned, regardless of how many.
    ///   2. `self.admitted.remove(&seq_id)` -- stop tracking it as running.
    pub fn complete(&mut self, seq_id: u64) {
        self.allocator.free_sequence(seq_id);
        self.admitted.remove(&seq_id);
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

    #[test]
    fn submit_does_not_admit_immediately() {
        let mut scheduler = Scheduler::new(4, 4);
        scheduler.submit(1);
        assert!(!scheduler.is_admitted(1));
        assert_eq!(scheduler.num_free_blocks(), 4);
    }

    #[test]
    fn dispatch_with_nothing_waiting_returns_empty() {
        let mut scheduler = Scheduler::new(4, 4);
        assert_eq!(scheduler.dispatch(), Vec::<u64>::new());
    }

    #[test]
    fn dispatch_admits_submitted_sequences_when_room() {
        let mut scheduler = Scheduler::new(4, 4);
        scheduler.submit(1);
        scheduler.submit(2);
        assert_eq!(scheduler.dispatch(), vec![1, 2]);
        assert!(scheduler.is_admitted(1));
        assert!(scheduler.is_admitted(2));
        assert_eq!(scheduler.num_free_blocks(), 2);
    }

    #[test]
    fn dispatch_stops_at_first_rejection_and_requeues_it() {
        let mut scheduler = Scheduler::new(1, 4); // room for exactly one running sequence
        scheduler.submit(1);
        scheduler.submit(2);

        assert_eq!(scheduler.dispatch(), vec![1]); // only seq 1 fit
        assert!(!scheduler.is_admitted(2));

        // seq 2 must still be waiting, not dropped -- a second dispatch
        // with no new room should admit nobody, not silently lose it
        assert!(scheduler.dispatch().is_empty());
    }

    #[test]
    fn requeued_sequence_stays_first_in_line_for_the_next_dispatch() {
        // Fairness check on the push_front-on-rejection behavior: a
        // sequence that's been waiting longer must get the next freed slot
        // ahead of one submitted more recently, even though both are
        // sitting in the queue when that slot opens up.
        let mut scheduler = Scheduler::new(1, 4);
        scheduler.submit(1);
        scheduler.submit(2);
        scheduler.dispatch(); // admits 1; seq 2 requeued at the front

        scheduler.submit(3); // arrives after seq 2 was already waiting
        scheduler.complete(1); // frees the only slot

        assert_eq!(scheduler.dispatch(), vec![2]); // 2 wins over 3, not FIFO by arrival-to-this-dispatch-call but by original submit order
    }

    #[test]
    fn complete_frees_blocks_and_unmarks_admitted() {
        let mut scheduler = Scheduler::new(4, 4);
        scheduler.try_admit(1);
        assert_eq!(scheduler.num_free_blocks(), 3);

        scheduler.complete(1);
        assert!(!scheduler.is_admitted(1));
        assert_eq!(scheduler.num_free_blocks(), 4);
    }

    #[test]
    fn complete_then_dispatch_backfills_a_waiting_sequence_without_draining_the_whole_batch() {
        // The core continuous-batching claim: seq 2 gets in as soon as seq
        // 1 finishes, without waiting for every other running sequence to
        // drain first -- that "wait for the whole batch" behavior is what
        // static batching (step 2) did, and this is the fix.
        let mut scheduler = Scheduler::new(1, 4);
        scheduler.try_admit(1);
        scheduler.submit(2);

        assert!(scheduler.dispatch().is_empty()); // no room yet, seq 2 still waiting

        scheduler.complete(1); // seq 1 finishes, frees its block
        assert_eq!(scheduler.dispatch(), vec![2]);
        assert!(scheduler.is_admitted(2));
        assert!(!scheduler.is_admitted(1));
    }
}
