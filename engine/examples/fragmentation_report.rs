//! Compares memory utilization between two KV-cache allocation strategies
//! against the same synthetic workload:
//!   - naive: each sequence reserves blocks for the worst-case max sequence
//!     length upfront, whether or not it ever generates that many tokens
//!     (the pre-paging strategy -- reserve the worst case per request)
//!   - paged: each sequence grows its block allocation lazily, via the real
//!     `BlockAllocator`, only as it actually accumulates tokens
//!
//! This is the core memory-efficiency claim step 3 (paged KV cache) exists
//! to prove -- internal fragmentation from worst-case reservation is the
//! problem paging solves, and is the headline result the vLLM paper this
//! milestone is modeled on reports. No GPU needed: this is pure bookkeeping,
//! runs locally.
//!
//! Run with: cargo run --example fragmentation_report -p engine

use engine::block_allocator::BlockAllocator;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, LogNormal};

/// Fixed seed so results are reproducible across runs -- a fragmentation
/// story that changes every time you run it isn't a story.
const SEED: u64 = 42;

/// Sample `n` sequence lengths (in tokens), deliberately skewed like real
/// chat/code workloads: most sequences short, with a long tail of much
/// longer ones. Mirrors why `bench_sharegpt.py` uses real ShareGPT data
/// instead of a uniform length spread -- uniform lengths under-stress the
/// exact fragmentation gap paging is supposed to close (naive reservation
/// only looks bad when most requests use far less than the worst case).
///
/// Steps:
///   1. Build a seeded RNG: `StdRng::seed_from_u64(SEED)`
///   2. Build a `LogNormal` distribution (`LogNormal::new(mean, std_dev)`)
///      with parameters chosen so most samples land "short" (tens to a few
///      hundred tokens) with occasional much longer outliers
///   3. Sample n values via `dist.sample(&mut rng)`, round to `usize` token
///      counts, clamp to >= 1 (a zero-length sequence doesn't make sense)
fn sample_sequence_lengths(n: usize) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let dist = LogNormal::new(5.01_f64, 1.2_f64).unwrap();
    (0..n).map(|_| (dist.sample(&mut rng).round() as usize).clamp(1, 99999)).collect()
}

/// Total blocks a NAIVE allocator would need to reserve upfront for this
/// workload: every sequence reserves `ceil(max_seq_len / block_size)`
/// blocks regardless of how many tokens it actually ends up using.
fn naive_blocks_required(num_sequences: usize, max_seq_len: usize, block_size: usize) -> usize {
    num_sequences * max_seq_len.div_ceil(block_size)
}

/// Total blocwks a PAGED allocator actually consumes for this workload: each
/// sequence grows its block allocation to cover only its true length, via
/// the real `BlockAllocator` (not a formula) -- this exercises the actual
/// `allocate_block_for` path the rest of the engine will call, rather than
/// just computing what it *should* use.
///
/// Steps:
///   1. Construct a `BlockAllocator` with `num_pool_blocks` blocks
///   2. For each sequence (0..lengths.len(), used as seq_id), call
///      `allocate_block_for` repeatedly until it owns
///      `ceil(length / block_size)` blocks -- if any call returns `None`,
///      the pool is undersized for this workload; panic with a clear
///      message rather than silently reporting a wrong number, since a
///      utilization report built on a failed allocation isn't meaningful
///   3. Return `num_pool_blocks - allocator.num_free()` (blocks actually
///      consumed at the end)
fn paged_blocks_used(lengths: &[usize], block_size: usize, num_pool_blocks: usize) -> usize {
    let mut block_allocator = BlockAllocator::new(num_pool_blocks, block_size);
    for seq_id in (0..lengths.len()) {
        let blocks_needed = lengths[seq_id].div_ceil(block_size);
        for i in (0..blocks_needed) { 
            match block_allocator.allocate_block_for(seq_id as u64) {
                Some(_) => {}
                None => {panic!("undersized")}
            }
        }
    }
    num_pool_blocks - block_allocator.num_free()
}

fn main() {
    let num_sequences = 64;
    let block_size = 16; // tokens per block -- matches vLLM's default paged-attention block size
    let max_seq_len = 32768; // Qwen2.5-Coder-7B-Instruct's max context length (see AGENT.md) -- the worst case naive reserves per sequence

    let lengths = sample_sequence_lengths(num_sequences);
    let naive = naive_blocks_required(num_sequences, max_seq_len, block_size);
    // Size the pool to `naive` so paging can't fail here -- this report is
    // about the utilization gap, not exhaustion (that's a separate question).
    let paged = paged_blocks_used(&lengths, block_size, naive);

    let utilization = 100.0 * paged as f64 / naive as f64;

    println!("workload: {num_sequences} sequences, block_size = {block_size} tokens, naive worst-case reservation = {max_seq_len} tokens/sequence");
    println!("naive blocks required: {naive}");
   
    println!("paged blocks used:     {paged}");
    println!("utilization:           {utilization:.2}% (paged / naive)");
    println!("fragmentation avoided: {:.2}% of blocks naive would have reserved but never used", 100.0 - utilization);
}
