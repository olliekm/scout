"""
THROWAWAY prototype -- not part of the real engine (that's Rust, per AGENT.md).
Purpose only: understand the asyncio.Queue + asyncio.Future pattern that lets
many concurrent "requests" rendezvous into one batch, in isolation, with no
model/tokenizer complexity mixed in.

Run it:
  python bench/scratch/asyncio_batching_toy.py

Read it top to bottom, then read the walkthrough at the bottom of this file.
"""

import asyncio
import random
import time

MAX_BATCH_SIZE = 4
MAX_WAIT_S = 0.5

# The shared handoff point. Producers (fake "requests") put work items here.
# The consumer (the "batcher" background task) is the only thing that reads
# from it. asyncio.Queue is safe to share across coroutines running on the
# same event loop -- that's the whole reason it exists.
queue: asyncio.Queue = asyncio.Queue()


async def fake_request(request_id: int):
    """Stands in for one incoming HTTP request. It has some "work" (a number
    to square) and needs to get back exactly its own result -- not anyone
    else's -- even though the actual computation may happen batched together
    with other requests it knows nothing about.
    """
    work = random.randint(1, 20)

    # A Future is a promise: "something will eventually put a value here."
    # This request creates its OWN future, hands it to the batcher via the
    # queue, then awaits it. Awaiting a future suspends *this* coroutine only
    # -- the event loop is free to run other requests/the batcher meanwhile.
    future = asyncio.get_event_loop().create_future()

    t_submit = time.perf_counter()
    await queue.put((request_id, work, future))

    result = await future  # blocks here until the batcher resolves it
    t_done = time.perf_counter()

    print(
        f"[request {request_id}] work={work} -> result={result} "
        f"(waited {t_done - t_submit:.2f}s)"
    )


async def batcher():
    """Runs forever in the background. Pulls up to MAX_BATCH_SIZE items off
    the queue, OR whatever has arrived within MAX_WAIT_S -- whichever comes
    first -- then does ONE unit of "batched work" and resolves every future
    in that batch with its own answer.

    This loop is the direct analogue of what a real batching engine's
    scheduler does: collect -> form a batch -> run it once -> fan results
    back out to callers.
    """
    while True:
        batch = []
        deadline = time.perf_counter() + MAX_WAIT_S

        while len(batch) < MAX_BATCH_SIZE:
            timeout = deadline - time.perf_counter()
            if timeout <= 0:
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=timeout)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        if not batch:
            continue  # nothing arrived in this window, try again

        request_ids = [item[0] for item in batch]
        print(f"\n>>> running batch of {len(batch)}: requests {request_ids}")

        # Stand-in for "run model.generate() on the whole batch at once."
        # The point being demonstrated: this sleep happens ONCE per batch,
        # not once per request -- that's the entire value proposition of
        # batching.
        await asyncio.sleep(0.3)

        for request_id, work, future in batch:
            future.set_result(work * work)


async def main():
    batcher_task = asyncio.create_task(batcher())

    # Fire off 10 "requests" with staggered arrival times, to see batches
    # form from partial groups (timeout-triggered) as well as full groups
    # (size-triggered).
    requesters = []
    for i in range(10):
        requesters.append(asyncio.create_task(fake_request(i)))
        await asyncio.sleep(random.uniform(0.05, 0.25))

    await asyncio.gather(*requesters)
    batcher_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())


# --- Walkthrough -------------------------------------------------------
#
# 1. Each fake_request() creates its own Future and puts (id, work, future)
#    on the shared queue, then awaits that future. It does NOT know or care
#    how its work gets done -- batched with others, alone, whatever.
#
# 2. batcher() is the only consumer. It greedily drains the queue up to
#    MAX_BATCH_SIZE items, but won't wait past MAX_WAIT_S total for the
#    batch to fill -- so a batch might be size 4 (hit the cap) or smaller
#    (timeout fired first because traffic was sparse).
#
# 3. The "generate() call" is simulated with ONE asyncio.sleep(0.3) for the
#    whole batch, not one per request. That's the point: N requests sharing
#    ~1 unit of GPU work instead of N units.
#
# 4. future.set_result(...) is what wakes up the specific fake_request()
#    coroutine that's awaiting that exact future -- this is how each caller
#    gets back ONLY its own answer, even though the work happened as a group.
#
# When you port this to server.py:
#   - fake_request()  -> the FastAPI endpoint handler (async def generate)
#   - work            -> (prompt, max_tokens)
#   - the sleep        -> tokenize the whole batch (with padding +
#                          attention_mask), run model.generate() once,
#                          decode each row back to text
#   - future.set_result -> resolve with that request's decoded text
