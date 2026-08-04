"""
Step 2 concurrent benchmark harness.

Unlike bench.py (sequential, streaming, measures TTFT/TPOT against the naive
server.py), this fires a burst of N requests at server_static_batch.py's
/generate ALL AT ONCE, so they actually land in the same batching window and
get grouped by the batcher -- the whole point of static batching only shows
up under concurrent load, not sequential.

/generate on the batched server is non-streaming (it can only return once
the whole batch's generate() call finishes), so there's no per-token TTFT/
TPOT to measure here. What we measure instead:
  - per-request wall-clock latency (submit -> response), including time
    spent waiting for a batch to form
  - end-to-end throughput: total output tokens across the whole burst,
    divided by the time for the LAST request in the burst to finish
  - batch_size the server reports each request was grouped into (echoed
    back in the response) -- confirms grouping actually happened, not just
    inferred from timing

Usage:
  python bench/bench_concurrent.py --url http://localhost:8002 --burst 8
  python bench/bench_concurrent.py --url http://localhost:8002 --burst 8 --out results/step2_burst8.json
"""

import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path

import requests

PROMPTS_PATH = Path(__file__).parent / "prompts.json"


def load_prompts():
    with open(PROMPTS_PATH) as f:
        return json.load(f)


def run_one(url: str, prompt: str, max_tokens: int) -> dict:
    """Send one non-streaming request, time it end to end. Runs inside a
    thread pool so multiple calls are in flight on the wire simultaneously
    -- `requests` is blocking, so real concurrency here comes from threads,
    not asyncio (there's no event loop on the client side worth building
    for a load generator this simple)."""
    t_start = time.perf_counter()
    resp = requests.post(
        f"{url}/generate",
        json={"prompt": prompt, "max_tokens": max_tokens},
        timeout=120,
    )
    resp.raise_for_status()
    latency = time.perf_counter() - t_start

    body = resp.json()
    return {
        "latency_s": latency,
        "batch_size": body.get("batch_size"),
        "text": body["text"],
    }


def run_burst(url: str, prompts: list[dict]) -> tuple[list[dict], float]:
    """Fire every prompt in `prompts` at the server at once. Returns
    (per-request results, wall-clock time for the whole burst to finish)."""
    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [
            pool.submit(run_one, url, p["prompt"], p["max_tokens"]) for p in prompts
        ]
        results = [f.result() for f in futures]
    burst_time = time.perf_counter() - t_start
    return results, burst_time


def percentile(values, p):
    s = sorted(values)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8002")
    ap.add_argument(
        "--burst",
        type=int,
        default=8,
        help="number of requests to fire simultaneously (cycles through prompts.json if burst > len(prompts))",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompts = load_prompts()
    # Cycle through the fixed prompt set to build a burst of the requested size.
    burst_prompts = [prompts[i % len(prompts)] for i in range(args.burst)]

    health = requests.get(f"{args.url}/health", timeout=10).json()
    print(f"[health] model={health['model']} dtype={health['dtype']} mode={health.get('mode')}")
    print(f"[burst] firing {args.burst} concurrent requests...")

    results, burst_time = run_burst(args.url, burst_prompts)

    latencies = [r["latency_s"] for r in results]
    batch_sizes = [r["batch_size"] for r in results]
    total_output_tokens = sum(
        len(r["text"].split()) for r in results
    )  # rough proxy; good enough for a throughput comparison, not exact token count

    print("\n=== per-request ===")
    for i, r in enumerate(results):
        print(f"[{i}] latency={r['latency_s']*1000:.0f}ms batch_size={r['batch_size']}")

    summary = {
        "burst_size": args.burst,
        "burst_wall_time_s": burst_time,
        "latency_s": {
            "mean": statistics.mean(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
        "batch_sizes_seen": sorted(set(batch_sizes)),
        "approx_total_output_tokens": total_output_tokens,
        "approx_throughput_tok_s": total_output_tokens / burst_time if burst_time > 0 else 0.0,
    }

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"results": results, "summary": summary}, f, indent=2)
        print(f"\n[saved raw results to {args.out}]")


if __name__ == "__main__":
    main()
