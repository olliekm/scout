"""
Step 1 baseline benchmark harness.

Measures the client-observed metrics that actually get scrutinized for an
inference server:
  - TTFT (time to first token)   -- queueing + prefill latency
  - TPOT (time per output token) -- steady-state decode latency
  - throughput                   -- output tokens / sec, end to end

Hits /generate_stream (not /generate) because only the streamed endpoint lets
the client timestamp individual tokens. server_time_s from /generate is a
useful cross-check but can't give you TTFT/TPOT.

Requests are sent strictly sequentially, one at a time. server.py has no
batching -- concurrent requests would just queue behind the GIL/model.generate()
call, so concurrency here would measure queueing delay, not generation speed.
That's a deliberate property of static/continuous batching stages later, not
something to fake at the naive-baseline stage.

Usage:
  python bench/bench.py --url http://localhost:8000 --repeats 3
  python bench/bench.py --url http://localhost:8000 --out results/step1.json
"""

import argparse
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
    """Send one streaming request, timestamp each token as it arrives.

    Returns per-request metrics. Any single request failing is a bug worth
    surfacing immediately (raise), not silently skipping -- a partial
    benchmark run is worse than a loud crash.
    """
    t_start = time.perf_counter()
    resp = requests.post(
        f"{url}/generate_stream",
        json={"prompt": prompt, "max_tokens": max_tokens},
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    token_times = []
    for line in resp.iter_lines():
        if not line:
            continue
        json.loads(line)  # validate shape; raises if server sends garbage
        token_times.append(time.perf_counter())

    if not token_times:
        raise RuntimeError(f"No tokens received for prompt: {prompt!r}")

    ttft = token_times[0] - t_start
    total_time = token_times[-1] - t_start
    output_tokens = len(token_times)

    # TPOT = mean gap between consecutive tokens, i.e. steady-state decode
    # speed *excluding* the first-token/prefill latency already captured by
    # TTFT. If only one token came back there's no "steady state" to measure.
    if output_tokens > 1:
        gaps = [b - a for a, b in zip(token_times, token_times[1:])]
        tpot = statistics.mean(gaps)
    else:
        tpot = None

    return {
        "ttft_s": ttft,
        "tpot_s": tpot,
        "total_time_s": total_time,
        "output_tokens": output_tokens,
        "tokens_per_sec": output_tokens / total_time if total_time > 0 else 0.0,
    }


def percentile(values, p):
    """Nearest-rank percentile. Good enough at n=few-dozen; don't reach for
    numpy/scipy interpolation methods here, they'd be false precision."""
    s = sorted(values)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


def summarize(runs: list[dict]) -> dict:
    ttfts = [r["ttft_s"] for r in runs]
    tpots = [r["tpot_s"] for r in runs if r["tpot_s"] is not None]
    tps = [r["tokens_per_sec"] for r in runs]
    return {
        "n": len(runs),
        "ttft_s": {
            "mean": statistics.mean(ttfts),
            "p50": percentile(ttfts, 50),
            "p95": percentile(ttfts, 95),
        },
        "tpot_s": {
            "mean": statistics.mean(tpots) if tpots else None,
            "p50": percentile(tpots, 50) if tpots else None,
            "p95": percentile(tpots, 95) if tpots else None,
        },
        "tokens_per_sec": {
            "mean": statistics.mean(tps),
            "p50": percentile(tps, 50),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--repeats", type=int, default=3, help="repeats per prompt")
    ap.add_argument("--out", default=None, help="optional path to dump raw JSON results")
    args = ap.parse_args()

    prompts = load_prompts()

    # Health check first -- fail fast with a clear message rather than a
    # confusing connection error mid-benchmark.
    health = requests.get(f"{args.url}/health", timeout=10).json()
    print(f"[health] model={health['model']} dtype={health['dtype']}")

    all_runs = []
    by_prompt = {}

    for spec in prompts:
        name, prompt, max_tokens = spec["name"], spec["prompt"], spec["max_tokens"]
        runs = []
        for i in range(args.repeats):
            r = run_one(args.url, prompt, max_tokens)
            runs.append(r)
            print(
                f"[{name} {i+1}/{args.repeats}] "
                f"ttft={r['ttft_s']*1000:.0f}ms "
                f"tpot={(r['tpot_s'] or 0)*1000:.1f}ms/tok "
                f"tok/s={r['tokens_per_sec']:.1f}"
            )
        by_prompt[name] = summarize(runs)
        all_runs.extend(runs)

    overall = summarize(all_runs)

    print("\n=== per-prompt summary ===")
    for name, s in by_prompt.items():
        print(
            f"{name:16s} ttft_p50={s['ttft_s']['p50']*1000:6.0f}ms "
            f"tpot_p50={(s['tpot_s']['p50'] or 0)*1000:6.1f}ms/tok "
            f"tok/s={s['tokens_per_sec']['mean']:6.1f}"
        )

    print("\n=== overall ===")
    print(json.dumps(overall, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"by_prompt": by_prompt, "overall": overall}, f, indent=2)
        print(f"\n[saved raw results to {args.out}]")


if __name__ == "__main__":
    main()
