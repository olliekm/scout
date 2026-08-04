"""
ShareGPT-based throughput benchmark.

Standalone script (some duplication with bench_concurrent.py's request-firing
logic, kept deliberately separate rather than sharing code -- this hits a real
dataset with real length variance, which is a different kind of test than the
fixed 6-prompt burst test).

Uses the same ShareGPT_V3_unfiltered_cleaned_split.json file vLLM's own
benchmark_throughput.py uses (https://github.com/vllm-project/vllm), so
numbers produced here are the kind that are directly comparable to published
vLLM/serving-engine benchmarks -- this is the closest thing the field has to
a standard corpus for this purpose.

What makes ShareGPT a better throughput stress test than the fixed prompt set:
  - Real human/ChatGPT conversation turns -> wide, realistic length variance
    (a few tokens to several thousand), which is exactly what stresses
    batching/padding/paging decisions. The fixed 6-prompt set never produces
    that spread.
  - It's what the field actually benchmarks against, so a "beat vLLM's naive
    baseline on ShareGPT" story is a recognizable, checkable claim.

Usage:
  python bench/bench_sharegpt.py --url http://localhost:8002 --num-prompts 32
  python bench/bench_sharegpt.py --url http://localhost:8002 --num-prompts 32 --out results/sharegpt32.json
"""

import argparse
import concurrent.futures
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path

import requests
from transformers import AutoTokenizer

DATASET_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/"
    "resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)
CACHE_PATH = Path(__file__).parent / "data" / "ShareGPT_V3_unfiltered_cleaned_split.json"


def ensure_dataset() -> Path:
    """Download once, cache locally -- this file is ~600MB+, don't refetch
    every run. Not committed to git (bench/data/ should be gitignored)."""
    if CACHE_PATH.exists():
        return CACHE_PATH
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] fetching ShareGPT dataset (one-time, large file)...")
    urllib.request.urlretrieve(DATASET_URL, CACHE_PATH)
    print(f"[download] saved to {CACHE_PATH}")
    return CACHE_PATH


def load_sampled_prompts(n: int, seed: int, tokenizer, min_tokens: int, max_tokens: int) -> list[dict]:
    """Load the raw dataset, extract first-turn human prompts, filter to a
    sane length range, sample n of them.

    Filtering matches the spirit of vLLM's own benchmark_throughput.py:
    skip conversations with no turns, and skip prompts that are too short to
    be a meaningful generation workload (e.g. "hi") or absurdly long (skews
    a single request's latency so much it dominates burst timing).
    """
    path = ensure_dataset()
    with open(path) as f:
        raw = json.load(f)

    candidates = []
    for conv in raw:
        turns = conv.get("conversations", [])
        if len(turns) < 2:
            continue
        first = turns[0]
        if first.get("from") != "human":
            continue
        prompt = first["value"].strip()
        if not prompt:
            continue
        n_tokens = len(tokenizer.encode(prompt))
        if min_tokens <= n_tokens <= max_tokens:
            candidates.append({"prompt": prompt, "prompt_tokens": n_tokens})

    if len(candidates) < n:
        raise RuntimeError(
            f"only found {len(candidates)} candidate prompts in "
            f"[{min_tokens}, {max_tokens}] tokens, need {n}"
        )

    rng = random.Random(seed)
    sampled = rng.sample(candidates, n)
    # Fixed max_tokens for the *response* -- this benchmark is about testing
    # against varied PROMPT (input) lengths, matching real ShareGPT input
    # variance; output length is capped uniformly so runs stay comparable.
    for item in sampled:
        item["max_tokens"] = 128
    return sampled


def run_one(url: str, prompt: str, max_tokens: int) -> dict:
    t_start = time.perf_counter()
    resp = requests.post(
        f"{url}/generate",
        json={"prompt": prompt, "max_tokens": max_tokens},
        timeout=180,
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
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-tokens", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="used only client-side, to filter/measure prompt lengths -- should match MODEL_ID on the server",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prompts = load_sampled_prompts(
        args.num_prompts, args.seed, tokenizer, args.min_tokens, args.max_tokens
    )
    lengths = [p["prompt_tokens"] for p in prompts]
    print(
        f"[sampled] {len(prompts)} prompts, "
        f"prompt_tokens min={min(lengths)} max={max(lengths)} mean={statistics.mean(lengths):.0f}"
    )

    health = requests.get(f"{args.url}/health", timeout=10).json()
    print(f"[health] model={health['model']} dtype={health['dtype']} mode={health.get('mode')}")
    print(f"[burst] firing {len(prompts)} concurrent requests...")

    results, burst_time = run_burst(args.url, prompts)

    latencies = [r["latency_s"] for r in results]
    batch_sizes = [r["batch_size"] for r in results]
    total_output_tokens = sum(len(tokenizer.encode(r["text"])) for r in results)

    summary = {
        "num_prompts": len(prompts),
        "prompt_tokens": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": statistics.mean(lengths),
        },
        "burst_wall_time_s": burst_time,
        "latency_s": {
            "mean": statistics.mean(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
        "batch_sizes_seen": sorted(set(batch_sizes)),
        "total_output_tokens": total_output_tokens,
        "throughput_tok_s": total_output_tokens / burst_time if burst_time > 0 else 0.0,
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
