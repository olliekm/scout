"""
Generate HumanEval completions from a running server_static_batch.py instance,
in the samples.jsonl format openai/human-eval's evaluate_functional_correctness
expects: one JSON object per line, {"task_id": ..., "completion": ...}.

Setup (on the pod, one-time):
  git clone https://github.com/openai/human-eval
  pip install -e human-eval
  # human-eval deliberately disables code execution by default (it runs
  # untrusted model-generated code) -- uncomment the execution line in
  # human_eval/execution.py before running evaluate_functional_correctness.
  # Only do this in a disposable environment (e.g. this pod), never on a
  # machine with anything sensitive on it.

Usage:
  python bench/humaneval_gen.py --url http://localhost:8002 --out bench/results/humaneval_fp16_samples.jsonl
  # then, on the pod, from wherever human-eval was cloned:
  evaluate_functional_correctness bench/results/humaneval_fp16_samples.jsonl

Prompting note: HumanEval prompts are raw Python function signatures + a
docstring (NOT a chat message) -- the model is expected to complete the
function body directly. Sending it through the normal chat template (as
server_static_batch.py's /generate does by default for a `prompt` field)
would make the model respond conversationally instead of completing code,
which breaks the eval's expectation that `completion` is directly
concatenable after the HumanEval prompt. So this script talks to a raw
completion path, not the chat-templated one -- see the --raw flag wiring
into the server, which must NOT apply apply_chat_template for this to work.
"""

import argparse
import json
import re
from pathlib import Path

import requests
from human_eval.data import read_problems


def extract_completion(raw_text: str, prompt: str) -> str:
    """Model output for a code-completion prompt often wraps the answer in a
    markdown code fence and may repeat the prompt or add explanation before/
    after the code. Pull out just the function body continuation.

    Strategy: prefer a fenced ```python ... ``` block if present (common for
    instruct-tuned models even when asked to complete code directly); fall
    back to using the raw text as-is if no fence is found.
    """
    fence_match = re.search(r"```(?:python)?\s*\n(.*?)```", raw_text, re.DOTALL)
    code = fence_match.group(1) if fence_match else raw_text

    # If the model echoed the prompt back (common), strip it so what's left
    # is just the continuation to append after HumanEval's own prompt.
    if code.startswith(prompt):
        code = code[len(prompt):]

    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8002")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, default=None, help="only run first N problems, for a quick smoke test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    problems = read_problems()
    task_ids = list(problems.keys())
    if args.limit:
        task_ids = task_ids[: args.limit]

    print(f"[humaneval] generating completions for {len(task_ids)} problems...")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for i, task_id in enumerate(task_ids):
            prompt = problems[task_id]["prompt"]

            # raw=true: server must skip apply_chat_template and feed the
            # HumanEval prompt to the model as-is (a direct code completion,
            # not a chat turn). See server_static_batch.py's /generate --
            # this requires the `raw` field to be wired in there.
            resp = requests.post(
                f"{args.url}/generate",
                json={"prompt": prompt, "max_tokens": args.max_tokens, "raw": True},
                timeout=180,
            )
            resp.raise_for_status()
            raw_text = resp.json()["text"]

            completion = extract_completion(raw_text, prompt)
            f.write(json.dumps({"task_id": task_id, "completion": completion}) + "\n")

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(task_ids)}]")

    print(f"[humaneval] wrote {len(task_ids)} completions to {out_path}")


if __name__ == "__main__":
    main()
