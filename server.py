"""
Step 1 baseline inference server.

Goal: the SIMPLEST possible serving layer over HuggingFace generate(), so it
can act as the baseline that every later step (static batching, paged KV-cache,
continuous batching, custom CUDA kernel) is measured against.

There is deliberately NO batching here. One request = one generate() call.
That is the point: this is the "naive" number you will beat.

Two endpoints:
  POST /generate         -> non-streaming. Returns text + server-side timing.
  POST /generate_stream  -> streams one JSON line per token, so the CLIENT can
                            measure time-to-first-token (TTFT) and per-token
                            latency (TPOT). These are the metrics that matter.

Run on the pod (inside tmux):
  pip install fastapi uvicorn transformers accelerate torch
  MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct uvicorn server:app --host 0.0.0.0 --port 8000

Verify one request works before pointing the load tester at it:
  curl -X POST localhost:8000/generate -H 'Content-Type: application/json' \
       -d '{"prompt": "Write a haiku about GPUs.", "max_tokens": 64}'
"""

import os
import json
import time
from threading import Thread

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# --- Config (override via env so you don't edit code to swap models) ----------
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
# fp16 is the realistic serving dtype. Make it explicit and configurable so you
# can run the FP16-vs-FP32 comparison you've done before if you want it.
DTYPE = torch.float16 if os.environ.get("DTYPE", "fp16") == "fp16" else torch.float32
DEVICE = "cuda"

app = FastAPI()

# Loaded once at startup, not per request. Loading per request would dominate
# every measurement and is the classic baseline mistake.
tokenizer = None
model = None


class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = 128


def _build_inputs(prompt: str):
    # Qwen instruct models expect the chat template. Using it keeps outputs sane
    # and keeps prompt token counts realistic for your measurements.
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt").to(DEVICE)


@app.on_event("startup")
def load_model():
    global tokenizer, model
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    model.eval()

    # Warmup: the first few generate() calls pay one-time costs (CUDA kernel
    # autotuning, allocator warmup). Measuring those inflates your baseline and
    # silently makes later "improvements" look bigger than they are. Burn a few.
    warm = _build_inputs("warmup")
    with torch.no_grad():
        for _ in range(3):
            model.generate(**warm, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    print(f"[startup] {MODEL_ID} loaded + warmed in {time.perf_counter() - t0:.1f}s")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "dtype": str(DTYPE)}


@app.post("/generate")
def generate(req: GenRequest):
    """Non-streaming. Good for a quick sanity curl and for throughput runs where
    you only care about total tokens / total time."""
    inputs = _build_inputs(req.prompt)
    prompt_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=req.max_tokens, do_sample=False
        )
    torch.cuda.synchronize()
    server_time = time.perf_counter() - t0

    output_ids = out[0][prompt_len:]
    output_tokens = output_ids.shape[0]
    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    # Return server-side truth so the harness can separate it from client-side
    # latency (which includes queueing + network).
    return {
        "text": text,
        "prompt_tokens": prompt_len,
        "output_tokens": output_tokens,
        "server_time_s": server_time,
    }


@app.post("/generate_stream")
def generate_stream(req: GenRequest):
    """Streams one JSON line per token. The client timestamps each line:
       - TTFT  = time until the first line arrives
       - TPOT  = mean gap between subsequent lines
       - output_tokens = number of lines
    This is how you get the latency metrics that actually get probed."""
    inputs = _build_inputs(req.prompt)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs, streamer=streamer, max_new_tokens=req.max_tokens, do_sample=False
    )
    # generate() blocks, so run it in a thread and read tokens off the streamer.
    thread = Thread(target=model.generate, kwargs=gen_kwargs)

    def event_stream():
        thread.start()
        for piece in streamer:
            if piece:  # streamer can yield empty strings; skip them
                yield json.dumps({"token": piece}) + "\n"
        thread.join()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")