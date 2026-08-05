"""
Step 2 prototype: static batching.

THROWAWAY Python prototype (per AGENT.md, the real engine is Rust). Purpose:
validate the batching/padding logic in a language you're fast in, before
fighting Rust's ownership model on the same concepts. Port the *design*,
not this code, to Rust once it works here.

Same two endpoints as server.py's baseline, but /generate no longer calls
model.generate() directly. Instead it enqueues onto a shared batcher
background task, which:
  - collects up to MAX_BATCH_SIZE requests, or waits at most MAX_WAIT_S,
    whichever comes first (see bench/scratch/asyncio_batching_toy.py for
    the isolated version of this pattern)
  - left-pads all prompts in the batch to the same length
  - runs ONE model.generate() call for the whole batch
  - slices the output back apart and returns each request only its own text

Run on the pod:
  MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct uvicorn server_static_batch:app --host 0.0.0.0 --port 8001

(Different port from server.py so you can run both side by side and compare.)
"""

import asyncio
import os
import time

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DTYPE = torch.float16 if os.environ.get("DTYPE", "fp16") == "fp16" else torch.float32
DEVICE = "cuda"

# "" (default) = full precision (DTYPE above). "int8" = bitsandbytes LLM.int8()
# quantization -- linear layer weights stored as int8, activation-outlier
# columns kept in higher precision internally (bitsandbytes' own judgment
# call about which computations are precision-sensitive, not something we
# hand-tune here). See AGENT.md's quantization stretch milestone.
QUANTIZE = os.environ.get("QUANTIZE", "")

MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", 4))
MAX_WAIT_S = float(os.environ.get("MAX_WAIT_S", 0.1))

app = FastAPI()

tokenizer = None
model = None
queue: "asyncio.Queue[BatchItem]" = None  # created at startup, needs a running loop


class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    raw: bool = False


class BatchItem:
    """One request waiting to be batched. Analogue of the
    (request_id, work, future) tuple in the toy example, but as a class
    since there's more than one field of "work" here (prompt text AND a
    per-request max_tokens -- see the note on that below)."""

    __slots__ = ("prompt", "max_tokens", "raw", "future", "t_submit")

    def __init__(self, prompt: str, max_tokens: int, raw: bool, future: "asyncio.Future"):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.raw = raw
        self.future = future
        self.t_submit = time.perf_counter()


def build_chat_text(prompt: str) -> str:
    """Same chat-template wrapping as server.py's _build_inputs, split out
    because in the batched path you need the raw templated STRING for each
    request before tokenizing them all together as a batch (you can't easily
    batch-tokenize if you tokenize one at a time first)."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


async def collect_batch(already_have: int) -> list[BatchItem]:
    """YOUR CODE HERE.

    Pull items off `queue`, up to MAX_BATCH_SIZE of them, but don't wait
    longer than MAX_WAIT_S total for the batch to fill. Return whatever you
    collected (could be 1 item, could be MAX_BATCH_SIZE, never 0 -- only
    call this when you know at least one item is coming, or make the caller
    handle the empty case).

    This is the same pattern as batcher()'s inner while-loop in
    bench/scratch/asyncio_batching_toy.py -- the only new wrinkle is you're
    returning the list instead of immediately processing it, so keep the
    deadline/asyncio.wait_for structure but adapt it to return-a-list shape.
    """
    batch = []
    deadline = time.perf_counter() + MAX_WAIT_S

    while len(batch) + already_have < MAX_BATCH_SIZE:
        timeout = deadline - time.perf_counter()
        if timeout <= 0:
            break
        try:
            item = await asyncio.wait_for(queue.get(), timeout=timeout)
            batch.append(item)
            deadline = time.perf_counter() + MAX_WAIT_S
        except asyncio.TimeoutError:
            break

    return batch


def build_batch_tensors(items: list[BatchItem]):
    """YOUR CODE HERE.

    Given a list of BatchItems, return tokenizer inputs suitable for a
    single batched model.generate() call.

    Things to get right:
      - Build the chat-templated text for each item (use build_chat_text).
      - Tokenize them together as a batch, NOT one at a time -- the
        tokenizer supports passing a list of strings directly, and can pad
        them for you. Look at the `padding=True` argument.
      - Qwen/most causal LMs need LEFT padding for generation (not the
        default right padding) -- when prompts are right-padded, the model's
        next-token prediction reads from the last position in the sequence,
        which would be a pad token, not real content. Check
        tokenizer.padding_side and set it before tokenizing.
      - Return whatever model.generate() needs: input_ids, attention_mask,
        moved to DEVICE. Also return the per-sequence prompt lengths (you'll
        need them to slice each output apart later -- but careful: with
        left-padding, all sequences end at the same position in the tensor,
        so think about whether you need original per-item prompt length, or
        the padded batch prompt length, to correctly slice out only the
        NEW tokens for each item).
      - max_new_tokens for generate() has to be ONE number for the whole
        batch (static batching can't give different requests different
        lengths mid-call) -- decide how you derive it from the batch's
        individual max_tokens values, and note this as a real limitation
        of static batching worth remembering for later stages.
    """

    batch_chat_templated = [item.prompt if item.raw else build_chat_text(item.prompt) for item in items]
    tokenizer.padding_side = "left"
    batched_tensors = tokenizer(batch_chat_templated, return_tensors="pt", padding=True).to(DEVICE)
    return batched_tensors, batched_tensors["input_ids"].shape[1]


def slice_outputs(items: list[BatchItem], generate_output, batch_prompt_len: int) -> list[str]:
    """YOUR CODE HERE.

    generate_output is the full [batch_size, seq_len] tensor model.generate()
    returned (prompt + generated tokens, for every row, padded to the same
    width). For each item, return just the newly generated text (decoded,
    skip_special_tokens=True) -- not the prompt, not padding, not other
    requests' tokens.

    Return a list of decoded strings, same order as `items`.
    """
    decoded_output = []
    for i, item in enumerate(items):
        new_tokens = generate_output[i, batch_prompt_len:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        decoded_output.append(decoded)
    return decoded_output


async def batcher_loop():
    """Runs forever in the background (started at app startup). Forms one
    batch via collect_batch(), runs it, resolves every item's future with
    its decoded text. Wraps generate in torch.no_grad() same as server.py.

    Left mostly filled in since the interesting logic is in the three
    functions above -- but read this to see how they fit together.
    """
    while True:
        first = await queue.get()
        items = [first] + await collect_batch(already_have=1)

        inputs, batch_prompt_len = build_batch_tensors(items)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max(i.max_tokens for i in items),
                do_sample=False,
            )
        torch.cuda.synchronize()
        batch_time = time.perf_counter() - t0

        texts = slice_outputs(items, out, batch_prompt_len)

        print(f"[batch] size={len(items)} time={batch_time:.2f}s")
        for item, text in zip(items, texts):
            item.future.set_result(
                {"text": text, "batch_size": len(items), "batch_time_s": batch_time}
            )


@app.on_event("startup")
async def load_model():
    global tokenizer, model, queue
    queue = asyncio.Queue()

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if QUANTIZE == "int8":
        # device_map="auto" places quantized weights on GPU as they load --
        # replaces the manual .to(DEVICE) call used in the full-precision
        # path below (bitsandbytes quantizes during load, not after).
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    model.eval()

    warm = tokenizer(build_chat_text("warmup"), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        for _ in range(3):
            model.generate(**warm, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    print(f"[startup] {MODEL_ID} loaded + warmed in {time.perf_counter() - t0:.1f}s")

    asyncio.create_task(batcher_loop())


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "dtype": str(DTYPE),
        "quantize": QUANTIZE or "none",
        "mode": "static_batch",
    }


@app.post("/generate")
async def generate(req: GenRequest):
    future = asyncio.get_event_loop().create_future()
    await queue.put(BatchItem(req.prompt, req.max_tokens, req.raw, future))
    result = await future
    return result
