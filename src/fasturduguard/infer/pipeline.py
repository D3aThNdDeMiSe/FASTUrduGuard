"""Inference pipeline + benchmarking harness.

Demonstrates:
  * dynamic length-bucketed batching (token-budget instead of fixed batch size)
  * CPU<->GPU pipelining via two CUDA streams (best-effort)
  * concurrent ensemble across heterogeneous models via ThreadPoolExecutor

Run the benchmarking harness with `python scripts/benchmark.py`.
"""
from __future__ import annotations

import dataclasses
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

log = logging.getLogger("fug.infer")


# ---------------------------------------------------------------------------
# Dynamic batching
# ---------------------------------------------------------------------------
def length_bucket_batches(texts: Sequence[str], lengths: Sequence[int],
                          token_budget: int = 8192) -> Iterable[list[int]]:
    """Yield batches (lists of indices) sorted by length, packing each batch up to
    `token_budget` total tokens (i.e. max_len_in_batch * batch_size <= token_budget)."""
    order = sorted(range(len(texts)), key=lambda i: lengths[i])
    batch: list[int] = []
    cur_max_len = 0
    for i in order:
        new_max = max(cur_max_len, lengths[i])
        if (len(batch) + 1) * new_max > token_budget and batch:
            yield batch
            batch, cur_max_len = [], 0
        batch.append(i)
        cur_max_len = max(cur_max_len, lengths[i])
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Single-model inference (with CUDA stream overlap when available)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ModelHandle:
    name: str
    tok: object
    mdl: object
    device: str          # "cuda" | "cpu"
    max_seq_len: int = 256


def load_handle(ckpt_dir: Path, max_seq_len: int = 256, device: str | None = None) -> ModelHandle:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    is_lora = (ckpt_dir / "adapter_config.json").exists()
    if is_lora:
        try:
            from peft import AutoPeftModelForSequenceClassification
            mdl = AutoPeftModelForSequenceClassification.from_pretrained(str(ckpt_dir))
        except Exception:
            mdl = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir))
    else:
        mdl = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir))
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True)
    if tok.pad_token is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(dev).eval()
    return ModelHandle(name=ckpt_dir.name, tok=tok, mdl=mdl, device=dev, max_seq_len=max_seq_len)


def predict(handle: ModelHandle, texts: Sequence[str], token_budget: int = 8192,
            use_streams: bool = True) -> np.ndarray:
    import torch
    tok = handle.tok
    mdl = handle.mdl
    dev = torch.device(handle.device)

    # tokenize lengths cheaply (no padding)
    enc = tok(list(texts), truncation=True, max_length=handle.max_seq_len, padding=False)
    lengths = [len(x) for x in enc["input_ids"]]

    out = np.zeros((len(texts), mdl.config.num_labels), dtype=np.float32)

    streams = []
    if use_streams and handle.device == "cuda":
        try:
            streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        except Exception:
            streams = []

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    batches = list(length_bucket_batches(texts, lengths, token_budget))
    next_inputs = None
    next_indices: list[int] | None = None

    def _prepare(idx_list: list[int]):
        ids = [enc["input_ids"][i] for i in idx_list]
        max_len = max(len(x) for x in ids)
        ids_pad = [x + [pad_id] * (max_len - len(x)) for x in ids]
        am_pad = [[1] * len(x) + [0] * (max_len - len(x)) for x in ids]
        return {
            "input_ids": torch.tensor(ids_pad, dtype=torch.long),
            "attention_mask": torch.tensor(am_pad, dtype=torch.long),
        }

    with torch.no_grad():
        for bi, idx_list in enumerate(batches):
            if streams:
                s = streams[bi % 2]
                with torch.cuda.stream(s):
                    inputs = {k: v.to(dev, non_blocking=True) for k, v in _prepare(idx_list).items()}
                    logits = mdl(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
            else:
                inputs = {k: v.to(dev) for k, v in _prepare(idx_list).items()}
                logits = mdl(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for j, original_idx in enumerate(idx_list):
                out[original_idx] = probs[j]
        if streams:
            torch.cuda.synchronize()
    return out


# ---------------------------------------------------------------------------
# Concurrent ensemble
# ---------------------------------------------------------------------------
def ensemble_predict(handles: Sequence[ModelHandle], weights: Sequence[float],
                     texts: Sequence[str]) -> tuple[np.ndarray, dict[str, float]]:
    """Concurrent multi-model prediction. Returns (final_argmax, per_model_latency_s)."""
    latencies: dict[str, float] = {}

    def _run(h):
        t0 = time.perf_counter()
        p = predict(h, texts)
        latencies[h.name] = time.perf_counter() - t0
        return p

    with ThreadPoolExecutor(max_workers=len(handles)) as ex:
        futs = {ex.submit(_run, h): h.name for h in handles}
        probs = [None] * len(handles)
        order = {h.name: i for i, h in enumerate(handles)}
        for fut in as_completed(futs):
            name = futs[fut]
            probs[order[name]] = fut.result()

    stack = np.stack(probs, axis=0)              # (M, N, C)
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 1, 1)
    blended = (w * stack).sum(axis=0) / max(np.sum(weights), 1e-9)
    return np.argmax(blended, axis=1), latencies


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def bench_throughput(handle: ModelHandle, texts: Sequence[str],
                     token_budgets: Sequence[int] = (1024, 2048, 4096, 8192, 16384)) -> list[dict]:
    out = []
    for budget in token_budgets:
        t0 = time.perf_counter()
        _ = predict(handle, texts, token_budget=budget, use_streams=True)
        dt = time.perf_counter() - t0
        out.append({
            "model": handle.name,
            "token_budget": int(budget),
            "throughput_per_s": len(texts) / dt,
            "latency_total_s": dt,
        })
    return out
