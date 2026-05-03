"""Sentence-level LIME explanations.

We split the article into sentences using Urdu/English punctuation, treat each
sentence as a binary feature, and let LIME perturb (mask) sentences. The model
is queried via a callable that re-joins the surviving sentences and returns
softmax probabilities.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger("fug.explain.lime")


_SPLIT = re.compile(r"(?<=[\.\?\!\u06D4\u061F\n])\s+")


def split_sentences(text: str) -> list[str]:
    s = re.sub(r"\s+", " ", text).strip()
    parts = _SPLIT.split(s)
    return [p for p in parts if p.strip()]


def _make_predict_fn(ckpt_dir: Path, max_seq_len: int = 256, batch_size: int = 16) -> Callable:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if (ckpt_dir / "adapter_config.json").exists():
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
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(dev).eval()

    def fn(texts: list[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                enc = tok(batch, padding=True, truncation=True, max_length=max_seq_len,
                          return_tensors="pt").to(dev)
                logits = mdl(**enc).logits
                p = torch.softmax(logits, dim=-1).cpu().numpy()
                out.append(p)
        return np.concatenate(out, axis=0)

    return fn


def explain_sentences(ckpt_dir: Path, text: str, num_features: int = 10,
                      num_samples: int = 200) -> dict:
    sents = split_sentences(text)
    if len(sents) <= 1:
        # nothing to perturb
        pred_fn = _make_predict_fn(ckpt_dir)
        probs = pred_fn([text])[0]
        return {"sentences": sents, "scores": [1.0] * len(sents),
                "predicted": int(np.argmax(probs)), "probs": probs.tolist()}

    try:
        from lime.lime_text import LimeTextExplainer
    except Exception as e:
        log.warning("lime not installed (%s) -- returning uniform scores.", e)
        pred_fn = _make_predict_fn(ckpt_dir)
        probs = pred_fn([text])[0]
        return {"sentences": sents, "scores": [1.0 / len(sents)] * len(sents),
                "predicted": int(np.argmax(probs)), "probs": probs.tolist()}

    pred_fn = _make_predict_fn(ckpt_dir)
    full_probs = pred_fn([text])[0]
    target = int(np.argmax(full_probs))

    SEP = " ||| "

    def perturbed_predict(masked_strings: list[str]) -> np.ndarray:
        # masked_strings come back as the original tokens with some replaced by UNKWORDZ
        rebuilt: list[str] = []
        for ms in masked_strings:
            kept = [s for s in ms.split(SEP) if s.strip() and "UNKWORDZ" not in s]
            rebuilt.append(" ".join(kept) if kept else " ")
        return pred_fn(rebuilt)

    explainer = LimeTextExplainer(
        class_names=[f"c{i}" for i in range(full_probs.shape[0])],
        bow=False,
        split_expression=re.escape(SEP),
        random_state=42,
    )
    lime_input = SEP.join(sents)
    exp = explainer.explain_instance(
        lime_input,
        perturbed_predict,
        num_features=min(num_features, len(sents)),
        num_samples=num_samples,
        labels=[target],
    )
    scores = [0.0] * len(sents)
    for tok_str, weight in exp.as_list(label=target):
        # tok_str is one of the original sentences (LIME keeps the surface)
        try:
            idx = sents.index(tok_str)
            scores[idx] = float(weight)
        except ValueError:
            pass

    return {"sentences": sents, "scores": scores,
            "predicted": target, "probs": full_probs.tolist()}
