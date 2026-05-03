"""Token-level attention attribution from the final transformer layer."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("fug.explain.attention")


def token_attribution(ckpt_dir: Path, text: str, max_seq_len: int = 256) -> dict:
    """Return per-token attention weights from CLS-vs-tokens (head-mean, last layer)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    is_lora = (ckpt_dir / "adapter_config.json").exists()
    if is_lora:
        try:
            from peft import AutoPeftModelForSequenceClassification
            mdl = AutoPeftModelForSequenceClassification.from_pretrained(str(ckpt_dir), output_attentions=True)
        except Exception:
            mdl = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir), output_attentions=True)
    else:
        mdl = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir), output_attentions=True)
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True)
    if tok.pad_token is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(dev).eval()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl(**enc, output_attentions=True)
    attns = out.attentions       # tuple of (num_layers,) of (1, H, T, T)
    if not attns:
        return {"tokens": tok.convert_ids_to_tokens(enc["input_ids"][0]),
                "scores": [0.0] * enc["input_ids"].shape[1],
                "predicted": int(np.argmax(out.logits.cpu().numpy(), axis=-1))}
    last = attns[-1]                       # (1, H, T, T)
    head_mean = last.mean(dim=1)[0]        # (T, T)
    cls_attn = head_mean[0].cpu().numpy()  # attention from [CLS]
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    pred = int(np.argmax(out.logits.cpu().numpy(), axis=-1))
    return {"tokens": tokens, "scores": cls_attn.tolist(), "predicted": pred,
            "logits": out.logits[0].cpu().numpy().tolist()}
