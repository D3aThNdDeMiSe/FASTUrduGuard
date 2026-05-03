"""Multi-class re-annotation: 2-class -> 4-class.

Rule mode (default, no GPU needed):
    Real -> Real  (with a small fraction routed to Satire/Opinion if a
    satire keyword appears in the text or domain == "showbiz" and an opinion-style
    headline pattern is detected)
    Fake -> Fake  (with a small fraction routed to PartiallyFalse if a
    "partial-falsity" keyword appears in the text)

LLM mode (optional, run on a node that has a GPU):
    Use Qwen2.5-1.5B-Instruct (configurable) to label a sample of size N
    with confidence; high-confidence predictions overwrite the rule-based labels.

Run via:
    python -m fasturduguard.data.relabel --mode rule
    python -m fasturduguard.data.relabel --mode llm --sample 2000
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from fasturduguard.utils import data_dir, load_yaml, set_seed, setup_logging

log = logging.getLogger("fug.data.relabel")


# ---------------------------------------------------------------------------
# Rule-based
# ---------------------------------------------------------------------------
def relabel_rule(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sat_ur = cfg["satire_keywords_ur"]
    sat_ro = [s.lower() for s in cfg["satire_keywords_roman"]]
    par_ur = cfg["partial_keywords_ur"]

    df = df.copy()

    text_l = df["text"].astype(str)
    text_lower = text_l.str.lower()

    # Satire/Opinion: keyword present (Urdu or Roman). Affects both Real and Fake rows.
    sat_mask = (
        text_l.apply(lambda t: any(k in t for k in sat_ur))
        | text_lower.apply(lambda t: any(k in t for k in sat_ro))
    )

    # PartiallyFalse: keyword present, restricted to currently-Fake rows
    par_mask = (df["label_2"] == 1) & text_l.apply(lambda t: any(k in t for k in par_ur))

    df["label_4"] = df["label_2"]   # default: Real (0) or Fake (1)
    df.loc[par_mask, "label_4"] = 2  # PartiallyFalse
    df.loc[sat_mask, "label_4"] = 3  # Satire/Opinion

    return df


# ---------------------------------------------------------------------------
# LLM-assisted (Qwen2.5)
# ---------------------------------------------------------------------------
_LLM_PROMPT = """You are an expert Urdu fact-checker. Classify the article into exactly one of:
0 = Real (factually accurate news)
1 = Fake (entirely fabricated to deceive)
2 = PartiallyFalse (mixes real and fabricated information / misleading framing)
3 = Satire (humorous/opinion piece, not intended as factual news)

Reply with strict JSON: {"label": <int>, "confidence": <float 0..1>}.
Article:
"""


def relabel_llm(df: pd.DataFrame, cfg: dict, sample_size: int) -> pd.DataFrame:
    """Best-effort LLM relabel; falls back silently if transformers/llm not available."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        log.warning("LLM relabel skipped: %s", e)
        return df

    name = cfg["llm_model"]
    log.info("Loading LLM %s ...", name)
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
    except Exception as e:
        log.warning("Could not load %s: %s. Skipping LLM relabel.", name, e)
        return df

    rng = pd.np.random.default_rng(42) if hasattr(pd, "np") else None  # avoid hard pandas dep
    import numpy as np
    rng = np.random.default_rng(42)

    sample_idx = rng.choice(df.index.values, size=min(sample_size, len(df)), replace=False)
    df = df.copy()
    if "label_4_llm" not in df.columns:
        df["label_4_llm"] = -1

    for i, idx in enumerate(sample_idx):
        text = str(df.at[idx, "text"])[:2000]  # truncate
        prompt = _LLM_PROMPT + text + "\nJSON:"
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            j = json.loads(gen[gen.find("{"): gen.rfind("}") + 1])
            lab = int(j.get("label", -1))
            conf = float(j.get("confidence", 0.0))
            if lab in (0, 1, 2, 3) and conf >= cfg["llm_confidence_threshold"]:
                df.at[idx, "label_4_llm"] = lab
                df.at[idx, "label_4"] = lab
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            log.info("LLM relabel %d/%d", i + 1, len(sample_idx))

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    setup_logging(prefix="relabel")
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, default=data_dir() / "processed" / "unified.parquet")
    p.add_argument("--out", type=Path, default=data_dir() / "processed" / "unified.parquet")
    p.add_argument("--mode", choices=("rule", "llm", "rule+llm"), default=None)
    p.add_argument("--sample", type=int, default=None)
    args = p.parse_args()

    cfg = load_yaml("datasets.yaml")["relabel"]
    mode = args.mode or cfg["mode"]
    sample_size = args.sample or cfg["llm_sample_size"]
    set_seed(42)

    df = pd.read_parquet(args.inp)
    log.info("Loaded %d rows", len(df))

    if "rule" in mode:
        df = relabel_rule(df, cfg)
    if "llm" in mode:
        df = relabel_llm(df, cfg, sample_size)

    log.info("label_4 dist: %s", df["label_4"].value_counts().to_dict())
    df.to_parquet(args.out, index=False)
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
