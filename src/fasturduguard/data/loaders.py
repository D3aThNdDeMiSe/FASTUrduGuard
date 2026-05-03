"""Per-source loaders.

Each loader returns a DataFrame with a uniform schema:
    text, label_2, source, domain, extra_id

label_2:
    0 = Real / True / LEGIT
    1 = Fake / False / FAKE
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger("fug.data")


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------
_REAL_TOKENS = {"real", "true", "legit", "legitimate", "genuine"}
_FAKE_TOKENS = {"fake", "false", "fabricated", "unreal", "rumour", "rumor"}


def normalize_label(raw) -> int | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in _REAL_TOKENS:
        return 0
    if s in _FAKE_TOKENS:
        return 1
    return None


def _drop_bad(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    df = df.copy()
    df[label_col] = df[label_col].map(normalize_label)
    df = df.dropna(subset=[text_col, label_col])
    df = df[df[text_col].astype(str).str.strip().str.len() >= 20]
    df[label_col] = df[label_col].astype(int)
    return df


# ---------------------------------------------------------------------------
# Urdu-LLD
# ---------------------------------------------------------------------------
def load_ax_to_grind(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"News_Items": "text", "Label": "label"})
    df = _drop_bad(df, "text", "label")
    out = pd.DataFrame({
        "text": df["text"].astype(str),
        "label_2": df["label"].astype(int),
        "source": "urdu_lld_ax_to_grind",
        "domain": "general",
        "extra_id": df["Sr_No"].astype(str) if "Sr_No" in df else "",
    })
    log.info("Ax-to-Grind: %d rows", len(out))
    return out


def load_labeled_news(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path).rename(columns={"news": "text", "label": "label"})
    df = _drop_bad(df, "text", "label")
    out = pd.DataFrame({
        "text": df["text"].astype(str),
        "label_2": df["label"].astype(int),
        "source": "urdu_lld_labeled_news",
        "domain": "general",
        "extra_id": "",
    })
    log.info("Labeled_Urdu_News: %d rows", len(out))
    return out


def load_notri(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(columns={"News_Text": "text", "Label": "label", "Category": "domain"})
    df = _drop_bad(df, "text", "label")
    out = pd.DataFrame({
        "text": df["text"].astype(str),
        "label_2": df["label"].astype(int),
        "source": "urdu_lld_notri",
        "domain": df["domain"].fillna("general").astype(str).str.lower(),
        "extra_id": df.get("Index", pd.Series([""] * len(df))).astype(str),
    })
    log.info("Notri: %d rows", len(out))
    return out


# ---------------------------------------------------------------------------
# Bend the Truth (filesystem-based)
# ---------------------------------------------------------------------------
_DOMAIN_PREFIXES = {
    "sp": "sports",
    "sbz": "business",
    "tch": "technology",
    "hl": "health",
    "shw": "showbiz",
    "ent": "entertainment",
    "bus": "business",
}


def _domain_from_filename(stem: str) -> str:
    s = stem.lower()
    # Files look like "sp1.txt", "sbz23.txt", "tch4.txt"
    for k, v in _DOMAIN_PREFIXES.items():
        if s.startswith(k):
            return v
    return "general"


def load_bend_the_truth(train_root: Path, test_root: Path) -> pd.DataFrame:
    rows = []
    for split, root in [("train", train_root), ("test", test_root)]:
        for cls_name, label in [("Fake", 1), ("Real", 0)]:
            d = root / cls_name
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.txt")):
                try:
                    text = f.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if len(text.strip()) < 20:
                    continue
                rows.append({
                    "text": text.strip(),
                    "label_2": label,
                    "source": "bend_the_truth",
                    "domain": _domain_from_filename(f.stem),
                    "extra_id": f"{split}/{f.stem}",
                })
    out = pd.DataFrame(rows)
    log.info("Bend the Truth: %d rows", len(out))
    return out


# ---------------------------------------------------------------------------
# Hook & Bait (multiple xlsx files)
# ---------------------------------------------------------------------------
def load_hook_and_bait(folder: Path, files: Iterable[str]) -> pd.DataFrame:
    frames = []
    for fname in files:
        p = folder / fname
        if not p.exists():
            log.warning("H&B file missing: %s", p)
            continue
        df = pd.read_excel(p)
        # column names vary slightly across files
        df.columns = [str(c).strip() for c in df.columns]
        text_col = next((c for c in df.columns if c.lower() == "news items"), None)
        label_col = next((c for c in df.columns if c.lower() == "label"), None)
        if text_col is None or label_col is None:
            log.warning("Skipping %s: cannot find text/label cols among %s",
                        fname, list(df.columns))
            continue
        df = df.rename(columns={text_col: "text", label_col: "label"})
        df = _drop_bad(df, "text", "label")
        frames.append(pd.DataFrame({
            "text": df["text"].astype(str),
            "label_2": df["label"].astype(int),
            "source": "hook_and_bait",
            "domain": "general",
            "extra_id": fname,
        }))
    if not frames:
        return pd.DataFrame(columns=["text", "label_2", "source", "domain", "extra_id"])
    out = pd.concat(frames, ignore_index=True)
    log.info("Hook & Bait: %d rows", len(out))
    return out
