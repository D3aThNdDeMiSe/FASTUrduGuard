"""Unify the three Urdu corpora into one parquet, dedup, train/val/test split.

Run via:
    python -m fasturduguard.data.unify --raw <path-to-datasets_raw>
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fasturduguard.utils import data_dir, load_yaml, set_seed, setup_logging
from fasturduguard.data import loaders


log = logging.getLogger("fug.data.unify")


def _hash_text(t: str) -> str:
    return hashlib.md5(t.encode("utf-8", errors="replace")).hexdigest()


def build_unified(raw_root: Path) -> pd.DataFrame:
    cfg = load_yaml("datasets.yaml")["roots"]

    lld_dir = raw_root / cfg["urdu_lld"]["folder"]
    btt_dir = raw_root / cfg["bend_the_truth"]["folder"]
    hb_dir = raw_root / cfg["hook_and_bait"]["folder"]

    frames: list[pd.DataFrame] = []

    if lld_dir.exists():
        files = cfg["urdu_lld"]["files"]
        if (lld_dir / files["ax_to_grind"]).exists():
            frames.append(loaders.load_ax_to_grind(lld_dir / files["ax_to_grind"]))
        if (lld_dir / files["labeled_news"]).exists():
            frames.append(loaders.load_labeled_news(lld_dir / files["labeled_news"]))
        if (lld_dir / files["notri"]).exists():
            frames.append(loaders.load_notri(lld_dir / files["notri"]))
    else:
        log.warning("Urdu-LLD folder missing: %s", lld_dir)

    if btt_dir.exists():
        train_root = btt_dir / cfg["bend_the_truth"]["train_root"]
        test_root = btt_dir / cfg["bend_the_truth"]["test_root"]
        frames.append(loaders.load_bend_the_truth(train_root, test_root))
    else:
        log.warning("Bend the Truth folder missing: %s", btt_dir)

    if hb_dir.exists():
        frames.append(loaders.load_hook_and_bait(hb_dir, cfg["hook_and_bait"]["files"]))
    else:
        log.warning("Hook & Bait folder missing: %s", hb_dir)

    if not frames:
        raise SystemExit("No data sources found under " + str(raw_root))

    df = pd.concat(frames, ignore_index=True)

    # Dedup by hashed text (Hook&Bait re-shipped Ax-to-Grind starting at row 31191).
    before = len(df)
    df["_h"] = df["text"].map(_hash_text)
    df = df.drop_duplicates(subset="_h", keep="first").drop(columns=["_h"]).reset_index(drop=True)
    log.info("Dedup: %d -> %d rows (%.1f%% removed)", before, len(df), 100*(1-len(df)/before))

    # Cleanup
    df["text"] = df["text"].astype(str).str.replace("\r", " ", regex=False).str.strip()
    df = df[df["text"].str.len() >= 20].reset_index(drop=True)

    return df


def add_split_column(df: pd.DataFrame, test_frac: float, val_frac: float, seed: int) -> pd.DataFrame:
    """Stratified train/val/test split written into a 'split' column."""
    idx_train, idx_test = train_test_split(
        df.index.values,
        test_size=test_frac,
        stratify=df["label_2"],
        random_state=seed,
    )
    val_relative = val_frac / (1 - test_frac)
    sub = df.loc[idx_train]
    idx_train2, idx_val = train_test_split(
        sub.index.values,
        test_size=val_relative,
        stratify=sub["label_2"],
        random_state=seed,
    )
    df = df.copy()
    df["split"] = "train"
    df.loc[idx_val, "split"] = "val"
    df.loc[idx_test, "split"] = "test"
    return df


def main() -> None:
    setup_logging(prefix="unify")
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True,
                   help="Path to the directory containing the three extracted source folders")
    p.add_argument("--out", type=Path, default=data_dir() / "processed" / "unified.parquet")
    p.add_argument("--summary", type=Path, default=data_dir() / "processed" / "unified_summary.json")
    args = p.parse_args()

    cfg = load_yaml("datasets.yaml")["split"]
    set_seed(cfg["seed"])

    df = build_unified(args.raw)

    # initial label_4 := label_2 (Real or Fake). relabel.py later may set 2 (Partial) or 3 (Satire).
    df["label_4"] = df["label_2"]

    df = add_split_column(df, cfg["test_frac"], cfg["val_frac"], cfg["seed"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    summary = {
        "rows": int(len(df)),
        "by_source": df["source"].value_counts().to_dict(),
        "by_split": df["split"].value_counts().to_dict(),
        "by_label_2": {str(k): int(v) for k, v in df["label_2"].value_counts().to_dict().items()},
        "domains": df["domain"].value_counts().head(15).to_dict(),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    import json
    with args.summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("Wrote %s", args.out)
    log.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
