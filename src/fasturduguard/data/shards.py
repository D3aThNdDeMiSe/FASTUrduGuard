"""Per-rank data shards (used by FedAvg / data-parallel mode).

Output: data/processed/shards/rank_<R>.parquet for R in 0..N-1
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fasturduguard.utils import data_dir, load_yaml, set_seed, setup_logging

log = logging.getLogger("fug.data.shards")


def stratified_split(df: pd.DataFrame, num_ranks: int, label_col: str, seed: int) -> list[pd.DataFrame]:
    """Stratified random round-robin split by class label."""
    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[] for _ in range(num_ranks)]
    for _, sub in df.groupby(label_col):
        idx = sub.index.values.copy()
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            parts[i % num_ranks].append(int(ix))
    return [df.loc[sorted(p)].reset_index(drop=True) for p in parts]


def by_source_split(df: pd.DataFrame, num_ranks: int) -> list[pd.DataFrame]:
    """Round-robin distinct sources to ranks (cleaner federated story)."""
    sources = sorted(df["source"].unique())
    bucket = {s: i % num_ranks for i, s in enumerate(sources)}
    return [df[df["source"].map(bucket) == r].reset_index(drop=True) for r in range(num_ranks)]


def main() -> None:
    setup_logging(prefix="shards")
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, default=data_dir() / "processed" / "unified.parquet")
    p.add_argument("--out_dir", type=Path, default=data_dir() / "processed" / "shards")
    p.add_argument("--num_ranks", type=int, default=None)
    p.add_argument("--strategy", default=None,
                   choices=("stratified_random", "by_source"))
    args = p.parse_args()

    cfg = load_yaml("datasets.yaml")["shards"]
    num_ranks = args.num_ranks or cfg["num_ranks"]
    strategy = args.strategy or cfg["strategy"]
    set_seed(cfg["seed"])

    df = pd.read_parquet(args.inp)
    train = df[df["split"] == "train"].reset_index(drop=True)

    if strategy == "stratified_random":
        parts = stratified_split(train, num_ranks, "label_4", cfg["seed"])
    elif strategy == "by_source":
        parts = by_source_split(train, num_ranks)
    else:
        raise ValueError(strategy)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    for r, part in enumerate(parts):
        # Always carry val + test entries on every rank for local evaluation
        merged = pd.concat(
            [part.assign(split="train"), df[df["split"] != "train"]],
            ignore_index=True,
        )
        out = args.out_dir / f"rank_{r}.parquet"
        merged.to_parquet(out, index=False)
        summary.append({
            "rank": r,
            "rows": int(len(merged)),
            "train": int(len(part)),
            "by_label_4": {str(k): int(v) for k, v in part["label_4"].value_counts().to_dict().items()},
        })
        log.info("rank_%d -> %s  train=%d  by_label=%s",
                 r, out, len(part), summary[-1]["by_label_4"])

    import json
    (args.out_dir / "shards_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
