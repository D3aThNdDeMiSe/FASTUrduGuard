"""Inference benchmark: throughput, latency, sequential vs concurrent ensemble.

Saves results into results/bench/.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fasturduguard.infer.pipeline import (
    bench_throughput, ensemble_predict, load_handle, predict,
)
from fasturduguard.utils import data_dir, results_dir, setup_logging


log = logging.getLogger("fug.bench")


def find_checkpoints(results_root: Path) -> list[Path]:
    out = []
    for rd in sorted(results_root.glob("rank_*")):
        for ck in sorted((rd / "checkpoints").glob("*")):
            if (ck / "adapter_config.json").exists() or (ck / "config.json").exists():
                out.append(ck)
    return out


def main() -> None:
    setup_logging(prefix="bench")
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200,
                   help="Number of test articles to benchmark over.")
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--results_root", type=Path, default=results_dir())
    p.add_argument("--unified", type=Path, default=data_dir() / "processed" / "unified.parquet")
    args = p.parse_args()

    out_dir = args.results_root / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.unified)
    test = df[df["split"] == "test"].sample(min(args.n, len(df)), random_state=42)
    texts = test["text"].astype(str).tolist()
    log.info("Benchmarking on %d articles", len(texts))

    cks = find_checkpoints(args.results_root)
    if not cks:
        raise SystemExit(f"No checkpoints under {args.results_root}/rank_*/checkpoints")

    # ------------------------------------------------------------------
    # Throughput per model at varying token-budgets
    # ------------------------------------------------------------------
    rows = []
    handles = []
    for ck in cks:
        try:
            h = load_handle(ck, max_seq_len=args.max_seq_len)
            handles.append(h)
            for r in bench_throughput(h, texts):
                r["checkpoint"] = str(ck)
                rows.append(r)
        except Exception as e:
            log.warning("Bench skip %s: %s", ck, e)
    pd.DataFrame(rows).to_csv(out_dir / "throughput.csv", index=False)

    # ------------------------------------------------------------------
    # Sequential vs concurrent ensemble
    # ------------------------------------------------------------------
    if len(handles) >= 2:
        # Sequential
        t0 = time.perf_counter()
        seq_lat = {}
        for h in handles:
            tt = time.perf_counter()
            _ = predict(h, texts, token_budget=4096, use_streams=True)
            seq_lat[h.name] = time.perf_counter() - tt
        seq_total = time.perf_counter() - t0

        # Concurrent
        weights = [1.0] * len(handles)
        t0 = time.perf_counter()
        _, conc_lat = ensemble_predict(handles, weights, texts)
        conc_total = time.perf_counter() - t0

        ens = {
            "n_models": len(handles),
            "n_articles": len(texts),
            "sequential_total_s": seq_total,
            "sequential_per_model_s": seq_lat,
            "concurrent_total_s": conc_total,
            "concurrent_per_model_s": conc_lat,
            "speedup": seq_total / conc_total if conc_total > 0 else None,
        }
        (out_dir / "ensemble_compare.json").write_text(
            json.dumps(ens, indent=2), encoding="utf-8"
        )
        log.info("Sequential ensemble: %.2fs   Concurrent: %.2fs   speedup: %.2fx",
                 seq_total, conc_total, ens["speedup"] or 0)


if __name__ == "__main__":
    main()
