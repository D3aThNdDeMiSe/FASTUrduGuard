"""Measure intra-node *CPU parallelism* contention that the Phase-2 proposal describes.

Problem this answers
---------------------
The Phase-2 mid-report claims lightweight transformer models run on CPU *concurrently*
while a heavy GPU model trains. That is valid on a workstation with tens of cores
but blows up on a 16-thread laptop CPU because HF `Trainer`s + `prefork` Pools all
fight for cores.

Usage (PowerShell examples)
----------------------------
Quick 5 k-sample, 1-epoch micro-benchmark (≈ tens of minutes, not tens of hours)::

    python scripts/benchmark_intra_node.py --subset-rows 5000 --epochs 1

Full paper-grade timings (hours)::

    python scripts/benchmark_intra_node.py --subset-rows -1 --epochs 3 --batch-size 16

Outputs land in `results/bench_intranode/{summary.json,README.md}`
so you can drop the numbers straight into Phase-3 section  5.X "Intra-node
parallelism contention analysis".
"""

from __future__ import annotations

import argparse
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from fasturduguard.train.trainer import train_one
from fasturduguard.utils import coordinator_dir, data_dir, load_json, setup_logging


def _subset(df: pd.DataFrame, *, n_rows: int, seed: int):
    train_full = df[df["split"] == "train"].reset_index(drop=True)
    val_full = df[df["split"] == "val"].reset_index(drop=True)
    test_full = df[df["split"] == "test"].reset_index(drop=True)

    if n_rows < 0:
        train = train_full
    else:
        n = min(n_rows, len(train_full))
        train, _ = train_test_split(
            train_full,
            train_size=n,
            stratify=train_full["label_4"],
            random_state=seed,
        )
        train = train.reset_index(drop=True)

    max_val = min(2_000, len(val_full))
    max_test = min(2_000, len(test_full))
    val = val_full.sample(n=max_val, random_state=seed) if max_val else val_full
    test = test_full.sample(n=max_test, random_state=seed) if max_test else test_full

    print(
        f"Bench subset sizes  train={len(train)}  val={len(val)}  "
        f"test={len(test)}  (requested train cap={n_rows})"
    )
    return train, val, test


def _models_for_rank(rank: int) -> list[str]:
    mpath = coordinator_dir() / "manifest.json"
    if not mpath.exists():
        raise SystemExit(f"Missing coordinator manifest.json at {mpath}")
    manifest = load_json(mpath)
    return manifest["mode_a"]["assignments"][str(rank)]


def _train_wallclock(
    model_cfg: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    rank: int,
    out_suffix: Path,
    epochs: int,
    batch: int,
    max_seq_len: int,
    tr_yaml: dict[str, Any],
) -> dict[str, Any]:
    rd = Path("results/bench_intranode/tmp") / out_suffix.name / model_cfg["name"]
    rd.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    metrics = train_one(
        model_cfg=model_cfg,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        out_dir=rd,
        rank=rank,
        epochs=epochs,
        lr=tr_yaml["learning_rate"],
        batch_size=batch,
        grad_accum=tr_yaml["gradient_accumulation_steps"],
        max_seq_len=max_seq_len,
        weight_decay=tr_yaml["weight_decay"],
        warmup_ratio=tr_yaml["warmup_ratio"],
        fp16=tr_yaml.get("fp16", True),
        seed=tr_yaml["seed"],
    )
    return {
        **metrics,
        "wallclock_s": time.perf_counter() - t0,
    }


def main() -> None:
    logger = setup_logging(prefix="bench-intranode")
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=int(os.environ.get("FUG_RANK", "0")),
                   help="Which manifest bucket to probe (normally same as node's FUG_RANK).")
    p.add_argument("--subset-rows", type=int, default=5000,
                   help="Max train rows AFTER stratified subsampling; -1=all train.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from fasturduguard.utils import load_yaml

    models_yaml = load_yaml("models.yaml")
    train_cfg = models_yaml["training"]
    by_name = {m["name"]: m.copy() for m in models_yaml["models"]}

    assigned = _models_for_rank(args.rank)

    uni = pd.read_parquet(data_dir() / "processed" / "unified.parquet")
    train_df, val_df, test_df = _subset(uni, n_rows=args.subset_rows, seed=args.seed)

    bench_dir = Path("results/bench_intranode")
    bench_dir.mkdir(parents=True, exist_ok=True)

    cpu_models = [
        name for name in assigned
        if by_name[name].get("placement", "gpu") == "cpu"
    ]
    if len(cpu_models) < 2:
        raise SystemExit(
            "Need ≥2 CPU-placed models in manifest to benchmark CPU concurrency. "
            f"manifest rank {args.rank}: {assigned}"
        )

    runs: dict[str, Any] = {}

    # ── Concurrent CPU ──────────────────────────────────────────────────
    t_conc_wall = time.perf_counter()
    futures = []
    with ThreadPoolExecutor(max_workers=len(cpu_models)) as ex:
        for name in cpu_models:
            futures.append(
                ex.submit(
                    _train_wallclock,
                    by_name[name],
                    train_df,
                    val_df,
                    test_df,
                    rank=args.rank,
                    out_suffix=bench_dir / "concurrent",
                    epochs=args.epochs,
                    batch=args.batch_size,
                    max_seq_len=args.max_seq_len,
                    tr_yaml=train_cfg,
                )
            )
    conc_results = [f.result() for f in futures]
    t_conc = time.perf_counter() - t_conc_wall
    runs["concurrent_cpu_wall_s"] = t_conc
    runs["concurrent_cpu_models"] = {
        m["model"]: {
            "self_wall_s": m["wallclock_s"],
            "macro_f1_test": m["test"].get("eval_macro_f1")
            or m["test"].get("macro_f1"),
        }
        for m in conc_results
    }

    # ── Sequential CPU (same workloads, one-after-another) ──────────────
    t_seq_wall = time.perf_counter()
    seq_metrics = []
    for name in cpu_models:
        seq_metrics.append(
            _train_wallclock(
                by_name[name],
                train_df,
                val_df,
                test_df,
                rank=args.rank,
                out_suffix=bench_dir / "sequential_cpu",
                epochs=args.epochs,
                batch=args.batch_size,
                max_seq_len=args.max_seq_len,
                tr_yaml=train_cfg,
            )
        )
    runs["sequential_cpu_wall_s"] = time.perf_counter() - t_seq_wall
    runs["sequential_cpu_models"] = {
        m["model"]: {
            "self_wall_s": m["wallclock_s"],
            "macro_f1_test": m["test"].get("eval_macro_f1")
            or m["test"].get("macro_f1"),
        }
        for m in seq_metrics
    }

    slow_vs_fast = runs["sequential_cpu_wall_s"] / runs["concurrent_cpu_wall_s"]

    summary = {
        "rank_benchmarked": args.rank,
        "cpu_models": cpu_models,
        "subset_rows": args.subset_rows,
        "epochs": args.epochs,
        "effective_batch_product": args.batch_size * models_yaml["training"].get(
            "gradient_accumulation_steps", 1
        ),
        "sequential_wall_s": runs["sequential_cpu_wall_s"],
        "concurrent_wall_s": runs["concurrent_cpu_wall_s"],
        # >1 means concurrent is *slower* overall (core oversubscription typical on laptops)
        "sequential_vs_concurrent_slowdown_factor": slow_vs_fast,
        "raw": runs,
    }

    (bench_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    md = bench_dir / "README.md"
    md.write_text(
        f"# Intra-node CPU parallelism benchmark (`rank_{args.rank}`)\n\n"
        f"*Models*: {cpu_models}\n\n"
        f"*Train rows*: {len(train_df)}  *(subset cap {args.subset_rows})*\n\n"
        "| schedule | wall-clock |\n|---|---:|\n"
        f"| concurrent | {runs['concurrent_cpu_wall_s']:.1f} s |\n"
        f"| sequential | {runs['sequential_cpu_wall_s']:.1f} s |\n\n"
        f"**Sequential wall ÷ concurrent wall** (values >1 ⇒ concurrent is slower): "
        f"{slow_vs_fast:.2f} ×\n"
        "---\n"
        "**Interpretation**\n\n"
        "If concurrent wall-clock ≫ sequential × (1/`n_models`), the CPU cores are "
        "**oversubscribed** by multiple HF `Trainer` instances + tokenisation pools — "
        "exactly what we observed on 16-thread laptop CPUs.",
        encoding="utf-8",
    )
    logger.info("Wrote results to %s", bench_dir.resolve())


if __name__ == "__main__":
    main()
