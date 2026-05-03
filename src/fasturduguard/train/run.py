"""Per-rank training driver (Mode A: model-parallel across nodes).

Reads coordinator/manifest.json, looks up this rank's assigned model list,
trains them sequentially on the local GPU/CPU. Within a rank, models that are
config'd as `placement: cpu` are dispatched to a `ThreadPoolExecutor`
concurrently with the GPU model -> we get a per-node hybrid CPU+GPU schedule
in addition to the cross-node split.

Run on each node:
    $env:FUG_RANK="0" ; python -m fasturduguard.train.run
    $env:FUG_RANK="1" ; python -m fasturduguard.train.run
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from fasturduguard.coord.manifest import get_assignment
from fasturduguard.utils import (
    coordinator_dir, data_dir, get_rank, load_json, load_yaml,
    rank_results_dir, set_seed, setup_logging,
)
from fasturduguard.train.trainer import train_one


log = logging.getLogger("fug.train.run")


def _load_data(shard_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if shard_path and shard_path.exists():
        df = pd.read_parquet(shard_path)
    else:
        unified = data_dir() / "processed" / "unified.parquet"
        df = pd.read_parquet(unified)
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return train, val, test


def _train_model_wrapper(args: dict[str, Any]) -> dict[str, Any]:
    """Top-level wrapper so it's picklable for ThreadPoolExecutor (no-op there but safer)."""
    return train_one(**args)


def main() -> None:
    setup_logging(prefix=f"rank{get_rank()}")
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=coordinator_dir() / "manifest.json")
    p.add_argument("--shard", type=Path, default=None,
                   help="Optional rank-shard parquet (Mode B). Default: full unified.parquet.")
    p.add_argument("--label_col", default="label_4", choices=("label_2", "label_4"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_train_rows", type=int, default=None,
                   help="Optional cap on training rows for quick smoke runs.")
    p.add_argument("--max_eval_rows", type=int, default=None,
                   help="Optional cap on val/test rows for smoke runs.")
    p.add_argument("--only", default=None,
                   help="Train only this single model name (override manifest).")
    args = p.parse_args()

    rank = get_rank()
    rank_dir = rank_results_dir(rank)
    log.info("Rank=%d  rank_dir=%s", rank, rank_dir)

    if args.manifest.exists():
        manifest = load_json(args.manifest)
        assigned = get_assignment(manifest, rank)
    else:
        log.warning("No manifest at %s -- defaulting rank=%d to all models", args.manifest, rank)
        assigned = [m["name"] for m in load_yaml("models.yaml")["models"]]

    if args.only:
        assigned = [m for m in assigned if m == args.only] or [args.only]

    log.info("Assigned models for rank %d: %s", rank, assigned)

    cfg = load_yaml("models.yaml")
    by_name = {m["name"]: m for m in cfg["models"]}
    tr_cfg = cfg["training"]
    set_seed(tr_cfg["seed"])

    train_df, val_df, test_df = _load_data(args.shard)
    if args.max_train_rows:
        train_df = train_df.sample(n=min(args.max_train_rows, len(train_df)),
                                   random_state=tr_cfg["seed"]).reset_index(drop=True)
    if args.max_eval_rows:
        n = args.max_eval_rows
        val_df = val_df.sample(n=min(n, len(val_df)), random_state=tr_cfg["seed"]).reset_index(drop=True)
        test_df = test_df.sample(n=min(n, len(test_df)), random_state=tr_cfg["seed"]).reset_index(drop=True)
    log.info("Data sizes: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    epochs = args.epochs or tr_cfg["epochs"]
    bs = args.batch_size or tr_cfg["per_device_train_batch_size"]
    seq = args.max_seq_len or tr_cfg["max_seq_len"]

    # Split assigned set by placement: gpu queue is sequential, cpu queue runs in parallel threads
    gpu_models = [n for n in assigned if by_name[n].get("placement") == "gpu"]
    cpu_models = [n for n in assigned if by_name[n].get("placement") == "cpu"]
    log.info("GPU queue=%s   CPU queue=%s", gpu_models, cpu_models)

    results: list[dict[str, Any]] = []
    timings: dict[str, Any] = {"start": time.time(), "models": {}}

    def _job(name: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        m = train_one(
            model_cfg=by_name[name],
            train_df=train_df, val_df=val_df, test_df=test_df,
            out_dir=rank_dir,
            rank=rank,
            label_col=args.label_col,
            epochs=epochs,
            lr=tr_cfg["learning_rate"],
            batch_size=bs,
            grad_accum=tr_cfg["gradient_accumulation_steps"],
            max_seq_len=seq,
            weight_decay=tr_cfg["weight_decay"],
            warmup_ratio=tr_cfg["warmup_ratio"],
            fp16=tr_cfg.get("fp16", True),
            seed=tr_cfg["seed"],
        )
        timings["models"][name] = {"elapsed_s": time.perf_counter() - t0,
                                   "device_used": m.get("device_used")}
        return m

    # CPU models: 1 thread each in parallel (capped to len). The HF Trainer releases GIL
    # during forward/backward, so threads are an OK approximation of `multiprocessing` for
    # this workload and don't risk import-storms on Windows.
    cpu_futures = []
    if cpu_models:
        ex = ThreadPoolExecutor(max_workers=min(len(cpu_models), 2))
        cpu_futures = [ex.submit(_job, n) for n in cpu_models]

    for name in gpu_models:
        try:
            results.append(_job(name))
        except Exception as e:
            log.exception("GPU model %s failed: %s", name, e)
            timings["models"][name] = {"error": str(e)}

    for fut in as_completed(cpu_futures):
        try:
            results.append(fut.result())
        except Exception as e:
            log.exception("CPU model failed: %s", e)

    timings["end"] = time.time()
    timings["wall_s"] = timings["end"] - timings["start"]

    (rank_dir / "metrics" / "timings.json").write_text(
        json.dumps(timings, indent=2), encoding="utf-8"
    )
    (rank_dir / "metrics" / "summary.json").write_text(
        json.dumps([{
            "model": r["model"], "macro_f1": r["test"].get("eval_macro_f1") or r["test"].get("macro_f1"),
            "accuracy": r["test"].get("eval_accuracy") or r["test"].get("accuracy"),
            "training_runtime_s": r.get("training_runtime_s"),
            "peak_vram_mb": r.get("peak_vram_mb"),
        } for r in results], indent=2), encoding="utf-8"
    )

    log.info("Rank %d done in %.1f s. %d models trained.", rank, timings["wall_s"], len(results))


if __name__ == "__main__":
    main()
