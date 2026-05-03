"""Aggregate per-rank artifacts into the final leaderboard + ensemble eval.

  python -m fasturduguard.agg.aggregate
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

from fasturduguard.agg.ensemble import weighted_softmax, weighted_vote
from fasturduguard.agg import plots
from fasturduguard.utils import (
    data_dir, load_yaml, results_dir, setup_logging,
)

log = logging.getLogger("fug.agg")


def collect_metrics(rank_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for rd in rank_dirs:
        mdir = rd / "metrics"
        if not mdir.is_dir():
            continue
        for f in sorted(mdir.glob("*.json")):
            if f.name in {"timings.json", "summary.json"}:
                continue
            try:
                m = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            test = m.get("test", {})
            rows.append({
                "rank": m.get("rank"),
                "model": m.get("model"),
                "macro_f1": test.get("eval_macro_f1") or test.get("macro_f1") or 0.0,
                "weighted_f1": test.get("eval_weighted_f1") or test.get("weighted_f1") or 0.0,
                "accuracy": test.get("eval_accuracy") or test.get("accuracy") or 0.0,
                "training_runtime_s": m.get("training_runtime_s", 0.0),
                "peak_vram_mb": m.get("peak_vram_mb", 0),
                "device_used": m.get("device_used"),
                "checkpoint_dir": m.get("checkpoint_dir"),
            })
    return pd.DataFrame(rows)


def predict_probabilities(model_cfg: dict, ckpt_dir: Path, texts: list[str], max_len: int = 256, batch_size: int = 32) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    is_lora = (Path(ckpt_dir) / "is_lora").exists() or (Path(ckpt_dir) / "adapter_config.json").exists()
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device).eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            logits = mdl(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out.append(probs)
    return np.concatenate(out, axis=0)


def main() -> None:
    setup_logging(prefix="aggregate")
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", type=Path, default=results_dir())
    p.add_argument("--unified", type=Path, default=data_dir() / "processed" / "unified.parquet")
    p.add_argument("--label_col", default="label_4")
    p.add_argument("--no_predict", action="store_true",
                   help="Skip ensemble prediction (just produce leaderboard).")
    args = p.parse_args()

    cfg = load_yaml("models.yaml")
    by_name = {m["name"]: m for m in cfg["models"]}
    n_classes = cfg["training"]["num_labels_4class"]
    classes = list(cfg["classes_4"].values())

    # 1) Collect per-model metrics
    rank_dirs = sorted([d for d in args.results_root.iterdir() if d.is_dir() and d.name.startswith("rank_")])
    if not rank_dirs:
        raise SystemExit(f"No rank_* dirs under {args.results_root}")
    leaderboard = collect_metrics(rank_dirs)
    if leaderboard.empty:
        raise SystemExit("No per-model metrics.json found.")
    leaderboard = leaderboard.sort_values("macro_f1", ascending=False)
    log.info("\n%s", leaderboard.to_string(index=False))

    # 2) Plots: leaderboard + per-rank timing + speedup
    plots.plot_leaderboard(leaderboard, args.results_root / "leaderboard.png")
    leaderboard.to_csv(args.results_root / "leaderboard.csv", index=False)

    timings = []
    for rd in rank_dirs:
        f = rd / "metrics" / "timings.json"
        if f.exists():
            t = json.loads(f.read_text(encoding="utf-8"))
            t["rank"] = int(rd.name.split("_")[1])
            timings.append(t)
    if timings:
        plots.plot_timing(timings, args.results_root / "timing.png")
        seq_total = sum(info.get("elapsed_s", 0)
                        for t in timings for info in t["models"].values())
        parallel_wall = max((t.get("wall_s", 0) for t in timings), default=0)
        if parallel_wall > 0:
            plots.plot_speedup(seq_total, parallel_wall, args.results_root / "speedup.png")
            (args.results_root / "timing.json").write_text(
                json.dumps({"sequential_total_s": seq_total,
                            "parallel_wall_s": parallel_wall,
                            "speedup": seq_total / parallel_wall},
                           indent=2),
                encoding="utf-8",
            )

    if args.no_predict:
        return

    # 3) Ensemble eval on the held-out test set
    df = pd.read_parquet(args.unified)
    test = df[df["split"] == "test"].reset_index(drop=True)
    log.info("Ensemble eval over %d test rows", len(test))

    probs_list, weights_list, names_used = [], [], []
    for _, row in leaderboard.iterrows():
        ckpt = Path(str(row["checkpoint_dir"]))
        if not ckpt.exists():
            log.warning("Skip %s: checkpoint missing %s", row["model"], ckpt)
            continue
        try:
            probs = predict_probabilities(by_name[row["model"]], ckpt, test["text"].tolist())
        except Exception as e:
            log.warning("Skip %s: predict failed (%s)", row["model"], e)
            continue
        probs_list.append(probs)
        weights_list.append(float(row["macro_f1"]))
        names_used.append(row["model"])

    if not probs_list:
        log.warning("No usable models for ensemble.")
        return

    y_true = test[args.label_col].values
    preds_argmax = [np.argmax(p, axis=1) for p in probs_list]

    vote_pred = weighted_vote(preds_argmax, weights_list, n_classes)
    soft_pred = weighted_softmax(probs_list, weights_list)

    def _row(name, y_pred):
        return {
            "method": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

    summary = [
        _row(f"single::{n}", p) for n, p in zip(names_used, preds_argmax)
    ] + [
        _row("ensemble::weighted_vote", vote_pred),
        _row("ensemble::weighted_softmax", soft_pred),
    ]
    out = pd.DataFrame(summary)
    out.to_csv(args.results_root / "ensemble_eval.csv", index=False)
    log.info("\n%s", out.to_string(index=False))

    # ensemble confusion matrix + classification report
    cm = confusion_matrix(y_true, soft_pred, labels=list(range(n_classes)))
    plots.plot_confusion(cm, classes, args.results_root / "ensemble_confusion.png",
                         title="Weighted-softmax ensemble — confusion matrix")
    rep = classification_report(y_true, soft_pred, target_names=classes,
                                digits=4, zero_division=0)
    (args.results_root / "ensemble_report.txt").write_text(rep, encoding="utf-8")


if __name__ == "__main__":
    main()
