"""Matplotlib plots for the aggregated leaderboard + per-rank profiling."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_leaderboard(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(df))))
    bars = ax.barh(df["model"] + "  (rank " + df["rank"].astype(str) + ")",
                   df["macro_f1"], color="#3672a4")
    for b, v in zip(bars, df["macro_f1"]):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center")
    ax.set_xlabel("Test Macro-F1")
    ax.set_xlim(0, 1)
    ax.set_title("FastUrduGuard — per-model leaderboard (Phase 3)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_timing(timings: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    rows = []
    for t in timings:
        for name, info in t["models"].items():
            rows.append({"rank": t["rank"], "model": name,
                         "elapsed_s": info.get("elapsed_s", 0)})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    pivot = df.pivot_table(index="model", columns="rank", values="elapsed_s", fill_value=0)
    pivot.plot(kind="barh", ax=ax)
    ax.set_xlabel("training wall-clock (s)")
    ax.set_title("Per-model training time, partitioned by rank")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, classes: list[str], out: Path,
                   title: str = "Confusion matrix") -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_speedup(seq_total_s: float, parallel_wall_s: float, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Sequential\n(estimate)", "Parallel\n(2 ranks)"],
           [seq_total_s, parallel_wall_s], color=["#aa3939", "#226f54"])
    ax.set_ylabel("seconds")
    sp = seq_total_s / parallel_wall_s if parallel_wall_s > 0 else 0
    ax.set_title(f"Wall-clock speedup ≈ {sp:.2f}×")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
