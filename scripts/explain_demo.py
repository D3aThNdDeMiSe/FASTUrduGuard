"""End-to-end explainability demo: predict + attention + LIME -> HTML."""
from __future__ import annotations

import argparse
from pathlib import Path

from fasturduguard.explain.attention import token_attribution
from fasturduguard.explain.lime_expl import explain_sentences
from fasturduguard.explain.visualize import render_html, write_html
from fasturduguard.utils import load_yaml, results_dir, setup_logging


def find_default_checkpoint() -> Path:
    """Use the highest-macro-F1 checkpoint reported in results/leaderboard.csv if present."""
    lb = results_dir() / "leaderboard.csv"
    if lb.exists():
        import pandas as pd
        df = pd.read_csv(lb).sort_values("macro_f1", ascending=False)
        for _, row in df.iterrows():
            ck = Path(str(row["checkpoint_dir"]))
            if ck.exists():
                return ck
    # fallback: first checkpoint we can find
    for rd in sorted(results_dir().glob("rank_*")):
        for ck in sorted((rd / "checkpoints").glob("*")):
            if (ck / "config.json").exists() or (ck / "adapter_config.json").exists():
                return ck
    raise SystemExit("No usable checkpoint found.")


def main() -> None:
    setup_logging(prefix="explain")
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--out", type=Path, default=results_dir() / "explanation.html")
    p.add_argument("--num_samples", type=int, default=200)
    args = p.parse_args()

    ck = args.checkpoint or find_default_checkpoint()
    cfg = load_yaml("models.yaml")
    classes = list(cfg["classes_4"].values())

    lime = explain_sentences(ck, args.text, num_samples=args.num_samples)
    attn = token_attribution(ck, args.text)

    html = render_html(
        predicted=lime["predicted"],
        class_names=classes,
        sentences=lime["sentences"],
        sent_scores=lime["scores"],
        tokens=attn["tokens"],
        tok_scores=attn["scores"],
        title=f"FastUrduGuard explanation ({ck.name})",
    )
    write_html(args.out, html)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
