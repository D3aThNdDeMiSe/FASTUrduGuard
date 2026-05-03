#!/usr/bin/env python3
"""Build the FastUrduGuard viva slide deck (formal structure, metric placeholders).

Usage (from repo root):
  python scripts/build_presentation.py
  python scripts/build_presentation.py --out path/to/deck.pptx
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pptx import Presentation
from pptx.util import Inches, Pt

REPO_ROOT = Path(__file__).resolve().parents[1]


def _placeholder_png(title: str, subtitle: str = "Replace before submission") -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(10.0, 5.2), dpi=120)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#eceff4")
    ax.add_patch(
        FancyBboxPatch(
            (0.04, 0.08), 0.92, 0.84, boxstyle="round,pad=0.02",
            linewidth=2.5, edgecolor="#5e81ac", facecolor="#e5e9f0", linestyle="--",
        )
    )
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=20, fontweight="bold", color="#2e3440")
    ax.text(0.5, 0.38, subtitle, ha="center", va="center", fontsize=13, color="#4c566a")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _workflow_png() -> io.BytesIO:
    """Simple workflow diagram (not performance metrics)."""
    fig, ax = plt.subplots(figsize=(10.0, 4.2), dpi=120)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    boxes = [
        (0.3, 2.1, 1.4, 0.9, "Raw corpora\n(3 sources)"),
        (2.1, 2.1, 1.4, 0.9, "unify +\ndedup + split"),
        (3.9, 2.1, 1.4, 0.9, "relabel\n(4 classes)"),
        (5.7, 2.1, 1.4, 0.9, "preprocess\n(Roman Urdu)"),
        (2.4, 0.5, 1.5, 0.85, "Rank 0\n(models A–C)"),
        (4.4, 0.5, 1.5, 0.85, "Rank 1\n(models D–F)"),
        (6.4, 0.5, 1.55, 0.85, "GitHub /\naggregation"),
        (8.05, 0.5, 1.55, 0.85, "Ensemble +\nexplain +\nbench"),
    ]
    for x, y, w, h, txt in boxes:
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", linewidth=1.8,
                           edgecolor="#2e3440", facecolor="#88c0d0", alpha=0.35),
        )
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9.5,
                color="#2e3440", fontweight="medium")

    # arrows approx
    for x0, x1, y in [(1.72, 2.08, 2.55), (3.52, 3.88, 2.55), (5.32, 5.68, 2.55)]:
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="->", color="#3b4252", lw=2))
    ax.plot([7.85, 7.85], [2.1, 2.95], color="#3b4252", lw=2)  # drop hint
    ax.annotate("", xy=(8.95, 1.95), xytext=(7.35, 2.55),
                arrowprops=dict(arrowstyle="->", color="#3b4252", lw=1.5))
    ax.text(5.0, 3.55, "End-to-end workflow (placeholder layout — edit in PowerPoint if desired)",
            ha="center", fontsize=11, fontweight="bold", color="#2e3440")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _set_title(slide, text: str) -> None:
    slide.shapes.title.text = text


def _body(slide) -> None:
    return slide.placeholders[1]


def _add_bullets(slide, title: str, lines: list[str]) -> None:
    _set_title(slide, title)
    tf = _body(slide).text_frame
    tf.clear()
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = Pt(20)


def build_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)   # widescreen 16:9
    prs.slide_height = Inches(7.5)

    blank = prs.slide_layouts[6]

    # 1 — Title
    t = prs.slides.add_slide(prs.slide_layouts[0])
    t.shapes.title.text = "FastUrduGuard"
    st = t.placeholders[1]
    st.text = (
        "Integrated parallel & distributed computing with NLP for Urdu misinformation detection\n\n"
        "Team: Sameer Asif (22I-0493) · Immad Shah (22I-0395)\n"
        "Course: Phase-3 · Cluster 1 (PDC + NLP)\n"
        "Repository: github.com/D3aThNdDeMiSe/FASTUrduGuard"
    )
    for p in st.text_frame.paragraphs:
        p.font.size = Pt(18)

    # 2 — Problem
    s = prs.slides.add_slide(prs.slide_layouts[1])
    _add_bullets(
        s,
        "Problem statement",
        [
            "Urdu is widely used yet under-served by English-centric moderation and detection tools.",
            "False or misleading items spread quickly through text, memes, and Roman-Urdu social posts.",
            "Manual fact-checking does not scale; automated pipelines must handle noisy text and low-resource settings.",
            "Practical systems need fine-grained labels, explainability, and deployment-aware training — not lab-only accuracy.",
        ],
    )

    # 3 — Literature / baseline
    s = prs.slides.add_slide(prs.slide_layouts[1])
    _add_bullets(
        s,
        "Literature review — baseline (Islam et al., 2025)",
        [
            "Islam et al., “Unified LLMs for Misinformation Detection in Low-Resource Linguistic Settings,” arXiv:2506.01587, 2025.",
            "Introduces Urdu-LLD and strong binary {real, fake} results with six transformer encoders (e.g., mBERT, XLM-R).",
            "Single-workstation, sequential training; limited discussion of Roman Urdu, multi-class labels, or explainability.",
            "We adopt the same encoder families as a scientific control, but extend the problem setting and systems story.",
        ],
    )

    # 4 — Objectives
    s = prs.slides.add_slide(prs.slide_layouts[1])
    _add_bullets(
        s,
        "Research objectives",
        [
            "Unify and clean multiple public Urdu corpora under one schema with careful deduplication.",
            "Move beyond binary labels toward a four-way schema (Real, Fake, PartiallyFalse, Satire) with rule-based expansion.",
            "Fine-tune six encoders and combine them with a validation-weighted ensemble for robustness.",
            "Add explainability (attention + sentence-level LIME) and an inference-oriented evaluation path.",
            "Exploit two consumer GPUs on separate networks using a Git-coordinated training and aggregation workflow.",
        ],
    )

    # 5 — Proposed solution (text)
    s = prs.slides.add_slide(prs.slide_layouts[1])
    _add_bullets(
        s,
        "Proposed solution — overview",
        [
            "Production-shaped pipeline: data fusion → multilingual preprocessing → distributed training manifests → metrics via Git.",
            "Optional Federated-Averaging of small LoRA adapters where bandwidth is constrained.",
            "Post-training: aggregator builds leaderboards/plots; benchmark and explanation scripts support demos.",
            "Technical stack: PyTorch, Hugging Face Transformers/Trainer, PEFT-LoRA, scikit-learn metrics.",
        ],
    )

    # 6 — Workflow diagram
    slide = prs.slides.add_slide(blank)
    slide.shapes.add_picture(_workflow_png(), Inches(0.45), Inches(1.05), width=Inches(12.35))
    tx = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.0), Inches(0.65))
    tx.text_frame.text = "Workflow — data, training coordination, aggregation"
    tx.text_frame.paragraphs[0].font.size = Pt(28)
    tx.text_frame.paragraphs[0].font.bold = True

    # 7 — Methodology
    s = prs.slides.add_slide(prs.slide_layouts[1])
    _add_bullets(
        s,
        "Methodology — NLP",
        [
            "Unicode normalization (NFC), noise removal (URLs, HTML, elongated characters).",
            "Roman-Urdu detection and lexicon-aided transliteration when Roman tokens dominate.",
            "Stratified train / validation / test split on multi-class targets; multilingual tokenizers seq. length 256 (configurable).",
            "Fine-tuning: AdamW, weight decay, warmup + linear decay; optional LoRA on larger encoders.",
        ],
    )

    s = prs.slides.add_slide(prs.slide_layouts[1])
    _add_bullets(
        s,
        "Methodology — PDC / coordination",
        [
            "Rank-specific model queues from coordinator/manifest.json (load-balanced by footprint).",
            "Home / campus networks behind NAT: asynchronous Git pushes of lightweight metrics and artefacts.",
            "GPU-serial scheduling per laptop (--placement_override gpu) avoids CPU contention from concurrent tokenization.",
            "Aggregation job merges checkpoints/metrics locally for ensemble scoring and reproducible plots.",
        ],
    )

    # 8 — Challenges
    s = prs.slides.add_slide(prs.slide_layouts[1])
    _add_bullets(
        s,
        "Challenges during implementation",
        [
            "Mixed-precision training: aligning Trainer AMP (fp16/bf16) with model dtypes to avoid scaler/clip failures.",
            "Consumer GPU stack: limited or absent BitsAndBytes sm_120 support — fallback to BF16/FP16 + LoRA instead of NF4 everywhere.",
            "Two-node parallelism without rendezvous TCP: GitOps choreography instead of classical NCCL collectives.",
            "Class imbalance across four labels vs. headline-heavy corpora; weighted ensemble mitigates weak tail classes.",
        ],
    )

    # 9 — Comparison with baseline
    s = prs.slides.add_slide(prs.slide_layouts[1])
    _add_bullets(
        s,
        "Comparison with the baseline approach",
        [
            "Labels: binary (Islam et al.) → four-way schema with explicit partial-fake and satire handling.",
            "Data: single curated head-line set → unified multi-source corpus with hash-based cross-corpus deduplication.",
            "Training: one machine sequential → two-rank manifest + Git-mediated coordination and optional FedAvg-of-LoRA.",
            "Analysis: accuracy-focused reporting → ensemble, confusion analysis, explainability hooks, and inference benchmarks (to be finalized).",
        ],
    )

    # 10–11 — Placeholder figures
    for title, ph, top in [
        ("Results — model & ensemble performance (placeholder)", "Insert bar chart: accuracy / macro-F1 / per-class F1 from aggregate.py", Inches(1.15)),
        ("Results — confusion matrix & error analysis (placeholder)", "Insert heatmap: test-set confusion matrix + optional calibration plot", Inches(1.15)),
    ]:
        slide = prs.slides.add_slide(blank)
        slide.shapes.add_picture(_placeholder_png(ph), Inches(0.9), top, width=Inches(11.5))
        tx = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.0), Inches(0.7))
        tx.text_frame.text = title
        tx.text_frame.paragraphs[0].font.size = Pt(26)
        tx.text_frame.paragraphs[0].font.bold = True

    # 12 — Technical readiness / close
    s = prs.slides.add_slide(prs.slide_layouts[1])
    _add_bullets(
        s,
        "Technical readiness & oral examination",
        [
            "Laptop with conda/venv mirroring repository requirements.txt; datasets available locally.",
            "Trained checkpoints and parquet shards kept on-disk (large weights typically gitignored; metrics may be synced).",
            "Prepared live paths: scripts/node_train.py, aggregate pipeline, benchmark and explain_demo entry points.",
            "Thank you — questions welcome (Q&A guideline: approximately 5–7 minutes).",
        ],
    )
    for paragraph in _body(s).text_frame.paragraphs:
        paragraph.font.size = Pt(19)

    return prs


def default_output_paths() -> list[Path]:
    paths = [REPO_ROOT / "docs" / "FastUrduGuard_Presentation.pptx"]
    sibling = REPO_ROOT.parent / "Phase 3" / "FastUrduGuard_Presentation.pptx"
    if sibling.parent.is_dir():
        paths.append(sibling.resolve())
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, action="append", help="Destination .pptx (repeatable)")
    args = ap.parse_args()

    outs = list(args.out) if args.out else default_output_paths()
    prs = build_deck()
    for path in outs:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        prs.save(str(path))
        print("Wrote", path)


if __name__ == "__main__":
    main()
