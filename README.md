# FastUrduGuard

**Efficient Parallel Ensemble with Explainable Multi-Class Fake News Detection for Low-Resource Urdu**

Phase-3 implementation of the *FastUrduGuard* project (PDC + NLP, Cluster 1).
Base paper: Islam et al., *Unified Large Language Models for Misinformation Detection in Low-Resource Linguistic Settings* (arXiv:2506.01587, 2025).

This repo implements the full **two-node, rank-based, decentralized training pipeline** described in the Phase-2 mid-report and extends it for the final submission. The two physical nodes (RTX 5060, RTX 5060 Ti) sit on different ISPs and never share an interconnect; coordination happens **through a shared Git repository** (this repo) acting as the parameter / manifest server.

---

## 1. What this codebase does

| Component | NLP role | PDC role |
|---|---|---|
| `data/` | Unify three Urdu FND corpora, multi-class re-annotation | Parallel multi-process pipeline, Pool of N workers |
| `preprocess/` | NFC, Roman-Urdu detect+transliterate, code-mix safe | Pipelined producer/consumer stages |
| `coord/` | — | Manifest-driven rank scheduler, git pull/commit/push as the "wire" |
| `train/` | QLoRA fine-tune of 6 transformers on 4-class task | Hybrid CPU/GPU concurrent training, per-model rank assignment, FedAvg-of-LoRA option |
| `agg/` | Weighted majority-vote ensemble | Pulls all rank artifacts, aggregates, builds leaderboard |
| `infer/` | Multi-model ensemble prediction | Dynamic batching, CUDA-stream overlap, ThreadPoolExecutor ensemble |
| `explain/` | Attention + LIME unified rationale | Batched attention extraction |

---

## 2. Cluster topology

```
                        ┌────────────────────────────────────┐
                        │   GitHub repo  (this one)          │
                        │   ─ coordinator/manifest.json      │
                        │   ─ data/processed/{shards}.parquet│
                        │   ─ results/rank_0/, rank_1/       │
                        │   ─ .github/workflows/aggregate.yml│
                        └─────────────────┬──────────────────┘
                                          │ git pull / push (TLS)
              ┌──────────────────┬────────┴────────┬──────────────────┐
              │  Rank 0 (5060)   │                 │  Rank 1 (5060 Ti) │
              │  ISP A           │                 │  ISP B            │
              │  XLM-R + DistilB │                 │  DeBERTa + mBERT  │
              │  + XLNet (CPU)   │                 │  + RoBERTa (CPU)  │
              └──────────────────┘                 └──────────────────┘
```

### Two coordination modes (you can pick either, or run both)

* **Mode A — Model-parallel (default).** Each rank owns a subset of the six models, trains them on the **full** dataset locally, pushes checkpoints + metrics + plots back. Aggregator forms a weighted majority-vote ensemble across all six. This is the "embarrassingly parallel" PDC story; it is what the Phase-2 hybrid CPU/GPU schedule maps onto when the two CPU/GPU buckets live in different machines.

* **Mode B — Data-parallel + FedAvg of LoRA adapters (advanced).** Both ranks fine-tune the **same** model (XLM-R) on **disjoint data shards**, locally for *K* mini-rounds, then average their LoRA adapter tensors via the repo (`fed_round_t/adapter.safetensors`) before continuing. This is the closest you can get to synchronous data-parallel training over consumer ISPs without any direct interconnect. ~5 LoRA adapters × ~2 MB each per round = trivial bandwidth.

A GitHub Actions workflow watches `results/` and automatically re-runs aggregation when both ranks have pushed completed artifacts.

---

## 3. Quick start (each node)

Both nodes run identical commands. Only the env var `FUG_RANK` differs.

```bash
# 0. clone the coordinator repo
git clone <this-repo-url> FastUrduGuard && cd FastUrduGuard
pip install -r requirements.txt

# 1. one-time: prepare and shard the data (rank 0 only, then push)
python -m fasturduguard.data.unify       --raw <path-to-datasets_raw>
python -m fasturduguard.data.relabel     --mode rule
python -m fasturduguard.data.shards      --num_ranks 2

# 2. on each node:
#   Rank 0 (RTX 5060):
$env:FUG_RANK="0"; python scripts/node_train.py --mode model_parallel
#   Rank 1 (RTX 5060 Ti):
$env:FUG_RANK="1"; python scripts/node_train.py --mode model_parallel

# 3. each node pushes metrics, plots, profile, and **full** `checkpoints/<model>/` trees
#    (uses `git add -f` because weights are normally gitignored).
#    Minimal push: `python scripts/push_artifacts.py --adapters-only`
#    GitHub rejects single files > 100 MiB — use Git LFS or split hosts if a shard exceeds that.

python scripts/push_artifacts.py

# 4. once both ranks are present (CI does this automatically too):
python scripts/aggregate.py

# 5. inference benchmarks + explainability demo
python scripts/benchmark.py
python scripts/explain_demo.py --text "<sample article>"
```

For the FedAvg mode:

```bash
$env:FUG_RANK="0"; python scripts/node_fedavg.py --rounds 5
$env:FUG_RANK="1"; python scripts/node_fedavg.py --rounds 5
```

---

## 4. Datasets

Stored in `data/raw/` (or extract once from your local zips). Three sources:

| Dataset | Files | Rows | Source |
|---|---|---|---|
| Urdu-LLD (Ax-to-Grind, Notri, Labeled_Urdu_News) | `Urdu-Large-language-dataset-main/` | ~27 k | https://github.com/MislamSatti/Urdu-Large-language-dataset |
| Bend the Truth (Amjad et al., 2020) | `Datasets-for-Urdu-news-master/Urdu Fake News Dataset/` | 900 | https://github.com/MaazAmjad/Datasets-for-Urdu-news |
| Hook & Bait (Harris et al., 2025) | `Hook-and-Bait-Urdu-main/` | ~78 k | https://github.com/Sheetal83/Hook-and-Bait-Urdu |

The unify step normalizes these to a single parquet with columns
`text, label_2, label_4, source, domain, split` and de-duplicates
overlapping rows (Hook & Bait re-shipped Ax-to-Grind starting at row 31191).

---

## 5. Hardware notes

* **RTX 50-series (Blackwell, sm_120)**: requires CUDA 12.8+ and PyTorch nightly with `cu128`. `bitsandbytes` 4-bit currently **does not support sm_120**; the code auto-detects this and falls back to FP16 LoRA, which still fits on 8 GB if `max_seq_len=256`.
* **CPU-only fallback**: every training step works without a GPU; throughput drops ~25× on `xlm-roberta-base` but the pipeline is the same.

See `docs/ENVIRONMENT.md` for the full setup script.

---

## 6. Reproducing Phase-2's expected numbers

Defaults in `configs/models.yaml` reproduce the Phase-2 targets:

| Metric | Target | Where to find it |
|---|---|---|
| Macro-F1 (4-class) | ≥ 0.82 | `results/leaderboard.csv` |
| Total wall-clock (6 models) | ≤ 0.5 × sequential | `results/timing.json` |
| VRAM 4-bit reduction | ≥ 30 % | `results/profile/vram.csv` |
| Inference throughput | ≥ 2× batched | `results/bench/throughput.csv` |
| Quantization F1 drop | ≤ 2 % | `results/profile/quant_vs_fp16.csv` |

---

## 7. Layout

```
FastUrduGuard/
├── coordinator/        # manifest.json, FedAvg checkpoints, metadata
├── configs/            # YAML configs (models, datasets, training)
├── data/
│   ├── raw/            # extracted dataset folders (gitignored)
│   └── processed/      # unified parquet + per-rank shards
├── docs/
├── results/            # rank_0/, rank_1/, aggregated leaderboard, plots
├── scripts/            # CLI entrypoints used in the README quick-start
├── src/fasturduguard/  # library code
└── .github/workflows/  # CI: aggregate-on-completion
```

See each subfolder's `__init__` docstring for module-level docs.
