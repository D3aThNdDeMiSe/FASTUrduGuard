# End-to-end pipeline

```
                                   ┌──────────────────────────────┐
                                   │  Stage 1 — Data preparation  │
                                   │  (rank 0 only, ~5 min)       │
                                   ├──────────────────────────────┤
  data/raw/* (3 zips extracted) ──>│  fasturduguard.data.unify    │
                                   │     ├─ load 3 sources         │
                                   │     ├─ md5 dedup              │
                                   │     ├─ stratified split       │
                                   │  fasturduguard.data.relabel   │
                                   │     ├─ rule-based 4-class     │
                                   │     └─ optional Qwen LLM      │
                                   │  fasturduguard.data.shards    │
                                   │     └─ rank_0/1.parquet       │
                                   │  fasturduguard.coord.manifest │
                                   │     └─ load-balanced model    │
                                   │        -> rank assignment     │
                                   └──────────────────────────────┘
                                                  │ git push
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │ Stage 2 — Per-rank training  │
                                   ├──────────────────────────────┤
  Rank 0 (RTX 5060)                │                              │
   ├ git pull                       │                              │
   ├ read manifest.assignments[0]   │                              │
   ├ preprocess: parallel pipeline  │                              │
   │   (clean -> NFC -> Roman ->    │                              │
   │    transliterate)              │                              │
   ├ train GPU models sequentially  │                              │
   ├ train CPU models in parallel   │  Same code on rank 1, only   │
   ├ profile VRAM/throughput        │  FUG_RANK differs. Each rank │
   ├ write metrics + plots          │  works independently and     │
   └ git push results/rank_0/       │  pushes its artifacts.       │
                                   └──────────────────────────────┘
                                                  │ both ranks push
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │ Stage 3 — Aggregation (CI)   │
                                   ├──────────────────────────────┤
                                   │ - github action triggers on  │
                                   │   results/rank_*/metrics/**  │
                                   │ - leaderboard, timing,       │
                                   │   speedup plots              │
                                   │ - (locally) weighted-vote &  │
                                   │   weighted-softmax ensemble  │
                                   └──────────────────────────────┘
                                                  │
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │ Stage 4 — Inference + XAI    │
                                   ├──────────────────────────────┤
                                   │ - dynamic batching           │
                                   │ - CUDA-stream double buffer  │
                                   │ - ThreadPoolExecutor ens.    │
                                   │ - attention + LIME -> HTML   │
                                   └──────────────────────────────┘
```

## Mode B — FedAvg of LoRA adapters

```
  for round t = 1..T:
    rank R: git pull
    rank R: load global adapter from round_{t-1} (round 1 starts from base)
    rank R: train E local epochs on rank-{R}.parquet
    rank R: push round_{t}/rank_{R}/adapter*
    rank R: wait until peers have published
    rank 0: average all rank_*/adapter* -> round_{t}/global/
    rank 0: push round_{t}/global/
    others: wait for round_{t}/global/, loop
```

Average is element-wise mean over the LoRA adapter tensors only — base weights
are frozen and bit-identical on both nodes, so they don't need to flow over the
wire. Total bandwidth per round ≈ 2 × adapter_size (~2–4 MB), trivial even on
asymmetric home connections.
