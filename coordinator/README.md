# coordinator/

Shared cross-node state. Lives in git so two nodes on different ISPs can sync
through the GitHub remote without ever talking directly.

```
coordinator/
├── manifest.json        # Mode-A model -> rank assignment + Mode-B fedavg config
└── fedavg/              # Created on first FedAvg round
    └── round_<t>/
        ├── rank_0/      # adapter pushed by rank 0 in round t
        ├── rank_1/      # adapter pushed by rank 1 in round t
        └── global/      # element-wise averaged adapter, written by rank 0
```

`manifest.json` is committed by the data-prep step
(`python scripts/prepare_data.py ...`) and rarely changes. Edit it by hand if
you want to manually rebalance which models train on which node.
