"""Manifest = the work assignment table.

Schema (coordinator/manifest.json):

{
  "world_size": 2,
  "mode_a": {
      "type": "model_parallel",
      "assignments": {
          "0": ["xlm-roberta-base", "distilbert-multi", "xlnet-base"],
          "1": ["deberta-v3-base", "mbert", "roberta-base"]
      }
  },
  "mode_b": {
      "type": "fedavg",
      "model": "xlm-roberta-base",
      "rounds": 5,
      "shard_file": "data/processed/shards/rank_{rank}.parquet"
  }
}

`build_default_manifest` assigns models to ranks by alternating sorted-by-size,
which gives both ranks roughly equal compute load (small + big each).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fasturduguard.utils import (
    coordinator_dir, load_yaml, save_json, setup_logging,
)

log = logging.getLogger("fug.coord.manifest")


def build_default_manifest(world_size: int = 2) -> dict:
    cfg = load_yaml("models.yaml")
    models = sorted(cfg["models"], key=lambda m: m["size_mb"], reverse=True)

    # Greedy load-balance: keep two buckets; assign each model to the lighter bucket.
    buckets: list[list[str]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for m in models:
        i = loads.index(min(loads))
        buckets[i].append(m["name"])
        loads[i] += m["size_mb"]

    return {
        "world_size": world_size,
        "mode_a": {
            "type": "model_parallel",
            "assignments": {str(i): bs for i, bs in enumerate(buckets)},
            "estimated_load_mb": loads,
        },
        "mode_b": {
            "type": "fedavg",
            "model": "xlm-roberta-base",
            "rounds": 5,
            "local_epochs_per_round": 1,
            "shard_file": "data/processed/shards/rank_{rank}.parquet",
        },
    }


def get_assignment(manifest: dict, rank: int) -> list[str]:
    return manifest["mode_a"]["assignments"][str(rank)]


def main() -> None:
    setup_logging(prefix="manifest")
    p = argparse.ArgumentParser()
    p.add_argument("--world_size", type=int, default=2)
    p.add_argument("--out", type=Path, default=coordinator_dir() / "manifest.json")
    args = p.parse_args()

    m = build_default_manifest(args.world_size)
    save_json(args.out, m)
    log.info("Wrote %s :: %s", args.out, m["mode_a"]["assignments"])


if __name__ == "__main__":
    main()
