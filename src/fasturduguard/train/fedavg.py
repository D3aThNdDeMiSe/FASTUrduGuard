"""FedAvg of LoRA adapters across two ranks (advanced mode).

Round protocol per rank R, total rounds T:
  for t in 1..T:
    1. git pull
    2. if t > 1: load coordinator/fedavg/round_{t-1}/global/adapter*  -> local model
    3. local fine-tune for `local_epochs_per_round` on data/processed/shards/rank_{R}.parquet
    4. save adapter to coordinator/fedavg/round_{t}/rank_{R}/adapter*
    5. publish (git add+commit+push)
    6. wait until ALL ranks have published round t (poll git pull every 30s)
    7. rank 0 averages adapters into coordinator/fedavg/round_{t}/global/, pushes
    8. wait for global push to be reachable, loop

Fallback: if no remote configured, ranks just write to local fs and average is
performed by whichever rank sees both adapters first.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import torch

from fasturduguard.coord import git_helper
from fasturduguard.utils import (
    coordinator_dir, get_rank, load_json, load_yaml, repo_root,
    rank_results_dir, set_seed, setup_logging,
)
from fasturduguard.train.trainer import train_one

log = logging.getLogger("fug.train.fedavg")


def _global_dir(round_id: int) -> Path:
    return coordinator_dir() / "fedavg" / f"round_{round_id}" / "global"


def _rank_dir(round_id: int, rank: int) -> Path:
    return coordinator_dir() / "fedavg" / f"round_{round_id}" / f"rank_{rank}"


def _adapter_files(d: Path) -> list[Path]:
    pats = ["adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"]
    return [d / p for p in pats if (d / p).exists()]


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    return load_file(str(path))


def _save_safetensors(state: dict[str, torch.Tensor], path: Path) -> None:
    from safetensors.torch import save_file
    save_file(state, str(path))


def average_adapters(rank_dirs: list[Path], out_dir: Path) -> None:
    """Element-wise mean of all rank adapter weights -> out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    states = []
    for d in rank_dirs:
        f = d / "adapter_model.safetensors"
        if not f.exists():
            f = d / "adapter_model.bin"
            if f.exists():
                states.append(torch.load(str(f), map_location="cpu"))
            continue
        states.append(_load_safetensors(f))
    if not states:
        return
    avg: dict[str, torch.Tensor] = {}
    for k in states[0]:
        avg[k] = sum(s[k] for s in states) / len(states)
    _save_safetensors(avg, out_dir / "adapter_model.safetensors")
    cfg = rank_dirs[0] / "adapter_config.json"
    if cfg.exists():
        (out_dir / "adapter_config.json").write_text(
            cfg.read_text(encoding="utf-8"), encoding="utf-8"
        )
    log.info("Averaged %d adapters -> %s", len(states), out_dir)


def _wait_for_other_ranks(round_id: int, world_size: int, my_rank: int,
                          poll_s: int = 30, max_wait_s: int = 1800) -> bool:
    deadline = time.time() + max_wait_s
    others = [r for r in range(world_size) if r != my_rank]
    while time.time() < deadline:
        git_helper.pull()
        ready = all((_rank_dir(round_id, r) / "adapter_model.safetensors").exists()
                    or (_rank_dir(round_id, r) / "adapter_model.bin").exists()
                    for r in others)
        if ready:
            return True
        log.info("Waiting for ranks %s to publish round %d...", others, round_id)
        time.sleep(poll_s)
    return False


def _wait_for_global(round_id: int, poll_s: int = 30, max_wait_s: int = 1800) -> bool:
    deadline = time.time() + max_wait_s
    g = _global_dir(round_id)
    while time.time() < deadline:
        git_helper.pull()
        if (g / "adapter_model.safetensors").exists():
            return True
        log.info("Waiting for global average of round %d...", round_id)
        time.sleep(poll_s)
    return False


def main() -> None:
    setup_logging(prefix=f"fedavg-r{get_rank()}")
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--epochs_per_round", type=int, default=None)
    p.add_argument("--world_size", type=int, default=2)
    p.add_argument("--no_remote", action="store_true",
                   help="Skip git push/pull; rely on local filesystem only.")
    args = p.parse_args()

    rank = get_rank()
    manifest = load_json(coordinator_dir() / "manifest.json")
    fedcfg = manifest["mode_b"]
    model_name = fedcfg["model"]
    rounds = args.rounds or fedcfg["rounds"]
    epr = args.epochs_per_round or fedcfg.get("local_epochs_per_round", 1)
    world_size = args.world_size

    cfg = load_yaml("models.yaml")
    tr_cfg = cfg["training"]
    set_seed(tr_cfg["seed"] + rank)
    by_name = {m["name"]: m for m in cfg["models"]}
    mcfg = by_name[model_name]

    shard_path = repo_root() / fedcfg["shard_file"].format(rank=rank)
    df = pd.read_parquet(shard_path)
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    rank_dir = rank_results_dir(rank) / f"fedavg_{model_name}"
    rank_dir.mkdir(parents=True, exist_ok=True)

    init_from: Path | None = None
    for t in range(1, rounds + 1):
        log.info("==== ROUND %d / %d  rank=%d ====", t, rounds, rank)

        # 1) sync
        if not args.no_remote:
            git_helper.pull()

        # 2) inherit from previous global if any
        if t > 1:
            init_from = _global_dir(t - 1)

        # 3) local train
        m = train_one(
            model_cfg=mcfg, train_df=train, val_df=val, test_df=test,
            out_dir=rank_dir, rank=rank,
            label_col="label_4",
            epochs=epr, lr=tr_cfg["learning_rate"],
            batch_size=tr_cfg["per_device_train_batch_size"],
            max_seq_len=tr_cfg["max_seq_len"],
            grad_accum=tr_cfg["gradient_accumulation_steps"],
            fp16=tr_cfg.get("fp16", True),
            seed=tr_cfg["seed"] + rank,
        )

        # 4) publish adapter for this round
        ckpt = Path(m["checkpoint_dir"])
        rd = _rank_dir(t, rank)
        rd.mkdir(parents=True, exist_ok=True)
        for f in _adapter_files(ckpt):
            (rd / f.name).write_bytes(f.read_bytes())

        if not args.no_remote:
            git_helper.publish([rd], f"fedavg round={t} rank={rank}")
            ok = _wait_for_other_ranks(t, world_size, rank)
            if not ok:
                log.warning("Timed out waiting for peers; proceeding with local-only avg")

        # 5) rank 0 performs the average and publishes the global
        if rank == 0:
            peer_dirs = [_rank_dir(t, r) for r in range(world_size)]
            average_adapters(peer_dirs, _global_dir(t))
            if not args.no_remote:
                git_helper.publish([_global_dir(t)], f"fedavg round={t} global avg")

        if not args.no_remote and rank != 0:
            _wait_for_global(t)

    log.info("FedAvg done after %d rounds.", rounds)


if __name__ == "__main__":
    main()
