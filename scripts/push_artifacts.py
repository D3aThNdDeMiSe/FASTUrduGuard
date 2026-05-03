"""Push this rank's artifacts back to the shared coordinator repo."""
from __future__ import annotations

import argparse
import sys

from fasturduguard.coord import git_helper
from fasturduguard.utils import get_rank, rank_results_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--message", default=None)
    args = p.parse_args()

    rank = get_rank()
    rd = rank_results_dir(rank)

    msg = args.message or f"rank {rank} artifacts: metrics + plots + adapters"
    paths = []
    for sub in ("metrics", "plots", "profile"):
        paths.append(rd / sub)
    # adapter checkpoints (small, .gitignore allow-listed)
    for ck in (rd / "checkpoints").glob("*"):
        for f in ("adapter_model.safetensors", "adapter_config.json"):
            p = ck / f
            if p.exists():
                paths.append(p)

    git_helper.pull()
    ok = git_helper.publish(paths, msg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
