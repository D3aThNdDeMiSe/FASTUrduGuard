"""Push this rank's artifacts back to the shared coordinator repo.

By default stages every file under results/rank_<R>/checkpoints/<model>/ (full
weights + tokenizers + Trainer shards) using `git add -f`, because those paths
are gitignored to avoid accidental giant commits — but this script is the
intentional publisher.

For bandwidth-constrained pushes, use --adapters-only (LoRA tensors only).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fasturduguard.coord import git_helper
from fasturduguard.utils import get_rank, rank_results_dir

log = logging.getLogger("fug.coord.push")

_GH_WARN_BYTES = 95 * 1024 * 1024  # GitHub blocks non-LFS blobs > 100 MiB


def _warn_large_checkpoints(ck_dirs: list[Path]) -> None:
    for ck in ck_dirs:
        for f in ck.rglob("*"):
            if not f.is_file():
                continue
            try:
                sz = f.stat().st_size
            except OSError:
                continue
            if sz > _GH_WARN_BYTES:
                log.warning(
                    "Large file may be rejected by GitHub (>100 MiB without Git LFS): %s (%d MiB)",
                    f,
                    sz // (1024 * 1024),
                )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--message", default=None, help="Git commit message suffix")
    p.add_argument(
        "--adapters-only",
        action="store_true",
        help="Legacy mode: only adapter_model.safetensors + adapter_config.json per model",
    )
    args = p.parse_args()

    rank = get_rank()
    rd = rank_results_dir(rank)

    msg = args.message or (
        f"rank {rank}: metrics + plots + profile + "
        f'{"LoRA adapters only" if args.adapters_only else "full checkpoints"}'
    )

    paths: list[Path] = []
    for sub in ("metrics", "plots", "profile"):
        subp = rd / sub
        if subp.exists():
            paths.append(subp)

    ck_root = rd / "checkpoints"
    if not ck_root.is_dir():
        log.warning("No checkpoints dir: %s", ck_root)
    else:
        ck_dirs = sorted(p for p in ck_root.iterdir() if p.is_dir())
        if not ck_dirs:
            log.warning("No model checkpoint folders under %s", ck_root)

        if args.adapters_only:
            for ck in ck_dirs:
                for name in ("adapter_model.safetensors", "adapter_config.json"):
                    f = ck / name
                    if f.is_file():
                        paths.append(f)
        else:
            # Entire HF Trainer output tree (model.safetensors, config, tokenizer,
            # checkpoint-* subdirs, LoRA files, training_args.bin, etc.)
            for ck in ck_dirs:
                paths.append(ck)
            _warn_large_checkpoints(ck_dirs)

    git_helper.pull()
    ok = git_helper.publish(paths, msg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
