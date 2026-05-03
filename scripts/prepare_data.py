"""End-to-end data preparation: unify -> relabel -> shard.

Run on rank 0 once, then commit data/processed/ + push.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(mod: str, extra: list[str]) -> None:
    cmd = [sys.executable, "-m", mod, *extra]
    print(">>", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise SystemExit(f"step {mod} failed rc={rc}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, type=Path,
                   help="Path to extracted dataset roots (Urdu-Large-language-dataset-main, ...)")
    p.add_argument("--mode", default="rule", choices=("rule", "llm", "rule+llm"))
    p.add_argument("--num_ranks", type=int, default=2)
    p.add_argument("--strategy", default="stratified_random",
                   choices=("stratified_random", "by_source"))
    args = p.parse_args()

    _run("fasturduguard.data.unify", ["--raw", str(args.raw)])
    _run("fasturduguard.data.relabel", ["--mode", args.mode])
    _run("fasturduguard.data.shards",
         ["--num_ranks", str(args.num_ranks), "--strategy", args.strategy])
    _run("fasturduguard.coord.manifest", ["--world_size", str(args.num_ranks)])
    print("OK")


if __name__ == "__main__":
    main()
