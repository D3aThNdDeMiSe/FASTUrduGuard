"""Shared helpers: config loading, paths, logging, seeding."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def repo_root() -> Path:
    """Return the FastUrduGuard root regardless of CWD."""
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return repo_root() / "configs"


def data_dir() -> Path:
    return repo_root() / "data"


def results_dir() -> Path:
    return repo_root() / "results"


def coordinator_dir() -> Path:
    return repo_root() / "coordinator"


def rank_results_dir(rank: int) -> Path:
    p = results_dir() / f"rank_{rank}"
    (p / "metrics").mkdir(parents=True, exist_ok=True)
    (p / "checkpoints").mkdir(parents=True, exist_ok=True)
    (p / "plots").mkdir(parents=True, exist_ok=True)
    (p / "profile").mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_yaml(name: str) -> dict[str, Any]:
    p = configs_dir() / name
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO, prefix: str = "") -> logging.Logger:
    fmt = f"[{prefix}] %(asctime)s %(levelname)s %(name)s :: %(message)s" if prefix else \
          "%(asctime)s %(levelname)s %(name)s :: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout, force=True)
    return logging.getLogger("fug")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # local import: keep utils torch-free for non-train scripts
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------
def get_rank() -> int:
    """Resolve the active rank from FUG_RANK (preferred) or RANK env vars."""
    for v in ("FUG_RANK", "RANK"):
        if v in os.environ:
            try:
                return int(os.environ[v])
            except ValueError:
                pass
    return 0
