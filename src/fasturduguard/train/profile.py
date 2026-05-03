"""Lightweight VRAM/CPU/throughput profiling.

Best-effort: runs even if pynvml isn't available.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import psutil


def gpu_info() -> dict:
    """Return {available, name, total_mb, used_mb, util%}."""
    try:
        import pynvml
    except Exception:
        return {"available": False}
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        return {
            "available": True,
            "name": name,
            "total_mb": int(mem.total // (1024 * 1024)),
            "used_mb": int(mem.used // (1024 * 1024)),
            "free_mb": int(mem.free // (1024 * 1024)),
            "gpu_util_pct": int(util.gpu),
            "mem_util_pct": int(util.memory),
        }
    except Exception:
        return {"available": False}


def cpu_info() -> dict:
    return {
        "cpu_pct": psutil.cpu_percent(interval=0.05),
        "ram_used_mb": int(psutil.virtual_memory().used / (1024 * 1024)),
        "ram_total_mb": int(psutil.virtual_memory().total / (1024 * 1024)),
        "n_cpus": psutil.cpu_count(logical=True),
    }


class StopWatch:
    """Walltime + (optional) GPU peak memory tracker."""
    def __init__(self) -> None:
        self.t0 = 0.0
        self.elapsed_s = 0.0
        self.peak_vram_mb = 0

    def __enter__(self):
        self.t0 = time.perf_counter()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        return self

    def __exit__(self, *a):
        self.elapsed_s = time.perf_counter() - self.t0
        try:
            import torch
            if torch.cuda.is_available():
                self.peak_vram_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
        except Exception:
            pass


def write_profile(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
