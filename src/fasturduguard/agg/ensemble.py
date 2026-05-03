"""Weighted majority-vote ensemble across heterogeneous models.

Two operating modes:
  * `weighted_vote` — argmax over sum_m w_m * 1[f_m(x) = c].
  * `weighted_softmax` — argmax over sum_m w_m * softmax(z_m(x))[c].

Weights are usually the validation macro-F1 of each model.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def weighted_vote(preds_per_model: Sequence[np.ndarray], weights: Sequence[float], n_classes: int) -> np.ndarray:
    """preds_per_model: list of (N,) int arrays. weights: list of M floats."""
    N = len(preds_per_model[0])
    assert all(len(p) == N for p in preds_per_model)
    score = np.zeros((N, n_classes), dtype=np.float64)
    for w, p in zip(weights, preds_per_model):
        for c in range(n_classes):
            score[:, c] += w * (p == c)
    return np.argmax(score, axis=1)


def weighted_softmax(probs_per_model: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """probs_per_model: list of (N, C) float arrays."""
    stack = np.stack(probs_per_model, axis=0)               # (M, N, C)
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 1, 1)
    blended = (w * stack).sum(axis=0) / max(np.sum(weights), 1e-9)
    return np.argmax(blended, axis=1)
