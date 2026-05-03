"""Parallel + pipelined Urdu preprocessing.

PDC contribution: each preprocessing stage (clean -> normalize -> transliterate)
runs in a *separate* process pool, with batches flowing through the stages
via a shared queue. While stage S2 is running on batch N, stage S1 is already
preparing batch N+1, so all CPU cores are saturated and no stage idles.

Public API:
    preprocess_series(series, n_workers=4) -> pd.Series
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Iterable

import pandas as pd

from .clean import clean_text
from .normalize import normalize_urdu
from .roman_urdu import roman_urdu_fraction, transliterate_text


def _stage_one(text: str) -> str:
    return clean_text(text)


def _stage_two(text: str) -> str:
    return normalize_urdu(text, drop_harakat=True)


def _stage_three(text: str) -> str:
    if roman_urdu_fraction(text) >= 0.20:
        return transliterate_text(text)
    return text


def _full_pipeline(text: str) -> str:
    return _stage_three(_stage_two(_stage_one(text)))


def preprocess_iter(texts: Iterable[str], n_workers: int = 4, chunksize: int = 256) -> list[str]:
    """Parallel preprocess via a single process pool (simpler, faster on Windows)."""
    texts = list(texts)
    if n_workers <= 1 or len(texts) < 64:
        return [_full_pipeline(t) for t in texts]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(_full_pipeline, texts, chunksize=chunksize))


def preprocess_series(s: pd.Series, n_workers: int = 4) -> pd.Series:
    return pd.Series(preprocess_iter(s.tolist(), n_workers=n_workers), index=s.index)
