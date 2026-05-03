"""Lightweight sanity tests that don't need a GPU or HF model downloads.

Run with:  python -m pytest tests/  (or just `python tests/test_smoke.py`)
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


def test_imports():
    for m in [
        "fasturduguard",
        "fasturduguard.utils",
        "fasturduguard.data.unify",
        "fasturduguard.data.relabel",
        "fasturduguard.data.shards",
        "fasturduguard.preprocess.normalize",
        "fasturduguard.preprocess.clean",
        "fasturduguard.preprocess.roman_urdu",
        "fasturduguard.preprocess.pipeline",
        "fasturduguard.coord.manifest",
        "fasturduguard.coord.git_helper",
        "fasturduguard.agg.ensemble",
        "fasturduguard.agg.aggregate",
        "fasturduguard.agg.plots",
        "fasturduguard.infer.pipeline",
        "fasturduguard.explain.visualize",
    ]:
        importlib.import_module(m)


def test_normalize_and_clean():
    from fasturduguard.preprocess.clean import clean_text
    from fasturduguard.preprocess.normalize import normalize_urdu

    txt = "  Hello\u200bWorld!! visit https://example.com or @joe #news\nچار  افراد  "
    cleaned = clean_text(txt)
    assert "https://" not in cleaned
    assert "@joe" not in cleaned
    assert "news" in cleaned
    assert "\u200b" not in cleaned

    nfc = normalize_urdu("کۂ")
    assert isinstance(nfc, str) and len(nfc) > 0


def test_roman_urdu_detection_and_transliteration():
    from fasturduguard.preprocess.roman_urdu import (
        is_roman_urdu_token, roman_urdu_fraction, transliterate_text,
    )

    assert is_roman_urdu_token("aaj")
    assert is_roman_urdu_token("kuch")
    assert not is_roman_urdu_token("the")
    assert roman_urdu_fraction("aaj kal bohot mushkil hai") > 0.5

    out = transliterate_text("aaj kal bohot mushkil hai")
    assert any(ord(c) >= 0x0600 for c in out), "Should contain Urdu/Arabic codepoints"


def test_ensemble_voting():
    from fasturduguard.agg.ensemble import weighted_softmax, weighted_vote

    p1 = np.array([0, 0, 1, 1])
    p2 = np.array([0, 1, 1, 1])
    p3 = np.array([0, 1, 0, 1])
    out = weighted_vote([p1, p2, p3], [1.0, 1.0, 1.0], n_classes=2)
    assert out.tolist() == [0, 1, 1, 1]

    sp = weighted_softmax([
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        np.array([[0.6, 0.4], [0.6, 0.4]]),
    ], [1.0, 1.0])
    # blended = [[0.75, 0.25], [0.35, 0.65]] -> argmax = [0, 1]
    assert sp.tolist() == [0, 1]


def test_manifest_load_balance():
    from fasturduguard.coord.manifest import build_default_manifest

    m = build_default_manifest(2)
    assigns = m["mode_a"]["assignments"]
    flat = [n for v in assigns.values() for n in v]
    assert sorted(flat) == sorted(set(flat))   # no duplicates across ranks


def test_dynamic_batching():
    from fasturduguard.infer.pipeline import length_bucket_batches

    texts = [f"t{i}" for i in range(10)]
    lengths = [10, 100, 50, 30, 200, 25, 500, 60, 80, 5]
    batches = list(length_bucket_batches(texts, lengths, token_budget=600))
    assert len(batches) >= 1
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(10))


if __name__ == "__main__":
    test_imports()
    test_normalize_and_clean()
    test_roman_urdu_detection_and_transliteration()
    test_ensemble_voting()
    test_manifest_load_balance()
    test_dynamic_batching()
    print("OK")
