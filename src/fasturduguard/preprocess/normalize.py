"""Unicode + diacritic normalization for Urdu."""
from __future__ import annotations

import re
import unicodedata

# Common Urdu diacritics (harakat)
_HARAKAT = re.compile("[\u064B-\u0652\u0670\u06D6-\u06ED]")
# Multiple-form characters folded to canonical
_FOLDS = {
    "\u0643": "\u06A9",   # Arabic kaf  -> Urdu kaf
    "\u064A": "\u06CC",   # Arabic yeh  -> Urdu yeh
    "\u0649": "\u06CC",   # alef maksura -> Urdu yeh
    "\u06C0": "\u06C1",   # heh w/ hamza -> Urdu heh
    "\u0629": "\u06C1",   # ta marbuta -> Urdu heh
}


def fold_chars(s: str) -> str:
    return "".join(_FOLDS.get(c, c) for c in s)


def normalize_urdu(text: str, *, drop_harakat: bool = True) -> str:
    """NFC + char folding + (optional) diacritic stripping + whitespace squish."""
    if not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFC", text)
    s = fold_chars(s)
    if drop_harakat:
        s = _HARAKAT.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
