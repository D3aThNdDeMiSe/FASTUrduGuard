"""Lightweight text cleaning: strip URLs, HTML, social-media noise."""
from __future__ import annotations

import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<[^>]+>")
_HANDLE = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")          # keep word, drop '#'
_REPEAT = re.compile(r"(.)\1{3,}")         # collapse 4+ repeats
_NON_LING = re.compile(
    "["
    "\u200b\u200c\u200d\u200e\u200f"      # zero-width / direction marks
    "\ufeff"
    "]"
)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text
    s = _URL.sub(" ", s)
    s = _HTML.sub(" ", s)
    s = _HANDLE.sub(" ", s)
    s = _HASHTAG.sub(r"\1", s)
    s = _NON_LING.sub("", s)
    s = _REPEAT.sub(r"\1\1\1", s)
    s = re.sub(r"[\t\r\n]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
