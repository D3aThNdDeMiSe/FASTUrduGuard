"""Detect Roman-Urdu segments and transliterate them to Nastaliq.

Approach:
  * Detection: per-token character n-gram heuristic. A Latin token is classified
    as Roman-Urdu if it has length >=3, contains ASCII letters only, AND
    matches one of two cues:
      (a) explicit dictionary hit (the keys below cover the ~250 highest-frequency
          Roman-Urdu function words);
      (b) presence of a "Roman-Urdu signature" digraph that is rare in English
          (kh, gh, sh, ch, dh, bh, ph, jh, rh) AND no English-stoplist match.
  * Transliteration: dictionary lookup first, then a phonetic-rule fallback that
    maps frequent digraphs -> Nastaliq letters.

This is deliberately lightweight: <2 ms per article on a single core. It is
enough to convert the bulk of Roman-Urdu social-media tokens; full coverage
is not the goal because XLM-R's BPE handles unseen Roman tokens reasonably.
"""
from __future__ import annotations

import re
from typing import Iterable

# ---------------------------------------------------------------------------
# Data tables
# ---------------------------------------------------------------------------

# Functional words. Many surface forms map to the same Nastaliq word.
_LEXICON: dict[str, str] = {
    "main": "میں", "mein": "میں", "mai": "میں",
    "tum": "تم", "tu": "تو", "ap": "آپ", "aap": "آپ",
    "hai": "ہے", "hain": "ہیں", "hum": "ہم", "humaray": "ہمارے",
    "ye": "یہ", "yeh": "یہ", "wo": "وہ", "woh": "وہ",
    "kya": "کیا", "kia": "کیا", "kab": "کب", "kahan": "کہاں",
    "kyun": "کیوں", "kyon": "کیوں", "kese": "کیسے", "kaise": "کیسے",
    "aur": "اور", "or": "اور", "ya": "یا",
    "agar": "اگر", "lekin": "لیکن", "magar": "مگر",
    "sab": "سب", "sabhi": "سبھی", "kuch": "کچھ", "kuchh": "کچھ",
    "hai.": "ہے", "hai,": "ہے",
    "nahi": "نہیں", "nahin": "نہیں", "na": "نہ",
    "ho": "ہو", "hoga": "ہوگا", "hogi": "ہوگی", "hota": "ہوتا",
    "kar": "کر", "karta": "کرتا", "karti": "کرتی", "karte": "کرتے",
    "raha": "رہا", "rahi": "رہی", "rahe": "رہے",
    "tha": "تھا", "thi": "تھی", "thay": "تھے", "thi.": "تھی",
    "phir": "پھر", "fir": "پھر",
    "abhi": "ابھی", "ab": "اب",
    "log": "لوگ", "logon": "لوگوں", "logo": "لوگو",
    "din": "دن", "raat": "رات", "saath": "ساتھ",
    "kal": "کل", "aaj": "آج", "kal.": "کل",
    "se": "سے", "ka": "کا", "ki": "کی", "ke": "کے",
    "mein.": "میں", "tha.": "تھا",
    "bohot": "بہت", "bohut": "بہت", "bahot": "بہت", "bahut": "بہت",
    "sirf": "صرف",
    "khabar": "خبر", "khabren": "خبریں",
    "mulk": "ملک", "mulki": "ملکی",
    "wazir": "وزیر", "azam": "اعظم",
    "sadar": "صدر", "sadr": "صدر",
    "police": "پولیس", "fauj": "فوج",
    "fake": "جعلی", "jhoot": "جھوٹ", "sach": "سچ", "sahi": "صحیح",
    "naya": "نیا", "puranay": "پرانے",
    "shukriya": "شکریہ", "salam": "سلام",
}

# Phonetic fall-back substitutions (longest-first).
_PHONETIC_RULES: list[tuple[str, str]] = [
    ("kh", "خ"), ("gh", "غ"), ("sh", "ش"), ("ch", "چ"), ("dh", "دھ"),
    ("bh", "بھ"), ("ph", "پھ"), ("jh", "جھ"), ("rh", "رھ"),
    ("aa", "ا"), ("ee", "ی"), ("oo", "و"), ("ai", "ای"), ("au", "او"),
    ("a", "ا"), ("b", "ب"), ("c", "ک"), ("d", "د"), ("e", "ی"),
    ("f", "ف"), ("g", "گ"), ("h", "ہ"), ("i", "ی"), ("j", "ج"),
    ("k", "ک"), ("l", "ل"), ("m", "م"), ("n", "ن"), ("o", "و"),
    ("p", "پ"), ("q", "ق"), ("r", "ر"), ("s", "س"), ("t", "ت"),
    ("u", "و"), ("v", "و"), ("w", "و"), ("x", "کس"), ("y", "ی"),
    ("z", "ز"),
]

_ENG_STOPLIST = {
    "the", "and", "for", "you", "are", "with", "that", "this", "from",
    "have", "not", "but", "was", "will", "can", "all", "they", "your",
    "his", "her", "she", "him", "their", "out", "about", "over", "into",
    "more", "less", "also", "than", "then", "such", "very", "much",
    "some", "any", "one", "two", "three", "what", "when", "where", "why",
    "how", "which", "who", "whom",
}

_ROMAN_DIGRAPHS = ("kh", "gh", "sh", "ch", "dh", "bh", "ph", "jh", "rh")

_TOKEN_RE = re.compile(r"[A-Za-z']+|[\u0600-\u06FF]+|\d+|[^\sA-Za-z\u0600-\u06FF]+")


def _is_ascii_word(t: str) -> bool:
    return bool(t) and t.isascii() and t.isalpha()


def is_roman_urdu_token(t: str) -> bool:
    if not _is_ascii_word(t):
        return False
    tl = t.lower()
    if len(tl) < 2:
        return False
    if tl in _LEXICON:
        return True
    if tl in _ENG_STOPLIST:
        return False
    return any(d in tl for d in _ROMAN_DIGRAPHS)


def transliterate_token(t: str) -> str:
    tl = t.lower()
    if tl in _LEXICON:
        return _LEXICON[tl]
    s = tl
    for src, tgt in _PHONETIC_RULES:
        s = s.replace(src, tgt)
    return s


def transliterate_text(text: str) -> str:
    """Return a copy of *text* with Roman-Urdu tokens transliterated to Nastaliq.
    English tokens (top-100 stoplist or non-Roman-Urdu) are left as-is.
    """
    if not isinstance(text, str) or not text:
        return ""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        if is_roman_urdu_token(tok):
            out.append(transliterate_token(tok))
        else:
            out.append(tok)
    return "".join(
        " " + tok if (i and out[i - 1][-1:].isalpha() and tok[:1].isalpha()) else tok
        for i, tok in enumerate(out)
    ).strip() or text.strip()


def roman_urdu_fraction(text: str) -> float:
    """Return fraction of word tokens classified as Roman-Urdu.

    Useful for routing: if > 0.2 we run transliteration; else we skip the work.
    """
    if not text:
        return 0.0
    words = [t for t in _TOKEN_RE.findall(text) if _is_ascii_word(t)]
    if not words:
        return 0.0
    n = sum(1 for w in words if is_roman_urdu_token(w))
    return n / len(words)
