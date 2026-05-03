"""Render attention + LIME scores as a single self-contained HTML highlight."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Sequence


def _norm(xs: Sequence[float]) -> list[float]:
    if not xs:
        return []
    a = min(xs)
    b = max(xs)
    if b - a < 1e-9:
        return [0.5] * len(xs)
    return [(x - a) / (b - a) for x in xs]


def _bg(score: float, palette: tuple[str, str] = ("#fff5e6", "#f08c00")) -> str:
    """Score in [0,1] -> CSS background color (interpolated)."""
    s = max(0.0, min(1.0, score))
    def _c(h):  # "#rrggbb" -> (r,g,b)
        return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    a, b = _c(palette[0]), _c(palette[1])
    rgb = tuple(int(a[i] + (b[i] - a[i]) * s) for i in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def render_html(*, predicted: int, class_names: list[str],
                sentences: list[str], sent_scores: list[float],
                tokens: list[str] | None = None,
                tok_scores: list[float] | None = None,
                title: str = "FastUrduGuard explanation") -> str:
    sn = _norm(sent_scores)

    rows = []
    for sent, raw, n in zip(sentences, sent_scores, sn):
        rows.append(
            f'<div style="background:{_bg(n)};padding:6px 10px;margin:4px 0;'
            f'border-radius:4px;direction:rtl"><span style="color:#666;font-size:0.8em">'
            f'LIME={raw:+.3f}</span><br>{html.escape(sent)}</div>'
        )

    tok_html = ""
    if tokens and tok_scores:
        tn = _norm(tok_scores)
        spans = []
        for t, n in zip(tokens, tn):
            spans.append(f'<span style="background:{_bg(n,("#eaf4fb","#226faf"))};'
                         f'padding:1px 3px;margin:1px;border-radius:3px">'
                         f'{html.escape(t)}</span>')
        tok_html = f'<div style="direction:rtl;line-height:2.0">{"".join(spans)}</div>'

    pred = class_names[predicted] if 0 <= predicted < len(class_names) else f"class_{predicted}"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:Segoe UI,Arial;max-width:900px;margin:24px auto;padding:0 16px}}
h1{{font-size:1.4em}} h2{{font-size:1.1em;margin-top:1.2em}}
.pill{{display:inline-block;background:#226f54;color:white;padding:3px 10px;border-radius:12px}}
</style></head>
<body>
<h1>{html.escape(title)}</h1>
<p>Predicted class: <span class="pill">{html.escape(pred)}</span></p>
<h2>Sentence-level (LIME)</h2>
{''.join(rows)}
<h2>Token-level (final-layer attention from [CLS])</h2>
{tok_html}
</body></html>"""


def write_html(out: Path, html_str: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_str, encoding="utf-8")
