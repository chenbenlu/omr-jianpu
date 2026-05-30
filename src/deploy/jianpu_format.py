"""Beautified Jianpu rendering (presentation layer, deploy-only).

Real Jianpu is 2-D: octave dots sit above/below the degree digit and rhythm
underlines sit beneath it. This module renders it three ways off the *same*
per-symbol model:

- `jianpu_html`  — a CSS grid (one column per symbol, three rows) for Streamlit.
- `jianpu_svg`   — standalone SVG, for the CLI / file download.
- `pretty_jianpu`— 3-row monospace fallback (terminals can't render HTML/SVG).

Semantics (degree, accidental, octave, rhythm) are NOT re-derived here — each
symbol is rendered through the frozen `src.postproc` single-symbol path and the
resulting compact token is decomposed, so the numbers always match postproc.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from html import escape
from typing import Sequence

from src.data.generator import _token_to_duration
from src.postproc import TokenTuple
from src.postproc.jianpu import JianpuRenderConfig, tuples_to_jianpu

# Render each symbol on its own so the compact string is exactly one token,
# without headers. State (clef/key) is threaded by replaying structural tokens.
_CFG = JianpuRenderConfig(emit_header=False)
_STRUCTURAL = frozenset({"clef", "key_signature", "time_signature"})


@dataclass(frozen=True)
class JianpuCell:
    """One engraved column: a digit (with marks) or a barline."""

    body: str  # degree 1-7, "0" rest, "?" unknown, or "|" barline
    accidental: str = ""  # bb / b / # / ##
    dots_up: int = 0  # upper-octave dots
    dots_down: int = 0  # lower-octave dots
    underlines: int = 0  # rhythm beams (eighth=1, 16th=2, 32nd=3)
    tail: str = ""  # duration extension, e.g. " -", " - - -", " ."
    is_barline: bool = False
    is_note: bool = False  # a note/rest that can carry beams (not a barline)
    beat_group: int = -1  # cells sharing this id beam together (same beat)


def _parse_cell(compact: str) -> JianpuCell:
    """Decompose a one-symbol compact token into a structured cell.

    Grammar (see src/postproc/jianpu.py): {underline}{acc}{body}{octave}{tail}
    """
    s = compact
    n_underline = len(s) - len(s.lstrip("_"))
    s = s[n_underline:]

    acc = ""
    for cand in ("bb", "##", "b", "#"):
        if s.startswith(cand):
            acc = cand
            s = s[len(cand) :]
            break

    body = s[:1]
    s = s[1:]

    up = down = 0
    while s[:1] in ("'", ","):
        if s[0] == "'":
            up += 1
        else:
            down += 1
        s = s[1:]

    return JianpuCell(
        body=body,
        accidental=acc,
        dots_up=up,
        dots_down=down,
        underlines=n_underline,
        tail=s,  # remaining: " -", " - - -", " ." or ""
        is_note=True,
    )


def _beat_length(time_sig: str | None) -> float:
    """Quarter-note length of one beat for the running time signature."""
    if not time_sig:
        return 1.0
    try:
        from music21 import meter

        return float(meter.TimeSignature(time_sig).beatDuration.quarterLength)
    except Exception:
        return 1.0


def _cells(tuples: Sequence[TokenTuple]) -> list[JianpuCell]:
    """Per-symbol structured model with clef/key state and beam grouping.

    Each note/rest is assigned a `beat_group`: notes that fall in the same beat
    of the same measure share a group so their rhythm beams can be drawn as one
    continuous line. The bar position is accumulated from each symbol's duration
    and reset on barlines; the beat length follows the running time signature.
    """
    import math

    cells: list[JianpuCell] = []
    prefix: list[TokenTuple] = []
    beat_ql = 1.0
    pos = 0.0
    measure = 0
    group = 0
    last_key: tuple[int, int] | None = None

    for tup in tuples:
        type_tok = tup[0]
        if type_tok in _STRUCTURAL:
            prefix.append(tup)
            if type_tok == "time_signature" and tup[3] not in (None, "<UNK>"):
                beat_ql = _beat_length(tup[3])
        elif type_tok == "barline":
            cells.append(JianpuCell(body="|", is_barline=True))
            measure += 1
            pos = 0.0
            last_key = None
        elif type_tok in ("note", "rest"):
            try:
                compact = tuples_to_jianpu([*prefix, tup], _CFG)
                cell = _parse_cell(compact)
            except Exception:
                # A wild prediction (e.g. an out-of-range key signature) can make
                # postproc raise. Surface it as "?" so the demo stays up.
                cell = JianpuCell(body="?", is_note=True)
            beat_idx = int(math.floor(pos / beat_ql + 1e-9)) if beat_ql > 0 else 0
            key = (measure, beat_idx)
            if key != last_key:
                group += 1
                last_key = key
            cells.append(replace(cell, beat_group=group))
            pos += _cell_quarter_length(tup)
    return cells


def _cell_quarter_length(tup: TokenTuple) -> float:
    """Duration of a note/rest tuple in quarter notes (0 if unknown)."""
    rhythm = tup[2]
    if rhythm in (None, "<UNK>"):
        return 0.0
    try:
        return float(_token_to_duration(rhythm).quarterLength)
    except Exception:
        return 0.0


def _header_line(tuples: Sequence[TokenTuple]) -> str:
    header = tuples_to_jianpu(tuples, JianpuRenderConfig(emit_header=True))
    return header.split("\n", 1)[0] if "\n" in header else ""


# --- monospace (terminal fallback) ----------------------------------------


def _place(row: str, col: int, mark: str) -> str:
    end = col + len(mark)
    if end > len(row):
        row = row + " " * (end - len(row))
    return row[:col] + mark + row[end:]


def _cell_to_mono_rows(c: JianpuCell) -> tuple[str, str, str]:
    """One cell → (top, middle, bottom) monospace strings, digit-aligned."""
    if c.is_barline:
        return " ", "|", " "
    middle = f"{c.accidental}{c.body}{c.tail}"
    digit_col = len(c.accidental)
    blank = " " * len(middle)
    top = _place(blank, digit_col, "." * c.dots_up) if c.dots_up else blank
    marks = ("." * c.dots_down) + ("_" * c.underlines)
    bottom = _place(blank, digit_col, marks) if marks else blank
    return top, middle, bottom


def _decompose(compact: str) -> tuple[str, str, str]:
    """Back-compat: (top, middle, bottom) padded strings for one compact token."""
    return _cell_to_mono_rows(_parse_cell(compact))


def pretty_jianpu(tuples: Sequence[TokenTuple]) -> str:
    """3-row monospace Jianpu (terminal fallback)."""
    header_line = _header_line(tuples)
    top_cells: list[str] = []
    mid_cells: list[str] = []
    bot_cells: list[str] = []

    for c in _cells(tuples):
        top, mid, bot = _cell_to_mono_rows(c)
        w = max(len(top), len(mid), len(bot))
        top_cells.append(top.ljust(w))
        mid_cells.append(mid.ljust(w))
        bot_cells.append(bot.ljust(w))

    top = " ".join(top_cells).rstrip()
    mid = " ".join(mid_cells)
    bot = " ".join(bot_cells).rstrip()
    lines = [line for line in (header_line, top) if line]
    lines.append(mid)
    if bot:
        lines.append(bot)
    return "\n".join(lines)


def _beam_extends_right(cells: list[JianpuCell], i: int, layer: int) -> bool:
    """True if beam `layer` on cell i continues to cell i+1 (same beat group).

    A beam connects two adjacent notes only when both are notes in the same
    beat group and both carry at least `layer` underlines. This breaks the line
    across barlines, rests-with-no-beam, beat boundaries, and layer drops.
    """
    c = cells[i]
    if not (c.is_note and c.body != "0" and c.underlines >= layer):
        return False  # rests ("0") don't carry beams
    if i + 1 >= len(cells):
        return False
    n = cells[i + 1]
    return (
        n.is_note
        and n.body != "0"
        and n.beat_group == c.beat_group
        and n.underlines >= layer
    )


# --- HTML / CSS grid (Streamlit) -------------------------------------------


def jianpu_html(tuples: Sequence[TokenTuple]) -> str:
    """A CSS-grid Jianpu block. One grid column per symbol, three stacked rows.

    Octave dots align in the top/bottom rows directly over/under the digit;
    rhythm underlines render as a bottom border on the digit cell. Returns a
    self-contained HTML string for `st.markdown(..., unsafe_allow_html=True)`.
    """
    header = _header_line(tuples)
    cells = _cells(tuples)
    columns: list[str] = []
    for i, c in enumerate(cells):
        if c.is_barline:
            columns.append(
                '<div class="jp-col jp-bar">'
                '<div class="jp-top"></div><div class="jp-mid">|</div>'
                '<div class="jp-beams"></div><div class="jp-bot"></div>'
                "</div>"
            )
            continue
        top = "•" * c.dots_up
        bottom = "•" * c.dots_down
        digit_body = escape(f"{c.accidental}{c.body}")
        tail_text = escape(c.tail)
        # One bar per rhythm beam, explicitly placed by row so display order is
        # not at the mercy of inline-element whitespace handling. Jianpu order:
        # layer 1 (8th, the most-continuous line) sits closest to the digit;
        # extra beams (16th=layer 2, 32nd=layer 3) stack further below.
        # An extending beam gets `jp-x`, widening it across the column gap so
        # adjacent beams join into one continuous line.
        bar_items = []
        for layer in range(1, c.underlines + 1):
            extra = " jp-x" if _beam_extends_right(cells, i, layer) else ""
            bar_items.append(f'<span class="jp-beam jp-b{layer}{extra}"></span>')
        beams = "".join(bar_items)
        # The tail (" -", " - - -", " .") sits inside the digit row, absolutely
        # positioned just to the right of the digit so it shares the digit's
        # text baseline (not the bottom of the whole stack) while keeping the
        # digit's geometric center stable — octave dots and beams stay aligned.
        tail_html = f'<span class="jp-tail">{tail_text}</span>' if tail_text else ""
        # Per-column right padding grows with the tail so the next note doesn't
        # overlap the extension marks. Each character is ~0.7em wide at 1.4rem.
        n_tail_chars = sum(1 for ch in c.tail if ch != " ")
        tail_pad = f"padding-right:{0.28 + n_tail_chars * 0.7:.2f}rem;"
        columns.append(
            f'<div class="jp-col" style="{tail_pad}">'
            f'<div class="jp-top">{top}</div>'
            f'<div class="jp-mid">'
            f'<span class="jp-digit">{digit_body}</span>{tail_html}'
            "</div>"
            f'<div class="jp-beams">{beams}</div>'
            f'<div class="jp-bot">{bottom}</div>'
            "</div>"
        )

    header_html = f'<div class="jp-header">{escape(header)}</div>' if header else ""
    return (
        "<style>"
        ".jp-wrap{font-family:'DejaVu Sans',sans-serif;color:inherit;}"
        ".jp-header{font-size:0.9rem;opacity:0.8;margin-bottom:0.4rem;}"
        ".jp-row{display:flex;align-items:flex-end;flex-wrap:wrap;}"
        # Plain vertical-stack column: top dots / digit-row / beams / bot dots.
        # The digit-row holds the digit centered AND an absolutely-positioned
        # tail just to the right — the tail shares the digit's text baseline
        # without shifting the digit's geometric center, so octave dots and
        # beams (centered on the column) stay aligned to the digit.
        ".jp-col{display:flex;flex-direction:column;align-items:center;"
        "line-height:1.1;padding-left:0.28rem;padding-right:0.28rem;}"
        ".jp-top,.jp-bot{height:0.8rem;font-size:0.6rem;}"
        ".jp-mid{font-size:1.4rem;font-weight:600;padding:0 1px;"
        "position:relative;display:inline-block;}"
        ".jp-digit{display:inline-block;}"
        ".jp-tail{position:absolute;left:100%;top:0;white-space:pre;"
        "padding-left:0.1rem;}"
        # Beams live on a 3-row grid (layer 1 row 1, layer 2 row 2, layer 3 row 3),
        # so display order is fixed by `grid-row`, not by source/flex quirks.
        # Layer 1 (8th, longest) is row 1 = closest to the digit.
        ".jp-beams{display:grid;grid-template-rows:repeat(3,2.5px);"
        "row-gap:1.5px;width:0.9em;margin-top:1px;justify-items:start;}"
        ".jp-beam{height:1.5px;width:100%;background:currentColor;display:block;}"
        ".jp-b1{grid-row:1;}.jp-b2{grid-row:2;}.jp-b3{grid-row:3;}"
        # A continuing beam spans its own column plus the inter-column padding
        # (2×0.28rem) so it visually joins the next note's beam into one line.
        ".jp-beam.jp-x{width:calc(100% + 0.56rem + 0.9em);}"
        ".jp-bar .jp-mid{font-weight:300;opacity:0.6;}"
        "</style>"
        f'<div class="jp-wrap">{header_html}'
        f'<div class="jp-row">{"".join(columns)}</div></div>'
    )


# --- SVG (CLI / download) --------------------------------------------------


def jianpu_svg(tuples: Sequence[TokenTuple]) -> str:
    """Standalone SVG Jianpu — same layout as the HTML grid, file-portable."""
    cells = _cells(tuples)
    header = _header_line(tuples)

    col_w = 26
    pad_x = 12
    base_y = 60  # baseline of the digit row
    dot_r = 1.8
    parts: list[str] = []

    if header:
        parts.append(
            f'<text x="{pad_x}" y="20" font-size="13" fill="#444" '
            f'font-family="sans-serif">{escape(header)}</text>'
        )

    # Per-cell width grows with the tail (each "-" / "." in tail bumps the
    # cell right edge so the next note doesn't overlap the extension marks).
    widths = [col_w + len(c.tail.replace(" ", "")) * 11 for c in cells]
    centers: list[float] = []
    x = pad_x
    for w in widths:
        centers.append(x + col_w / 2)  # digit center stays at col_w/2 of cell
        x += w

    for i, c in enumerate(cells):
        cx = centers[i]
        if c.is_barline:
            parts.append(
                f'<line x1="{cx}" y1="{base_y - 18}" x2="{cx}" y2="{base_y + 4}" '
                'stroke="#999" stroke-width="1"/>'
            )
            continue
        # Digit centered at cx so octave dots / beams (which also reference cx)
        # align to the digit. The tail (" -", " - - -", " .") draws separately
        # starting just to the right so it doesn't shift the digit's center.
        digit_label = escape(f"{c.accidental}{c.body}")
        parts.append(
            f'<text x="{cx}" y="{base_y}" font-size="22" text-anchor="middle" '
            f'fill="#111" font-family="sans-serif" font-weight="600">{digit_label}</text>'
        )
        if c.tail:
            parts.append(
                f'<text x="{cx + 9}" y="{base_y}" font-size="22" text-anchor="start" '
                f'fill="#111" font-family="sans-serif" font-weight="600">'
                f"{escape(c.tail)}</text>"
            )
        for k in range(c.dots_up):
            parts.append(
                f'<circle cx="{cx}" cy="{base_y - 24 - k * 5}" r="{dot_r}" fill="#111"/>'
            )
        for k in range(c.dots_down):
            parts.append(
                f'<circle cx="{cx}" cy="{base_y + 8 + c.underlines * 4 + k * 5}" '
                f'r="{dot_r}" fill="#111"/>'
            )
        # rhythm beams: continuous to the next note when same-beat, else a stub.
        # Layer 1 (8th, longest/continuous) sits closest to the digit; extra
        # beams (16th, 32nd) stack further below.
        for layer in range(1, c.underlines + 1):
            uy = base_y + 5 + (layer - 1) * 4
            # Extend to the next note's center when beamed; otherwise a short
            # centered stub. An incoming beam from the left already reaches this
            # center, so the stub overlaps it seamlessly — no special-casing.
            x2 = centers[i + 1] if _beam_extends_right(cells, i, layer) else cx + 8
            parts.append(
                f'<line x1="{cx - 8}" y1="{uy}" x2="{x2}" y2="{uy}" '
                'stroke="#111" stroke-width="1.3"/>'
            )

    width = max(pad_x + sum(widths) + pad_x, 120)
    height = base_y + 30
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        f'{"".join(parts)}</svg>'
    )
