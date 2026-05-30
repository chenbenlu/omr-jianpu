"""Reconstruct a music21 Stream from predicted token tuples and engrave it.

The four decoupled streams the model emits are decoded (by `src.postproc`) into
`TokenTuple`s; this module inverts the generator
([src/data/generator.py](src/data/generator.py)) to rebuild a
`music21.stream.Stream`, then engraves it to a PNG.

Two backends:
- **lilypond** (primary, professional engraving) — used when the `lilypond`
  binary is on PATH.
- **verovio** (fallback) — reuses the project's existing
  `StaffRenderer` (`src.data.renderer`), so engraving always works even without
  lilypond installed.
"""

from __future__ import annotations

import io
import shutil
import tempfile
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from music21 import clef, duration, key, meter, note, stream
from PIL import Image

from src.data.generator import _token_to_duration
from src.postproc import TokenTuple

Backend = Literal["auto", "lilypond", "verovio"]

_SKIP = frozenset({None, "<UNK>"})


def _pitch_token_to_music21(tok: str) -> str:
    """`"F#5"`/`"Bb4"`/`"Bbb4"` → music21 name (`bb`→`--`, `b`→`-`)."""
    return tok.replace("bb", "--").replace("b", "-")


def tuples_to_stream(tuples: Sequence[TokenTuple]) -> stream.Score:
    """Inverse of the generator: predicted tuples → a structured `Score`.

    Builds an explicit `Part` of `Measure`s split on `barline` tokens, rather
    than a flat Stream with loose barlines. The flat form forced music21's
    exporter to *guess* measure boundaries, which could emit MusicXML that
    verovio rejects ("failed to parse the MusicXML payload"). Explicit measures
    produce well-formed output regardless of environment.

    Holes in predictions (`None`/`<UNK>` pitch or rhythm) are skipped rather
    than raising, so a partially-wrong prediction still engraves what it can.
    """
    part = stream.Part()
    measures: list[stream.Measure] = []
    cur = stream.Measure()

    def _has_content(m: stream.Measure) -> bool:
        return len(m) > 0

    for type_tok, pitch_tok, rhythm_tok, attribute_tok in tuples:
        if type_tok == "clef":
            if attribute_tok not in _SKIP:
                try:
                    cur.append(clef.clefFromString(attribute_tok))
                except Exception:
                    pass
        elif type_tok == "key_signature":
            if attribute_tok not in _SKIP and attribute_tok.startswith("ks"):
                try:
                    cur.append(key.KeySignature(int(attribute_tok[2:])))
                except ValueError:
                    pass
        elif type_tok == "time_signature":
            if attribute_tok not in _SKIP:
                try:
                    cur.append(meter.TimeSignature(attribute_tok))
                except (ValueError, meter.MeterException):
                    pass
        elif type_tok == "barline":
            # Close the current measure and start a fresh one.
            measures.append(cur)
            cur = stream.Measure()
        elif type_tok in ("note", "rest"):
            if rhythm_tok in _SKIP:
                continue
            try:
                dur = _token_to_duration(rhythm_tok)
            except duration.DurationException:
                continue
            if type_tok == "rest":
                cur.append(note.Rest(duration=dur))
            elif pitch_tok not in _SKIP:
                try:
                    cur.append(
                        note.Note(_pitch_token_to_music21(pitch_tok), duration=dur)
                    )
                except Exception:
                    continue

    if _has_content(cur):
        measures.append(cur)
    for m in measures:
        part.append(m)
    score = stream.Score()
    score.append(part)
    return score


def lilypond_available() -> bool:
    return shutil.which("lilypond") is not None


def which_backend(backend: Backend = "auto") -> str:
    if backend == "auto":
        return "lilypond" if lilypond_available() else "verovio"
    return backend


def _ensure_verovio_resources() -> None:
    """Point verovio at its bundled font/glyph data.

    verovio needs the Bravura/Leipzig music fonts shipped in its package `data`
    dir. Its auto-detection of that path can fail in some host processes (e.g.
    under Streamlit), giving "Bravura font could not be loaded" → `loadData`
    returns False → "failed to parse the MusicXML payload". Setting the global
    default resource path explicitly fixes every later-built toolkit, including
    the one inside the frozen `StaffRenderer`. Idempotent and cheap.
    """
    import os

    import verovio

    data_dir = os.path.join(os.path.dirname(verovio.__file__), "data")
    if os.path.isdir(data_dir):
        verovio.setDefaultResourcePath(data_dir)


def _render_verovio(s: stream.Stream) -> np.ndarray:
    from src.data.renderer import RenderConfig, StaffRenderer

    _ensure_verovio_resources()
    # A fresh StaffRenderer per call builds a clean verovio toolkit (the
    # SWIG-wrapped C++ object carries state from the previous loadData; reusing
    # one across renders in a long-lived process can wedge the parser).
    renderer = StaffRenderer(RenderConfig())
    try:
        return renderer.render(s)
    except Exception:
        # Normalize (auto-bar, fix tie/duration overflow) and retry once — this
        # turns a borderline Stream into MusicXML verovio reliably accepts.
        normalized = s.makeNotation()
        return StaffRenderer(RenderConfig()).render(normalized)


def _render_lilypond(s: stream.Stream) -> np.ndarray:
    # music21's LilypondConverter shells out to the binary at construction, so
    # this raises LilyTranslateException if lilypond is missing — caller guards.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "score"
        written = Path(s.write("lilypond.png", fp=str(out)))
        img = Image.open(io.BytesIO(written.read_bytes()))
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        return np.asarray(
            Image.alpha_composite(white, img.convert("RGBA")).convert("RGB"),
            dtype=np.uint8,
        )


def render_staff_png(
    tuples: Sequence[TokenTuple], backend: Backend = "auto"
) -> tuple[np.ndarray, str]:
    """Engrave the predicted music to an RGB image.

    Returns ``(image, backend_used)``. Honors an explicit ``"verovio"`` /
    ``"lilypond"`` request; ``"auto"`` prefers lilypond when installed. A
    lilypond failure (binary missing, render error) falls back to verovio so
    the demo never crashes.
    """
    s = tuples_to_stream(tuples)
    chosen = which_backend(backend)
    if chosen == "lilypond":
        try:
            return _render_lilypond(s), "lilypond"
        except Exception:
            if backend == "lilypond":
                raise
            chosen = "verovio"
    return _render_verovio(s), "verovio"
