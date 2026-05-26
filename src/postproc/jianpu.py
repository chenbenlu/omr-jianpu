"""Decoupled token tuples → Jianpu (numbered notation) text.

State-machine renderer: tracks the running clef and key signature as
``clef`` / ``key_signature`` / ``time_signature`` tokens appear in the
stream, and uses them to map each ``note`` token's letter+accidental+octave
to a Jianpu scale degree, chromatic alteration, and octave-dot count.

The text format is deliberately ASCII so the Streamlit UI can render it
in a monospace block and so it grep's easily in logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.data.vocabulary import VocabBundle
from src.postproc.decode import TokenTuple, ids_to_tuples

_LETTER_PC: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}
_LETTER_IDX: dict[str, int] = {
    "C": 0,
    "D": 1,
    "E": 2,
    "F": 3,
    "G": 4,
    "A": 5,
    "B": 6,
}
_ACC_TO_SEMITONES: dict[str, int] = {
    "bb": -2,
    "b": -1,
    "": 0,
    "#": 1,
    "##": 2,
}
# Major-scale semitone offsets from tonic for scale degrees 1..7.
_DIATONIC_OFFSETS: tuple[int, ...] = (0, 2, 4, 5, 7, 9, 11)

# (tonic_letter, accidental_semitones) for fifths in [-7, +7].
# Circle of fifths: ..., Cb, Gb, Db, Ab, Eb, Bb, F, C, G, D, A, E, B, F#, C#.
_TONIC_BY_FIFTHS: tuple[tuple[str, int], ...] = (
    ("C", -1),
    ("G", -1),
    ("D", -1),
    ("A", -1),
    ("E", -1),
    ("B", -1),
    ("F", 0),
    ("C", 0),
    ("G", 0),
    ("D", 0),
    ("A", 0),
    ("E", 0),
    ("B", 0),
    ("F", 1),
    ("C", 1),
)


def _tonic_for_fifths(fifths: int) -> tuple[str, int]:
    if not -7 <= fifths <= 7:
        raise ValueError(f"fifths must be in [-7, 7], got {fifths}")
    return _TONIC_BY_FIFTHS[fifths + 7]


def _key_name(fifths: int) -> str:
    letter, acc = _tonic_for_fifths(fifths)
    sigil = {-1: "b", 0: "", 1: "#"}[acc]
    return f"{letter}{sigil} major"


@dataclass(frozen=True)
class JianpuRenderConfig:
    octave_up: str = "'"
    octave_down: str = ","
    barline: str = " | "
    rest: str = "0"
    sep: str = " "
    unk: str = "?"
    emit_header: bool = True
    treble_central_octave: int = 4
    bass_central_octave: int = 3


_BASS_CLEFS: frozenset[str] = frozenset({"F4"})


def _parse_pitch_token(tok: str) -> tuple[str, int, int]:
    """Parse e.g. ``"F#5"`` → ``("F", 1, 5)``."""
    if len(tok) < 2:
        raise ValueError(f"malformed pitch token: {tok!r}")
    letter = tok[0]
    octave = int(tok[-1])
    acc = tok[1:-1]
    if letter not in _LETTER_PC:
        raise ValueError(f"bad letter {letter!r} in {tok!r}")
    if acc not in _ACC_TO_SEMITONES:
        raise ValueError(f"bad accidental {acc!r} in {tok!r}")
    return letter, _ACC_TO_SEMITONES[acc], octave


def _pitch_to_components(
    letter: str,
    acc_semitones: int,
    octave: int,
    fifths: int,
    central_octave: int,
) -> tuple[int, int, int]:
    """Map a parsed pitch to ``(degree, chromatic_alteration, octave_offset)``.

    ``degree`` is 1..7. ``chromatic_alteration`` is the signed semitone
    delta from the diatonic note for that degree in the current key, in
    range -2..+2 (clamped if a token spells something even weirder).
    ``octave_offset`` is the count of octave dots: positive = up, negative
    = down.
    """
    tonic_letter, tonic_acc = _tonic_for_fifths(fifths)
    tonic_pc = (_LETTER_PC[tonic_letter] + tonic_acc) % 12

    letter_idx = _LETTER_IDX[letter]
    tonic_letter_idx = _LETTER_IDX[tonic_letter]
    degree = (letter_idx - tonic_letter_idx) % 7 + 1

    expected_pc = (tonic_pc + _DIATONIC_OFFSETS[degree - 1]) % 12
    actual_pc = (_LETTER_PC[letter] + acc_semitones) % 12
    diff = (actual_pc - expected_pc) % 12
    if diff > 6:
        diff -= 12
    diff = max(-2, min(2, diff))

    position = letter_idx + 7 * octave
    tonic_position = tonic_letter_idx + 7 * central_octave
    octave_offset = (position - tonic_position) // 7

    return degree, diff, octave_offset


_RHYTHM_TAIL: dict[str, str] = {
    "whole": " - - -",
    "half": " -",
    "quarter": "",
    "eighth": "",
    "16th": "",
    "32nd": "",
}
_RHYTHM_UNDERLINES: dict[str, str] = {
    "whole": "",
    "half": "",
    "quarter": "",
    "eighth": "_",
    "16th": "__",
    "32nd": "___",
}


def _accidental_str(diff: int) -> str:
    return {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}[diff]


def _octave_marks(offset: int, cfg: JianpuRenderConfig) -> str:
    if offset > 0:
        return cfg.octave_up * offset
    if offset < 0:
        return cfg.octave_down * (-offset)
    return ""


@dataclass
class _State:
    fifths: int = 0
    central_octave: int = 4
    clef_known: bool = False


def _render_note_or_rest(
    is_rest: bool,
    pitch: str | None,
    rhythm: str | None,
    state: _State,
    cfg: JianpuRenderConfig,
) -> str:
    rhythm_is_unk = rhythm == "<UNK>"
    rhythm_is_null = rhythm is None
    has_dot = isinstance(rhythm, str) and not rhythm_is_unk and rhythm.endswith("_dot")
    base_rhythm = (
        rhythm[:-4]
        if has_dot
        else (rhythm if isinstance(rhythm, str) and not rhythm_is_unk else None)
    )

    if rhythm_is_unk or rhythm_is_null or base_rhythm not in _RHYTHM_TAIL:
        underline = ""
        tail = ""
    else:
        underline = _RHYTHM_UNDERLINES[base_rhythm]
        tail = _RHYTHM_TAIL[base_rhythm]
        if has_dot:
            tail = (tail + " .") if tail else " ."

    if is_rest:
        body = cfg.rest
        octave_str = ""
        accidental = ""
    else:
        if pitch == "<UNK>" or pitch is None:
            body = cfg.unk
            octave_str = ""
            accidental = ""
        else:
            try:
                letter, acc_semitones, octave = _parse_pitch_token(pitch)
            except ValueError:
                body = cfg.unk
                octave_str = ""
                accidental = ""
            else:
                degree, diff, octave_offset = _pitch_to_components(
                    letter,
                    acc_semitones,
                    octave,
                    state.fifths,
                    state.central_octave,
                )
                body = str(degree)
                octave_str = _octave_marks(octave_offset, cfg)
                accidental = _accidental_str(diff)

    if rhythm_is_unk:
        return f"{accidental}{body}{octave_str}{cfg.unk}"
    return f"{underline}{accidental}{body}{octave_str}{tail}"


def _header_for(name: str, value: str | None, cfg: JianpuRenderConfig) -> str:
    display = cfg.unk if value is None or value == "<UNK>" else value
    return f"[{name}: {display}]"


def tuples_to_jianpu(
    tuples: Sequence[TokenTuple],
    config: JianpuRenderConfig | None = None,
) -> str:
    cfg = config or JianpuRenderConfig()
    state = _State(
        fifths=0,
        central_octave=cfg.treble_central_octave,
        clef_known=False,
    )

    body_parts: list[str] = []
    headers: list[str] = []
    seen_header: dict[str, bool] = {"clef": False, "key": False, "time": False}

    for type_tok, pitch_tok, rhythm_tok, attribute_tok in tuples:
        if type_tok == "<UNK>":
            body_parts.append(cfg.unk)
            continue

        if type_tok == "note":
            body_parts.append(
                _render_note_or_rest(False, pitch_tok, rhythm_tok, state, cfg)
            )
            continue

        if type_tok == "rest":
            body_parts.append(_render_note_or_rest(True, None, rhythm_tok, state, cfg))
            continue

        if type_tok == "barline":
            body_parts.append(cfg.barline.strip())
            continue

        if type_tok == "clef":
            if attribute_tok in (None, "<UNK>"):
                if cfg.emit_header and not seen_header["clef"]:
                    headers.append(_header_for("Clef", attribute_tok, cfg))
                    seen_header["clef"] = True
                continue
            state.clef_known = True
            state.central_octave = (
                cfg.bass_central_octave
                if attribute_tok in _BASS_CLEFS
                else cfg.treble_central_octave
            )
            if cfg.emit_header and not seen_header["clef"]:
                headers.append(_header_for("Clef", attribute_tok, cfg))
                seen_header["clef"] = True
            continue

        if type_tok == "key_signature":
            if attribute_tok in (None, "<UNK>"):
                if cfg.emit_header and not seen_header["key"]:
                    headers.append(_header_for("Key", attribute_tok, cfg))
                    seen_header["key"] = True
                continue
            # Attribute token format: "ks+3" / "ks-2" / "ks+0".
            try:
                fifths = int(attribute_tok[2:])
            except (TypeError, ValueError):
                if cfg.emit_header and not seen_header["key"]:
                    headers.append(_header_for("Key", attribute_tok, cfg))
                    seen_header["key"] = True
                continue
            state.fifths = fifths
            if cfg.emit_header and not seen_header["key"]:
                headers.append(_header_for("Key", _key_name(fifths), cfg))
                seen_header["key"] = True
            continue

        if type_tok == "time_signature":
            if cfg.emit_header and not seen_header["time"]:
                headers.append(_header_for("Time", attribute_tok, cfg))
                seen_header["time"] = True
            continue

        body_parts.append(cfg.unk)

    body = cfg.sep.join(p for p in body_parts if p)
    if cfg.emit_header and headers:
        return " ".join(headers) + "\n" + body
    return body


def ids_to_jianpu(
    type_ids: Sequence[int],
    pitch_ids: Sequence[int],
    rhythm_ids: Sequence[int],
    attribute_ids: Sequence[int],
    vocab: VocabBundle,
    config: JianpuRenderConfig | None = None,
) -> str:
    tuples = ids_to_tuples(
        type_ids, pitch_ids, rhythm_ids, attribute_ids, vocab, strict=False
    )
    return tuples_to_jianpu(tuples, config)
